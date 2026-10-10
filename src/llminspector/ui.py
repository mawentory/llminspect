"""Local form for pelican drawings and context-marker probes.

The API key stays in the job object and is never written to a response.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llminspector.middle_drop import (
    ConfigError,
    MiddleDropOptions,
    build_client,
    build_prompt,
    redact,
    resolve_base_url,
    run_middle_drop,
    tokens_per_second,
)
from llminspector.pelican import Gate, PelicanOptions, run_pelican, svg_is_embeddable

PAGE_PATH = Path(__file__).with_name("ui_page.html")
MAX_RUNS = 100
MAX_CONCURRENCY = 32
MAX_CONTEXT_CHARS = 2_000_000
MARKER_PUBLIC = (
    "i",
    "model",
    "reasoning_effort",
    "http",
    "elapsed_s",
    "verdict_class",
    "verdict",
    "prompt_tokens",
    "input_tokens",
    "reasoning_tokens",
    "completion_tokens",
    "finish_reason",
    "timed_out",
    "error_body",
    "ttft_s",
    "tok_s",
    "cached_tokens",
    "output_tokens",
)
_CALL_FIELDS = (
    "state",
    "http",
    "error",
    "attempt",
    "ttft_s",
    "elapsed_s",
    "tok_s",
    "prompt_tokens",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
)
_IN_FLIGHT = {"connecting", "streaming", "http"}


@dataclass
class RunConfig:
    api_key: str
    base_url: str
    model: str
    effort: str
    pelican: bool
    marker: bool
    pelican_runs: int
    marker_runs: int
    concurrency: int
    context_chars: int


class Stopped(Exception):
    """The user stopped this run."""


class UIApp:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active: str | None = None
        self.lock = threading.Lock()

    def start(self, config: RunConfig) -> str | None:
        with self.lock:
            if self.active is not None:
                return None
            job_id = uuid.uuid4().hex
            job: dict[str, Any] = {
                "id": job_id,
                "status": "queued",
                "phase": "",
                "error": None,
                "log": [],
                "pelican": [],
                "marker": None,
                "calls": [],
                "concurrency": config.concurrency,
                "model": config.model,
                "base_url": config.base_url,
                "api_key": config.api_key,
                "config": config,
                "cancel": threading.Event(),
                "clients": [],
                "gates": [],
            }
            self.jobs[job_id] = job
            self.active = job_id
        thread = threading.Thread(target=self._execute, args=(job_id,), name=f"ui-{job_id[:8]}")
        thread.daemon = True
        thread.start()
        return job_id

    def stop(self, job_id: str) -> bool | None:
        """Stop one run and free the form immediately.

        None means the id is unknown. False means it already finished.
        Socket closes happen in the background so a stuck connection cannot
        keep the next run from starting.
        """
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job["status"] not in {"queued", "running", "stopping"}:
                return False
            job["cancel"].set()
            job["status"] = "stopped"
            job["phase"] = "stopped"
            job["error"] = None
            _freeze_stopped(job)
            clients = list(job["clients"])
            gates = list(job["gates"])
            if self.active == job_id:
                self.active = None
        for gate in gates:
            with gate.cv:
                gate.cv.notify_all()
        if clients:
            threading.Thread(
                target=_close_clients,
                args=(clients,),
                name="ui-close",
                daemon=True,
            ).start()
        return True

    def view(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return _public_view(job)

    def list_views(self) -> list[dict[str, Any]]:
        with self.lock:
            return [_public_view(job) for job in reversed(list(self.jobs.values()))]

    def _note(self, job: dict[str, Any], line: str) -> None:
        secret = str(job.get("api_key") or "")
        cleaned = redact(line, secret).rstrip()
        if not cleaned:
            return
        with self.lock:
            log = job["log"]
            log.append(cleaned)
            del log[:-80]

    def _execute(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            config: RunConfig = job["config"]
            if not job["cancel"].is_set():
                job["status"] = "running"
        errors: list[str] = []
        try:
            if config.pelican and not job["cancel"].is_set():
                try:
                    self._run_pelican(job, config)
                except Stopped:
                    pass
                except Exception as exc:  # noqa: BLE001
                    if not job["cancel"].is_set():
                        errors.append(self._fail(job, exc))
            if config.marker and not job["cancel"].is_set():
                try:
                    self._run_marker(job, config)
                except Stopped:
                    pass
                except Exception as exc:  # noqa: BLE001
                    if not job["cancel"].is_set():
                        errors.append(self._fail(job, exc))
            with self.lock:
                if not job["cancel"].is_set():
                    job["phase"] = "done"
                    if errors and not job["pelican"] and job["marker"] is None:
                        job["status"] = "error"
                        job["error"] = "; ".join(errors)
                    else:
                        job["status"] = "done"
                        job["error"] = "; ".join(errors) if errors else None
        except Exception as exc:  # noqa: BLE001
            if not job["cancel"].is_set():
                self._fail(job, exc)
                with self.lock:
                    job["status"] = "error"
                    job["error"] = redact(str(exc), str(job.get("api_key") or ""))
        finally:
            note_stop = False
            with self.lock:
                if job["cancel"].is_set():
                    job["status"] = "stopped"
                    job["phase"] = "stopped"
                    job["error"] = None
                    _freeze_stopped(job)
                    note_stop = not job.get("_stop_noted")
                    job["_stop_noted"] = True
                job.pop("api_key", None)
                job.pop("config", None)
                if self.active == job_id:
                    self.active = None
            if note_stop:
                self._note(job, "stopped")

    def _open_client(self, job: dict[str, Any]) -> Any:
        client = build_client()
        with self.lock:
            job["clients"].append(client)
            should_close = job["cancel"].is_set()
        if should_close:
            try:
                client.close()
            except Exception:  # noqa: BLE001, S110
                pass
        return client

    def _close_client(self, job: dict[str, Any], client: Any) -> None:
        try:
            client.close()
        except Exception:  # noqa: BLE001, S110
            pass
        with self.lock:
            clients: list[Any] = job["clients"]
            if client in clients:
                clients.remove(client)

    def _pause(self, job: dict[str, Any], seconds: float) -> None:
        job["cancel"].wait(seconds)

    def _run_pelican(self, job: dict[str, Any], config: RunConfig) -> None:
        with self.lock:
            job["phase"] = "pelican"
        self._seed_calls(job, "pelican", config.pelican_runs)
        if job["cancel"].is_set():
            raise Stopped()
        out = Path(tempfile.mkdtemp(prefix="llminspect-pelican-"))
        options = PelicanOptions(
            base_url=config.base_url,
            api_key=config.api_key,
            models=[config.model],
            efforts=[config.effort],
            iterations=config.pelican_runs,
            timeout=900.0,
            max_tokens=None,
            concurrency=config.concurrency,
            max_attempts=8,
            out=out,
        )
        client = self._open_client(job)
        try:
            samples = run_pelican(
                options,
                client=client,
                stdout=_LogStream(self, job),
                on_event=lambda event: self._live(job, "pelican", event),
                sleep=lambda seconds: self._pause(job, seconds),
                cancel=job["cancel"],
            )
        finally:
            self._close_client(job, client)
        public = [_pelican_public(sample.__dict__) for sample in samples]
        with self.lock:
            job["pelican"] = public
        if job["cancel"].is_set():
            raise Stopped()

    def _fail(self, job: dict[str, Any], exc: BaseException) -> str:
        secret = str(job.get("api_key") or "")
        message = redact(f"{type(exc).__name__}: {exc}", secret)
        self._note(job, message)
        return message

    def _run_marker(self, job: dict[str, Any], config: RunConfig) -> None:
        prompt_chars = len(build_prompt(config.context_chars))
        with self.lock:
            job["phase"] = "marker"
            job["prompt_chars"] = prompt_chars
            job["marker"] = _marker_report([], config.context_chars, prompt_chars, finished=False)
        self._seed_calls(job, "marker", config.marker_runs)
        records = self._marker_batch(job, config, range(1, config.marker_runs + 1))
        records.sort(key=lambda row: int(row.get("i") or 0))
        report = _marker_report(records, config.context_chars, prompt_chars, finished=True)
        self._note(job, report["summary"])
        with self.lock:
            job["marker"] = report

    def _marker_batch(
        self,
        job: dict[str, Any],
        config: RunConfig,
        indexes: range,
    ) -> list[dict[str, Any]]:
        gate = Gate(config.concurrency)
        with self.lock:
            job["gates"].append(gate)
        found: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            try:
                record = self._marker_once(job, config, index, min_interval=0.0, gate=gate)
            except Stopped:
                self._live(job, "marker", {"index": index, "state": "stopped", "error": "stopped"})
                return
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)
                self._live(
                    job,
                    "marker",
                    {
                        "index": index,
                        "state": "error",
                        "error": redact(f"{type(exc).__name__}: {exc}", config.api_key),
                    },
                )
                return
            with lock:
                found.append(record)
            self._upsert_marker(job, record, config)

        threads = [threading.Thread(target=worker, args=(index,)) for index in indexes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors and not found:
            raise errors[0]
        for exc in errors:
            self._note(job, f"{type(exc).__name__}: {exc}")
        return found

    def _marker_once(
        self,
        job: dict[str, Any],
        config: RunConfig,
        index: int,
        *,
        min_interval: float,
        gate: Gate | None = None,
    ) -> dict[str, Any]:
        secret = config.api_key
        last: dict[str, Any] | None = None
        for attempt in range(1, 9):
            if job["cancel"].is_set():
                raise Stopped()
            held = False
            if gate is not None:
                held = gate.acquire(job["cancel"])
                if not held:
                    raise Stopped()
            client = self._open_client(job)
            try:
                out = Path(tempfile.mkdtemp(prefix="llminspect-marker-")) / f"{index}.jsonl"
                options = MiddleDropOptions(
                    base_url=config.base_url,
                    api_key=secret,
                    models=[config.model],
                    efforts=[config.effort],
                    iterations=1,
                    filler_chars=config.context_chars,
                    timeout=900.0,
                    max_tokens=None,
                    min_interval=min_interval,
                    out=out,
                )
                buffer = io.StringIO()

                def on_event(
                    event: dict[str, Any], index: int = index, attempt: int = attempt
                ) -> None:
                    self._live(job, "marker", {**event, "index": index, "attempt": attempt})

                try:
                    records = run_middle_drop(
                        options,
                        client=client,
                        stdout=buffer,
                        on_event=on_event,
                        sleep=lambda seconds: self._pause(job, seconds),
                        cancel=job["cancel"],
                    )
                except ConfigError as exc:
                    self._note(job, str(exc))
                    raise
            finally:
                self._close_client(job, client)
                if held and gate is not None:
                    gate.release()
            if job["cancel"].is_set():
                raise Stopped()
            for line in buffer.getvalue().splitlines():
                self._note(job, line)
            last = _marker_public(
                records[0] if records else {"i": index, "verdict_class": "HTTP_ERROR"}
            )
            last["i"] = index
            if last.get("http") == 429 and gate is not None and attempt < 8:
                self._note(job, f"marker run {index} busy, slowing down")
                self._live(
                    job,
                    "marker",
                    {
                        "index": index,
                        "attempt": attempt,
                        "state": "retry",
                        "http": 429,
                        "error": last.get("error_body") or "HTTP 429",
                        "concurrency": max(1, gate.cap - 1) if gate.cap > 1 else 1,
                    },
                )
                gate.slow_down(lambda seconds: self._pause(job, seconds))
                if job["cancel"].is_set():
                    raise Stopped()
                self._live(job, "marker", {"index": index, "concurrency": gate.cap})
                continue
            failed = last.get("verdict_class") == "HTTP_ERROR" or bool(last.get("error_body"))
            self._live(
                job,
                "marker",
                {
                    "index": index,
                    "attempt": attempt,
                    "state": "error" if failed else "done",
                    "http": last.get("http"),
                    "error": last.get("error_body"),
                    "ttft_s": last.get("ttft_s"),
                    "elapsed_s": last.get("elapsed_s"),
                    "tok_s": last.get("tok_s"),
                    "prompt_tokens": last.get("prompt_tokens"),
                    "input_tokens": last.get("input_tokens"),
                    "output_tokens": last.get("output_tokens") or last.get("completion_tokens"),
                    "cached_tokens": last.get("cached_tokens"),
                },
            )
            return last
        assert last is not None
        return last

    def _seed_calls(self, job: dict[str, Any], kind: str, count: int) -> None:
        with self.lock:
            for index in range(1, count + 1):
                if _find_call(job["calls"], kind, index) is None:
                    job["calls"].append(_blank_call(kind, index))

    def _live(self, job: dict[str, Any], kind: str, event: dict[str, Any]) -> None:
        secret = str(job.get("api_key") or "")
        payload = dict(event)
        if payload.get("error"):
            payload["error"] = redact(str(payload["error"]), secret)
        index = int(payload.get("index") or 0)
        note = ""
        with self.lock:
            if payload.get("concurrency"):
                job["concurrency"] = payload["concurrency"]
            row = _find_call(job["calls"], kind, index)
            if row is None:
                row = _blank_call(kind, index)
                job["calls"].append(row)
            if job.get("cancel") is not None and job["cancel"].is_set():
                if row.get("state") in {"done", "stopped"} and payload.get("state") not in {
                    "done",
                    "stopped",
                }:
                    return
                if payload.get("state") not in {"done", "stopped", "error"}:
                    payload["state"] = "stopped"
                    payload["error"] = payload.get("error") or "stopped"
            if row.get("_t0") is None and payload.get("state") in {
                "connecting",
                "streaming",
                "http",
            }:
                row["_t0"] = time.monotonic()
            for key in _CALL_FIELDS:
                if key in payload and (payload[key] is not None or key == "error"):
                    row[key] = payload[key]
                    if key == "elapsed_s":
                        row["_elapsed_at"] = time.monotonic()
            if payload.get("state") in {"connecting", "streaming", "http"} and "error" not in payload:
                row["error"] = None
            if payload.get("state") in {"done", "stopped"}:
                row["error"] = payload.get("error")
            if kind == "pelican" and "ok" in payload:
                _upsert_pelican(
                    job,
                    _pelican_public(
                        {
                            "name": payload.get("name"),
                            "model": payload.get("model"),
                            "effort": payload.get("effort"),
                            "index": index,
                            "ok": payload.get("ok"),
                            "svg": payload.get("svg"),
                            "http": payload.get("http"),
                            "attempts": payload.get("attempts", payload.get("attempt")),
                            "error": payload.get("error"),
                            "ttft_s": payload.get("ttft_s"),
                            "elapsed_s": payload.get("elapsed_s"),
                            "tok_s": payload.get("tok_s"),
                            "prompt_tokens": payload.get("prompt_tokens"),
                            "input_tokens": payload.get("input_tokens"),
                            "output_tokens": payload.get("output_tokens"),
                            "reasoning_tokens": payload.get("reasoning_tokens"),
                            "cached_tokens": payload.get("cached_tokens"),
                        }
                    ),
                )
            state = payload.get("state")
            if state in {"error", "retry"} and payload.get("error"):
                note = f"{kind} {index} {payload['error']}"
        if note:
            self._note(job, note)

    def _upsert_marker(
        self, job: dict[str, Any], record: dict[str, Any], config: RunConfig
    ) -> None:
        with self.lock:
            marker = job.get("marker") or {"summary": "", "records": []}
            records = list(marker.get("records") or [])
            replaced = False
            for slot, existing in enumerate(records):
                if existing.get("i") == record.get("i"):
                    records[slot] = record
                    replaced = True
                    break
            if not replaced:
                records.append(record)
            records.sort(key=lambda row: int(row.get("i") or 0))
            prompt_chars = int(job.get("prompt_chars") or len(build_prompt(config.context_chars)))
            job["marker"] = _marker_report(
                records,
                config.context_chars,
                prompt_chars,
                finished=False,
            )


class _LogStream(io.TextIOBase):
    def __init__(self, app: UIApp, job: dict[str, Any]) -> None:
        self.app = app
        self.job = job

    def write(self, text: str) -> int:
        for line in text.splitlines():
            self.app._note(self.job, line)
        return len(text)


def _pelican_public(sample: dict[str, Any]) -> dict[str, Any]:
    svg = sample.get("svg")
    embed = bool(sample.get("ok") and isinstance(svg, str) and svg_is_embeddable(svg))
    return {
        "name": sample.get("name"),
        "model": sample.get("model"),
        "effort": sample.get("effort"),
        "index": sample.get("index"),
        "ok": bool(sample.get("ok")),
        "http": sample.get("http"),
        "attempts": sample.get("attempts"),
        "error": sample.get("error"),
        "embed": embed,
        "svg": svg if embed else None,
        "ttft_s": sample.get("ttft_s"),
        "elapsed_s": sample.get("elapsed_s"),
        "tok_s": sample.get("tok_s"),
        "prompt_tokens": sample.get("prompt_tokens"),
        "input_tokens": sample.get("input_tokens"),
        "output_tokens": sample.get("output_tokens"),
        "reasoning_tokens": sample.get("reasoning_tokens"),
        "cached_tokens": sample.get("cached_tokens"),
    }


def _find_call(calls: list[dict[str, Any]], kind: str, index: int) -> dict[str, Any] | None:
    for row in calls:
        if row.get("kind") == kind and row.get("index") == index:
            return row
    return None


def _blank_call(kind: str, index: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "index": index,
        "state": "queued",
        "http": None,
        "error": None,
        "attempt": None,
        "ttft_s": None,
        "elapsed_s": None,
        "tok_s": None,
        "prompt_tokens": None,
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
        "_t0": None,
    }


def _upsert_pelican(job: dict[str, Any], public: dict[str, Any]) -> None:
    rows = job["pelican"]
    for slot, existing in enumerate(rows):
        if existing.get("index") == public.get("index"):
            rows[slot] = public
            return
    rows.append(public)


def _freeze_stopped(job: dict[str, Any]) -> None:
    marker = job.get("marker")
    if isinstance(marker, dict) and marker.get("outcome") != "pass":
        records = list(marker.get("records") or [])
        config = job.get("config")
        context = int(config.context_chars) if config is not None else 0
        prompt_chars = int(job.get("prompt_chars") or 0)
        job["marker"] = _marker_report(
            records,
            context,
            prompt_chars,
            finished=False,
            stopped=True,
        )
    for row in job.get("calls") or []:
        if row.get("state") not in {"done", "error", "stopped"}:
            row["state"] = "stopped"
            if not row.get("error"):
                row["error"] = "stopped"


def _marker_report(
    records: list[dict[str, Any]],
    context_chars: int,
    prompt_chars: int,
    *,
    finished: bool,
    stopped: bool = False,
) -> dict[str, Any]:
    dropped = sum(1 for row in records if row.get("verdict_class") == "DROP")
    kept = sum(1 for row in records if row.get("verdict_class") == "PRESENT")
    other = len(records) - dropped - kept
    if stopped:
        outcome = "stopped"
        if kept == 1:
            headline = "Stopped. 1 run kept the middle marker."
        elif kept:
            headline = f"Stopped. {kept} runs kept the middle marker."
        elif records:
            headline = "Stopped. Results so far stay on the page."
        else:
            headline = "Stopped before a marker result came back."
    elif records and kept == len(records) and finished:
        outcome = "pass"
        headline = f"Passed. All {kept} runs kept the middle marker."
    elif records and kept == len(records):
        outcome = "running"
        headline = f"{kept} runs kept the middle marker so far."
    elif not records and not finished:
        outcome = "running"
        headline = "Waiting for marker results."
    elif not records:
        outcome = "fail"
        headline = "Failed. No marker result came back."
    else:
        outcome = "fail"
        parts: list[str] = []
        if dropped:
            parts.append(f"{dropped} dropped the middle")
        if kept:
            parts.append(f"{kept} kept it")
        if other:
            parts.append(f"{other} failed the call")
        headline = "Failed. " + ", ".join(parts) + "."
    summary = (
        f"{headline} context_chars={context_chars} prompt_chars={prompt_chars} "
        f"DROP={dropped} PRESENT={kept} other={other}"
    )
    return {
        "summary": summary,
        "headline": headline,
        "outcome": outcome,
        "records": records,
    }


def _public_call(row: dict[str, Any], now: float) -> dict[str, Any]:
    state = row.get("state")
    elapsed = row.get("elapsed_s")
    if state not in {"done", "error", "stopped"}:
        if row.get("_elapsed_at") is not None and isinstance(elapsed, (int, float)):
            elapsed = float(elapsed) + (now - float(row["_elapsed_at"]))
        elif row.get("_t0") is not None:
            elapsed = now - float(row["_t0"])
    elapsed_n = float(elapsed) if isinstance(elapsed, (int, float)) else 0.0
    tok_s = tokens_per_second(row.get("output_tokens"), row.get("ttft_s"), elapsed_n)
    if tok_s is None:
        tok_s = row.get("tok_s")
    return {
        "kind": row.get("kind"),
        "index": row.get("index"),
        "state": state,
        "http": row.get("http"),
        "error": row.get("error"),
        "attempt": row.get("attempt"),
        "ttft_s": row.get("ttft_s"),
        "elapsed_s": elapsed,
        "tok_s": tok_s,
        "prompt_tokens": row.get("prompt_tokens"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "reasoning_tokens": row.get("reasoning_tokens"),
        "cached_tokens": row.get("cached_tokens"),
    }


def _marker_public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in MARKER_PUBLIC}


def _public_view(job: dict[str, Any]) -> dict[str, Any]:
    now = time.monotonic()
    calls = [_public_call(row, now) for row in job.get("calls") or []]
    inflight = sum(1 for row in calls if row.get("state") in _IN_FLIGHT)
    return {
        "id": job["id"],
        "status": job["status"],
        "phase": job["phase"],
        "error": job["error"],
        "model": job.get("model"),
        "base_url": job.get("base_url"),
        "concurrency": job.get("concurrency"),
        "inflight": inflight,
        "log": list(job["log"]),
        "pelican": list(job["pelican"]),
        "marker": job["marker"],
        "calls": calls,
    }


def parse_run_config(data: dict[str, Any]) -> RunConfig:
    if not isinstance(data, dict):
        raise ConfigError("body must be an object")
    base_url = resolve_base_url(_text(data.get("base_url")), {})
    api_key = _text(data.get("api_key"))
    if not api_key:
        raise ConfigError("missing API key")
    model = _text(data.get("model"))
    if not model:
        raise ConfigError("missing model")
    pelican = bool(data.get("pelican"))
    marker = bool(data.get("marker"))
    if not pelican and not marker:
        raise ConfigError("choose pelican, marker, or both")
    effort = _text(data.get("effort")) or "max"
    return RunConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        effort=effort,
        pelican=pelican,
        marker=marker,
        pelican_runs=_bounded(data.get("pelican_runs"), 5, 1, MAX_RUNS, "pelican runs"),
        marker_runs=_bounded(data.get("marker_runs"), 5, 1, MAX_RUNS, "marker runs"),
        concurrency=_bounded(data.get("concurrency"), 4, 1, MAX_CONCURRENCY, "concurrency"),
        context_chars=_bounded(
            data.get("context_chars"), 423_588, 1, MAX_CONTEXT_CHARS, "context size"
        ),
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _bounded(value: Any, default: int, low: int, high: int, label: str) -> int:
    if value is None or value == "":
        number = default
    else:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must be an integer") from exc
    if number < low or number > high:
        raise ConfigError(f"{label} must be between {low} and {high}")
    return number


class UIServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: UIApp) -> None:
        self.app = app
        super().__init__(address, UIHandler)


class UIHandler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE_PATH.read_text(encoding="utf-8")
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._send_json(200, {"runs": self.server.app.list_views()})
            return
        if path.startswith("/api/runs/"):
            job_id = path.removeprefix("/api/runs/").strip("/")
            view = self.server.app.view(job_id)
            if view is None:
                self._send_json(404, {"error": "no such run"})
                return
            self._send_json(200, view)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/runs/") and path.endswith("/stop"):
            job_id = path.removeprefix("/api/runs/").removesuffix("/stop").strip("/")
            result = self.server.app.stop(job_id)
            if result is None:
                self._send_json(404, {"error": "no such run"})
            elif not result:
                self._send_json(409, {"error": "that run is not in progress"})
            else:
                self._send_json(202, {"id": job_id, "status": "stopped"})
            return
        if path != "/api/runs":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "bad content length"})
            return
        if length <= 0 or length > 1_000_000:
            self._send_json(400, {"error": "bad body"})
            return
        try:
            data = json.loads(self.rfile.read(length))
            config = parse_run_config(data)
        except (json.JSONDecodeError, ConfigError) as exc:
            message = exc.args[0] if isinstance(exc, ConfigError) else "body must be JSON"
            self._send_json(400, {"error": str(message)})
            return
        job_id = self.server.app.start(config)
        if job_id is None:
            self._send_json(409, {"error": "a run is already in progress"})
            return
        self._send_json(202, {"id": job_id})

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")


def _close_clients(clients: list[Any]) -> None:
    for client in clients:
        try:
            client.close()
        except Exception:  # noqa: BLE001, S112
            continue


def bind_ui(host: str = "127.0.0.1", port: int = 8787, app: UIApp | None = None) -> UIServer:
    return UIServer((host, port), app or UIApp())


def serve_ui(host: str = "127.0.0.1", port: int = 8787) -> int:
    try:
        httpd = bind_ui(host, port)
    except OSError as exc:
        print(f"cannot listen on {host}:{port}: {exc.strerror}")
        return 1
    bound_host, bound_port = httpd.server_address[:2]
    print(f"http://{bound_host}:{bound_port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0
