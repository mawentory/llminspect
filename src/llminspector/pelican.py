"""Ask a model for an SVG pelican and keep the drawings.

Each call is one chat completion. The prompt is fixed. No tools are sent.
max_tokens is omitted unless a positive cap is set. A 429 lowers the number
of in-flight calls by one, down to 1, waits, and retries that same call.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO

import httpx

from llminspector.middle_drop import (
    ConfigError,
    build_client,
    parse_sse,
    redact,
    stream_chat,
)

PROMPT = "Generate an SVG of a pelican riding a bicycle"
DEFAULT_EFFORT = "max"
DEFAULT_ITERATIONS = 5
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_ATTEMPTS = 8

_UNSAFE = ("<script", "onload=", "onclick=", "onerror=", "javascript:", "foreignobject")


@dataclass
class PelicanOptions:
    base_url: str
    api_key: str
    models: list[str]
    efforts: list[str]
    iterations: int
    timeout: float
    max_tokens: int | None
    concurrency: int
    max_attempts: int
    out: Path


@dataclass
class PelicanSample:
    name: str
    model: str
    effort: str
    index: int
    ok: bool
    svg: str | None
    http: int | None
    attempts: int
    error: str | None
    ttft_s: float | None = None
    elapsed_s: float | None = None
    tok_s: float | None = None
    prompt_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


class Gate:
    """In-flight cap that only shrinks. Waiters block until a slot opens."""

    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self.inflight = 0
        self.backoff = 2.0
        self.cv = threading.Condition()

    def acquire(self, cancel: threading.Event | None = None) -> bool:
        """Take one in-flight slot. Return False when cancel is set first."""
        with self.cv:
            while self.inflight >= self.cap:
                if cancel is not None and cancel.is_set():
                    return False
                self.cv.wait(timeout=0.25)
            if cancel is not None and cancel.is_set():
                return False
            self.inflight += 1
            return True

    def release(self) -> None:
        with self.cv:
            self.inflight -= 1
            self.cv.notify_all()

    def slow_down(self, sleep: Callable[[float], None]) -> None:
        with self.cv:
            if self.cap > 1:
                self.cap -= 1
            wait = self.backoff
            self.backoff = min(self.backoff * 2, 30.0)
            self.cv.notify_all()
        sleep(wait)


def request_body(model: str, effort: str | None, max_tokens: int | None) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if effort:
        payload["reasoning_effort"] = effort
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def extract_svg(text: str) -> str | None:
    if not text:
        return None
    lower = text.lower()
    start = lower.find("<svg")
    end = lower.rfind("</svg>")
    if start < 0 or end < start:
        return None
    return text[start : end + len("</svg>")]


def svg_is_embeddable(svg: str) -> bool:
    lower = svg.lower()
    return not any(token in lower for token in _UNSAFE)


def completion_text(raw: bytes) -> str:
    parsed = parse_sse(raw)
    if parsed["content"]:
        return parsed["content"]
    text = raw.decode("utf-8", errors="replace").strip()
    if not text.startswith("{"):
        return ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ""
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _busy(status: int | None, raw: bytes) -> bool:
    if status == 429:
        return True
    if status == 200:
        return False
    text = raw.decode("utf-8", errors="replace").lower()
    return "rate_limit" in text or "too many concurrent" in text


def _sample_name(model: str, effort: str, index: int) -> str:
    raw = f"{model}-{effort}-{index}"
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in raw)


def _check_options(options: PelicanOptions) -> None:
    if not options.models:
        raise ConfigError("missing model: pass --models")
    if not options.efforts:
        raise ConfigError("missing reasoning effort: pass --efforts")
    if options.iterations < 1:
        raise ConfigError("iterations must be at least 1")
    if options.concurrency < 1:
        raise ConfigError("concurrency must be at least 1")
    if options.max_attempts < 1:
        raise ConfigError("max attempts must be at least 1")
    if options.timeout <= 0:
        raise ConfigError("timeout must be positive")


def _metric_kwargs(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    return {
        "ttft_s": result.get("ttft_s"),
        "elapsed_s": result.get("elapsed_s"),
        "tok_s": result.get("tok_s"),
        "prompt_tokens": result.get("prompt_tokens"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "reasoning_tokens": result.get("reasoning_tokens"),
        "cached_tokens": result.get("cached_tokens"),
    }


def _one_call(
    options: PelicanOptions,
    client: httpx.Client,
    model: str,
    effort: str,
    index: int,
    sleep: Callable[[float], None],
    gate: Gate,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> PelicanSample:
    name = _sample_name(model, effort, index)
    body = request_body(model, effort, options.max_tokens)
    secret = options.api_key
    last_error = "no response"
    last_http: int | None = None
    last_result: dict[str, Any] | None = None

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(
                {
                    "index": index,
                    "model": model,
                    "effort": effort,
                    "attempt": event.get("attempt"),
                    **event,
                }
            )

    for attempt in range(1, options.max_attempts + 1):
        if cancel is not None and cancel.is_set():
            last_error = "stopped"
            break
        if not gate.acquire(cancel):
            last_error = "stopped"
            break
        emit({"state": "connecting", "attempt": attempt})
        try:
            try:
                result = stream_chat(
                    options.base_url,
                    secret,
                    body,
                    options.timeout,
                    client,
                    on_event=lambda event, attempt=attempt: emit({**event, "attempt": attempt}),
                    cancel=cancel,
                )
            except httpx.HTTPError as exc:
                if cancel is not None and cancel.is_set():
                    last_error = "stopped"
                    break
                last_error = redact(f"{type(exc).__name__}: {exc}", secret)
                last_http = None
                result = None
                emit({"state": "retry", "attempt": attempt, "error": last_error})
        finally:
            gate.release()
        if result is not None and result.get("stopped"):
            last_error = "stopped"
            last_result = result
            last_http = result.get("http")
            break
        if cancel is not None and cancel.is_set():
            last_error = "stopped"
            break
        if result is None:
            sleep(min(2**attempt, 30))
            if cancel is not None and cancel.is_set():
                last_error = "stopped"
                break
            continue
        last_result = result
        status = result["http"]
        raw = result["raw"]
        last_http = status
        if result.get("timed_out"):
            last_error = result.get("error") or "timed out"
            emit({"state": "retry", "attempt": attempt, "http": status, "error": last_error})
            sleep(min(2**attempt, 30))
            continue
        if _busy(status, raw):
            last_error = result.get("error") or "HTTP 429"
            emit(
                {
                    "state": "retry",
                    "attempt": attempt,
                    "http": status,
                    "error": last_error,
                    "concurrency": max(1, gate.cap - 1) if gate.cap > 1 else 1,
                }
            )
            gate.slow_down(sleep)
            emit({"concurrency": gate.cap, "attempt": attempt})
            continue
        if status == 200:
            svg = extract_svg(completion_text(raw))
            if svg:
                return PelicanSample(
                    name=name,
                    model=model,
                    effort=effort,
                    index=index,
                    ok=True,
                    svg=svg,
                    http=status,
                    attempts=attempt,
                    error=None,
                    **_metric_kwargs(result),
                )
            last_error = "response had no svg"
            emit({"state": "retry", "attempt": attempt, "http": status, "error": last_error})
        else:
            snippet = (
                result.get("error") or redact(raw.decode("utf-8", errors="replace"), secret)[:240]
            )
            last_error = snippet or f"HTTP {status}"
            emit({"state": "retry", "attempt": attempt, "http": status, "error": last_error})
        sleep(min(2**attempt, 30))
        if cancel is not None and cancel.is_set():
            last_error = "stopped"
            break
    return PelicanSample(
        name=name,
        model=model,
        effort=effort,
        index=index,
        ok=False,
        svg=None,
        http=last_http,
        attempts=options.max_attempts,
        error=redact(last_error, secret),
        **_metric_kwargs(last_result),
    )


def render_page(samples: list[PelicanSample], *, title: str) -> str:
    parts: list[str] = []
    for sample in samples:
        heading = f"{sample.model} effort={sample.effort} run {sample.index}"
        safe_heading = heading.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if sample.ok and sample.svg and svg_is_embeddable(sample.svg):
            body = sample.svg
        elif sample.ok:
            body = "<p>SVG was saved but not embedded.</p>"
        else:
            detail = sample.error or "failed"
            detail = detail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body = f"<p>{detail}</p>"
        parts.append(f"<section><h2>{safe_heading}</h2>{body}</section>")
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    sections = "\n".join(parts)
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 16px/1.4 sans-serif; }}
  header {{ padding: 16px 24px 8px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p {{ margin: 0; color: #aaa; }}
  section {{ padding: 16px 24px 32px; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  svg {{ display: block; width: auto; height: auto; max-width: none; background: #fff; }}
</style>
<header>
  <h1>{safe_title}</h1>
  <p>{PROMPT}. Scroll for every call.</p>
</header>
{sections}
"""


def write_gallery(out: Path, samples: list[PelicanSample], *, title: str, secret: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if sample.ok and sample.svg:
            (out / f"{sample.name}.svg").write_text(sample.svg, encoding="utf-8")
    page = redact(render_page(samples, title=title), secret)
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def run_pelican(
    options: PelicanOptions,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    stdout: TextIO | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> list[PelicanSample]:
    _check_options(options)
    out_stream = stdout if stdout is not None else sys.stdout
    secret = options.api_key
    owns_client = client is None
    if client is None:
        client = build_client()
    jobs: deque[tuple[str, str, int]] = deque(
        (model, effort, index)
        for effort in options.efforts
        for model in options.models
        for index in range(1, options.iterations + 1)
    )
    gate = Gate(options.concurrency)
    samples: list[PelicanSample] = []
    print_lock = threading.Lock()
    job_lock = threading.Lock()

    def emit(line: str) -> None:
        with print_lock:
            print(redact(line, secret), file=out_stream)

    def take() -> tuple[str, str, int] | None:
        if cancel is not None and cancel.is_set():
            return None
        with job_lock:
            if not jobs or (cancel is not None and cancel.is_set()):
                return None
            return jobs.popleft()

    def worker() -> None:
        while True:
            if cancel is not None and cancel.is_set():
                return
            job = take()
            if job is None:
                return
            model, effort, index = job
            sample = _one_call(
                options,
                client,
                model,
                effort,
                index,
                sleep,
                gate,
                on_event=on_event,
                cancel=cancel,
            )
            samples.append(sample)
            if sample.error == "stopped":
                finish_state = "stopped"
            elif sample.ok:
                finish_state = "done"
            else:
                finish_state = "error"
            if on_event is not None:
                on_event(
                    {
                        "state": finish_state,
                        "index": sample.index,
                        "model": sample.model,
                        "effort": sample.effort,
                        "attempt": sample.attempts,
                        "http": sample.http,
                        "error": sample.error,
                        "ok": sample.ok,
                        "svg": sample.svg,
                        "name": sample.name,
                        "ttft_s": sample.ttft_s,
                        "elapsed_s": sample.elapsed_s,
                        "tok_s": sample.tok_s,
                        "prompt_tokens": sample.prompt_tokens,
                        "input_tokens": sample.input_tokens,
                        "output_tokens": sample.output_tokens,
                        "reasoning_tokens": sample.reasoning_tokens,
                        "cached_tokens": sample.cached_tokens,
                    }
                )
            if sample.ok:
                emit(
                    f"run {sample.name} ok bytes={len(sample.svg or '')} attempt={sample.attempts}"
                )
            else:
                emit(f"run {sample.name} failed http={sample.http} {sample.error}")

    workers = [
        threading.Thread(target=worker, name=f"pelican-{n}")
        for n in range(min(len(jobs), gate.cap) or 1)
    ]
    try:
        max_note = "omitted" if not options.max_tokens else str(options.max_tokens)
        emit(
            f"models={','.join(options.models)} efforts={','.join(options.efforts)} "
            f"iterations={options.iterations} concurrency={options.concurrency} "
            f"max_tokens={max_note}"
        )
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        ordered = sorted(samples, key=lambda sample: (sample.model, sample.effort, sample.index))
        title = redact(
            f"{', '.join(options.models)} reasoning {', '.join(options.efforts)}",
            secret,
        )
        index = write_gallery(options.out, ordered, title=title, secret=secret)
        ok = sum(1 for sample in ordered if sample.ok)
        emit(f"page={index} ok={ok} failed={len(ordered) - ok}")
        return ordered
    finally:
        if owns_client:
            client.close()


def bind_gallery(directory: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    root = str(directory)

    class GalleryHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=root, **kwargs)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), GalleryHandler)


def serve_gallery(directory: Path, host: str = "127.0.0.1", port: int = 8765) -> int:
    try:
        httpd = bind_gallery(directory, host, port)
    except OSError as exc:
        print(f"cannot listen on {host}:{port}: {exc.strerror}", file=sys.stderr)
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


__all__ = [
    "PROMPT",
    "ConfigError",
    "PelicanOptions",
    "PelicanSample",
    "bind_gallery",
    "extract_svg",
    "render_page",
    "request_body",
    "run_pelican",
    "serve_gallery",
    "svg_is_embeddable",
]
