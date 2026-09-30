"""Repeat one chat-completions body and detect a silent middle-context drop.

Posts the same request bytes on every call and checks whether the middle of
the prompt survived. A DROP means start and end are PRESENT and mid is
MISSING. This is a reproducer, not an identity signal.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import httpx

FILLER_CHARS = 423_588
MIN_INTERVAL_S = 4.2
DEFAULT_TIMEOUT_S = 150.0
MARK_START = "<<<START>>>"
MARK_MID = "<<<MIDDLE>>>"
MARK_END = "<<<END>>>"
NEIGHBOR_BEFORE = "NEIGHBOR_BEFORE_7f3a"
NEIGHBOR_AFTER = "NEIGHBOR_AFTER_7f3a"
FILLER_UNIT = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu. "
)

_CONNECT_CAP_S = 30.0
_VERDICT_FIELDS = ("start", "mid", "end", "neighbor_before", "neighbor_after")
_PHRASES = (
    "no <<<middle>>>",
    "do not see <<<middle>>>",
    "i do not see <<<middle>>>",
)


class ConfigError(Exception):
    """Missing key, missing base URL, or an output path that cannot be opened."""


@dataclass
class MiddleDropOptions:
    base_url: str
    api_key: str
    models: list[str]
    efforts: list[str]
    iterations: int
    filler_chars: int
    timeout: float
    max_tokens: int | None
    min_interval: float
    out: Path


def build_prompt(filler_chars: int = FILLER_CHARS) -> str:
    half = filler_chars // 2
    left = (FILLER_UNIT * ((half // len(FILLER_UNIT)) + 2))[:half]
    right = (FILLER_UNIT * (((filler_chars - half) // len(FILLER_UNIT)) + 2))[
        : filler_chars - half
    ]
    task = (
        "Audit the document below for three exact markers and the two neighbor "
        "tokens that sit on the lines immediately before and after the middle "
        "marker. Reply with JSON only, no markdown: "
        '{"start":"PRESENT","mid":"PRESENT","end":"PRESENT",'
        '"neighbor_before":"PRESENT","neighbor_after":"PRESENT"} '
        "using PRESENT or MISSING for each. Do not restate the document. "
        f"Markers: {MARK_START} {MARK_MID} {MARK_END}. "
        f"Neighbors: {NEIGHBOR_BEFORE} {NEIGHBOR_AFTER}.\n\n"
    )
    return (
        task
        + MARK_START
        + "\n"
        + left
        + "\n"
        + NEIGHBOR_BEFORE
        + "\n"
        + MARK_MID
        + "\n"
        + NEIGHBOR_AFTER
        + "\n"
        + right
        + "\n"
        + MARK_END
        + "\n"
        + "End of document. Reply with the JSON verdict now.\n"
    )


def document_offset(prompt: str, marker: str = MARK_MID) -> int:
    return prompt.rfind(marker)


def request_body(
    model: str,
    prompt: str,
    effort: str | None,
    max_tokens: int | None,
) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
    }
    if effort:
        payload["reasoning_effort"] = effort
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_sse(raw: bytes | str) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    msg_id: str | None = None
    model_reported: str | None = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        msg_id = obj.get("id") or msg_id
        model_reported = obj.get("model") or model_reported
        if isinstance(obj.get("usage"), dict):
            usage = obj["usage"]
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        reason = choice.get("finish_reason")
        if reason:
            finish_reason = reason
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        piece = delta.get("content")
        if piece:
            content_parts.append(piece if isinstance(piece, str) else str(piece))
        if "reasoning_content" in delta:
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning if isinstance(reasoning, str) else str(reasoning))
        elif delta.get("reasoning"):
            reasoning = delta.get("reasoning")
            reasoning_parts.append(reasoning if isinstance(reasoning, str) else str(reasoning))
    return {
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
        "usage": usage,
        "finish_reason": finish_reason,
        "id": msg_id,
        "model_reported": model_reported,
    }


def _blank_verdict() -> dict[str, str | None]:
    return {field: None for field in _VERDICT_FIELDS}


def _window_label(upper: str, token: str) -> str | None:
    at = upper.find(token)
    if at < 0:
        return None
    window = upper[max(0, at - 30) : at + 48]
    if "MISSING" in window:
        return "MISSING"
    if "PRESENT" in window:
        return "PRESENT"
    return None


def verdict_from(text: str) -> dict[str, str | None]:
    result = _blank_verdict()
    if text:
        start_i = text.find("{")
        end_i = text.rfind("}")
        if start_i != -1 and end_i > start_i:
            try:
                obj = json.loads(text[start_i : end_i + 1])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                aliases = (
                    ("start", "start"),
                    ("mid", "mid"),
                    ("middle", "mid"),
                    ("end", "end"),
                    ("neighbor_before", "neighbor_before"),
                    ("neighbor_after", "neighbor_after"),
                )
                for src, dst in aliases:
                    if result[dst] is not None or src not in obj:
                        continue
                    val = obj[src]
                    if isinstance(val, str):
                        result[dst] = val.strip().upper()
        upper = text.upper()
        for field, token in (("start", "START"), ("mid", "MID"), ("end", "END")):
            if result[field] is None:
                result[field] = _window_label(upper, token)
        if result["mid"] is None:
            lower = text.lower()
            if any(phrase in lower for phrase in _PHRASES):
                result["mid"] = "MISSING"
    return result


def classify(verdict: dict[str, Any]) -> str:
    mid = verdict.get("mid")
    if (
        verdict.get("start") == "PRESENT"
        and verdict.get("end") == "PRESENT"
        and mid == "MISSING"
    ):
        return "DROP"
    if mid == "MISSING":
        return "MID_MISSING"
    if mid == "PRESENT":
        return "PRESENT"
    return "NO_VERDICT"


def usage_numbers(usage: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}
    completion = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    reasoning = completion.get("reasoning_tokens") if isinstance(completion, dict) else None
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "input_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
    }


def _configured(cli_value: str | None, env_value: str | None) -> str:
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    return ""


def resolve_base_url(
    cli_value: str | None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    raw = _configured(cli_value, env.get("LLMINSPECTOR_BASE_URL"))
    if not raw:
        raise ConfigError("missing base URL: pass --base-url or set LLMINSPECTOR_BASE_URL")
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raise ConfigError("base URL must start with http:// or https://")
    return raw.rstrip("/")


def resolve_api_key(
    cli_value: str | None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    raw = _configured(cli_value, env.get("LLMINSPECTOR_API_KEY"))
    if not raw:
        raise ConfigError("missing API key: pass --api-key or set LLMINSPECTOR_API_KEY")
    return raw


def redact(text: str, secret: str) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def wait_seconds(
    last_start: float | None,
    now: float,
    min_interval: float = MIN_INTERVAL_S,
) -> float:
    if last_start is None:
        return 0.0
    return max(0.0, min_interval - (now - last_start))


def build_client() -> httpx.Client:
    """HTTP/1.1 client with no default host and no redirect following."""
    return httpx.Client(follow_redirects=False)


def _error_text(exc: BaseException, secret: str) -> str:
    return f"{type(exc).__name__}: {redact(str(exc), secret)}"


def stream_chat(
    base_url: str,
    api_key: str,
    body: bytes,
    timeout: float,
    client: httpx.Client,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    wall = float(timeout)
    connect = min(_CONNECT_CAP_S, wall)
    req_timeout = httpx.Timeout(wall, connect=connect)
    started = time.monotonic()
    deadline = started + wall
    lock = threading.Lock()
    box: dict[str, Any] = {
        "http": None,
        "raw": bytearray(),
        "timed_out": False,
        "error": None,
    }

    def worker() -> None:
        try:
            with client.stream(
                "POST",
                url,
                content=body,
                headers=headers,
                follow_redirects=False,
                timeout=req_timeout,
            ) as response:
                with lock:
                    box["http"] = response.status_code
                for chunk in response.iter_bytes():
                    if time.monotonic() >= deadline:
                        with lock:
                            box["timed_out"] = True
                        return
                    if chunk:
                        with lock:
                            box["raw"].extend(chunk)
                    if time.monotonic() >= deadline:
                        with lock:
                            box["timed_out"] = True
                        return
        except httpx.TimeoutException:
            with lock:
                box["timed_out"] = True
        except Exception as exc:
            with lock:
                box["error"] = exc

    thread = threading.Thread(target=worker, name="middle-drop-stream", daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, deadline - time.monotonic()))
    elapsed = time.monotonic() - started
    with lock:
        status = box["http"]
        raw = bytes(box["raw"])
        timed_out = bool(box["timed_out"])
        error = box["error"]
    if thread.is_alive():
        timed_out = True
    if error is not None and not timed_out and not isinstance(error, httpx.TimeoutException):
        raise error
    if status is None:
        return {"http": 408, "raw": raw, "elapsed_s": elapsed, "timed_out": True}
    return {"http": status, "raw": raw, "elapsed_s": elapsed, "timed_out": timed_out}


def _scrub(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return redact(value, secret)
    if isinstance(value, dict):
        return {key: _scrub(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, secret) for item in value]
    return value


def _emit(stdout: TextIO, line: str, secret: str) -> None:
    print(redact(line, secret), file=stdout)


def _distinct(values: list[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _call_line(record: dict[str, Any], iterations: int) -> str:
    verdict = json.dumps(record["verdict"], ensure_ascii=False, separators=(",", ":"))
    elapsed = record["elapsed_s"]
    elapsed_s = f"{elapsed:.3f}" if isinstance(elapsed, float) else str(elapsed)
    return (
        f"{record['model']} effort={record['reasoning_effort']} "
        f"{record['i']}/{iterations} {record['verdict_class']} http={record['http']} "
        f"prompt_tokens={record['prompt_tokens']} input_tokens={record['input_tokens']} "
        f"reasoning_tokens={record['reasoning_tokens']} "
        f"completion_tokens={record['completion_tokens']} "
        f"mid_in_req={record['mid_in_request']} "
        f"sha256={record['request_sha256'][:12]} verdict={verdict} "
        f"reasoning_chars={record['reasoning_chars']} elapsed_s={elapsed_s}"
    )


def _summary_lines(records: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["model"]), str(record["reasoning_effort"]))
        groups.setdefault(key, []).append(record)
    lines = ["summary"]
    for (model, effort), rows in groups.items():
        drop = sum(1 for row in rows if row["verdict_class"] == "DROP")
        present = sum(1 for row in rows if row["verdict_class"] == "PRESENT")
        other = len(rows) - drop - present
        lines.append(
            f"{model}:{effort} DROP={drop} PRESENT={present} other={other} "
            f"prompt_tokens={_distinct([row['prompt_tokens'] for row in rows])} "
            f"reasoning_tokens={_distinct([row['reasoning_tokens'] for row in rows])} "
            f"drops={[row['i'] for row in rows if row['verdict_class'] == 'DROP']} "
            f"sha256={_distinct([row['request_sha256'] for row in rows])}"
        )
    return lines


def run_middle_drop(
    options: MiddleDropOptions,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    stdout: TextIO | None = None,
) -> list[dict[str, Any]]:
    out_stream = stdout if stdout is not None else sys.stdout
    secret = options.api_key
    out_path = Path(options.out)
    out_fh: TextIO | None = None
    owns_client = client is None
    records: list[dict[str, Any]] = []
    try:
        try:
            out_fh = out_path.open("w", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise ConfigError(f"cannot open output file: {exc.strerror}") from exc
        if client is None:
            client = build_client()
            owns_client = True
        prompt = build_prompt(options.filler_chars)
        offset = document_offset(prompt)
        max_note = "omitted" if not options.max_tokens else str(options.max_tokens)
        _emit(
            out_stream,
            (
                f"prompt_chars={len(prompt)} document_offset={offset} "
                f"models={','.join(options.models)} efforts={','.join(options.efforts)} "
                f"iterations={options.iterations} max_tokens={max_note} "
                f"base_url={options.base_url}"
            ),
            secret,
        )
        mid_in_request = offset >= 0
        neighbor_before_in_request = NEIGHBOR_BEFORE in prompt
        neighbor_after_in_request = NEIGHBOR_AFTER in prompt
        last_start: float | None = None
        for effort in options.efforts:
            for model in options.models:
                body = request_body(model, prompt, effort, options.max_tokens)
                digest = hashlib.sha256(body).hexdigest()
                for index in range(1, options.iterations + 1):
                    delay = wait_seconds(last_start, monotonic(), options.min_interval)
                    if delay > 0:
                        sleep(delay)
                    last_start = monotonic()
                    transport_error: str | None = None
                    try:
                        result = stream_chat(
                            options.base_url,
                            secret,
                            body,
                            options.timeout,
                            client,
                        )
                    except httpx.HTTPError as exc:
                        transport_error = _error_text(exc, secret)
                        result = {
                            "http": None,
                            "raw": b"",
                            "elapsed_s": monotonic() - last_start,
                            "timed_out": False,
                        }
                    http_status = result["http"]
                    raw = result["raw"]
                    if http_status != 200:
                        parsed = {
                            "content": "",
                            "reasoning": "",
                            "usage": {},
                            "finish_reason": None,
                            "id": None,
                            "model_reported": None,
                        }
                        verdict = _blank_verdict()
                        verdict_class = "HTTP_ERROR"
                        if transport_error is not None:
                            error_body = transport_error[:500]
                        else:
                            error_body = redact(
                                raw.decode("utf-8", errors="replace") if raw else "",
                                secret,
                            )[:500]
                    else:
                        parsed = parse_sse(raw)
                        verdict = verdict_from(parsed["content"])
                        if verdict["mid"] is None:
                            merged = verdict_from(parsed["reasoning"] + "\n" + parsed["content"])
                            if merged["mid"] is not None:
                                verdict = merged
                        verdict_class = classify(verdict)
                        error_body = None
                    numbers = usage_numbers(parsed["usage"])
                    content = parsed["content"]
                    reasoning = parsed["reasoning"]
                    record: dict[str, Any] = {
                        "i": index,
                        "model": model,
                        "reasoning_effort": effort,
                        "http": http_status,
                        "elapsed_s": result["elapsed_s"],
                        "timed_out": bool(result["timed_out"]),
                        "request_bytes": len(body),
                        "request_sha256": digest,
                        "mid_in_request": mid_in_request,
                        "neighbor_before_in_request": neighbor_before_in_request,
                        "neighbor_after_in_request": neighbor_after_in_request,
                        "mid_offset_document": offset,
                        "id": parsed["id"],
                        "model_reported": parsed["model_reported"],
                        "finish_reason": parsed["finish_reason"],
                        **numbers,
                        "usage": parsed["usage"],
                        "verdict": verdict,
                        "verdict_class": verdict_class,
                        "reasoning_chars": len(reasoning),
                        "content_chars": len(content),
                        "content_text": content[:1000],
                        "reasoning_head": reasoning[:600],
                    }
                    if http_status != 200:
                        record["error_body"] = error_body
                    record = _scrub(record, secret)
                    records.append(record)
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    _emit(out_stream, _call_line(record, options.iterations), secret)
                    if verdict_class == "DROP":
                        snippet = str(record["content_text"]).replace("\n", " ")[:500]
                        _emit(
                            out_stream,
                            f"DROP_DETAIL id={record['id']} content={snippet}",
                            secret,
                        )
        for line in _summary_lines(records):
            _emit(out_stream, line, secret)
    finally:
        if out_fh is not None:
            out_fh.close()
        if owns_client and client is not None:
            client.close()
    return records
