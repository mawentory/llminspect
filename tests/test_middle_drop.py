"""Offline coverage for the middle-context drop reproducer."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from llminspector.cli import main
from llminspector.middle_drop import (
    FILLER_CHARS,
    FILLER_UNIT,
    MARK_END,
    MARK_MID,
    MARK_START,
    NEIGHBOR_AFTER,
    NEIGHBOR_BEFORE,
    ConfigError,
    MiddleDropOptions,
    _StreamWatch,
    build_prompt,
    call_metrics,
    classify,
    counted_output,
    document_offset,
    parse_sse,
    redact,
    request_body,
    resolve_api_key,
    resolve_base_url,
    run_middle_drop,
    usage_numbers,
    verdict_from,
    wait_seconds,
)

KEY = "test-key-not-secret"
BASE = "https://example.invalid/v1"
CANONICAL_SHA = "a43f439b64e94e83a3a0291351454f5ecf600328cf9e89717f1b58f6afd333ce"
DROP_JSON = (
    '{"start":"PRESENT","mid":"MISSING","end":"PRESENT",'
    '"neighbor_before":"MISSING","neighbor_after":"MISSING"}'
)
FLAGS = (
    "--api-key",
    "--base-url",
    "--models",
    "--efforts",
    "--iterations",
    "--filler-chars",
    "--timeout",
    "--max-tokens",
    "--min-interval",
    "--out",
)


def _assert_public_text_clean(text: str) -> None:
    lowered = text.lower()
    banned = ("mar" + "vin", "s" + "k-", "1.198" + "48845")
    for needle in banned:
        assert needle not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_prompt_construction() -> None:
    assert len(FILLER_UNIT) == 165
    assert FILLER_UNIT.endswith("zulu. ")
    prompt = build_prompt()
    assert FILLER_CHARS == 423588
    assert len(prompt) == 424168
    assert document_offset(prompt) == 212280
    assert prompt.find(MARK_MID) != 212280
    assert prompt.count(MARK_MID) == 2
    assert MARK_START in prompt
    assert MARK_END in prompt
    offset = document_offset(prompt)
    before = prompt.rfind("\n", 0, offset)
    previous = prompt.rfind("\n", 0, before)
    assert prompt[previous + 1 : before] == NEIGHBOR_BEFORE
    after = prompt.find("\n", offset)
    following = prompt.find("\n", after + 1)
    assert prompt[after + 1 : following] == NEIGHBOR_AFTER
    assert build_prompt() == prompt
    assert document_offset("no marker here") == -1


def test_request_body_is_stable_and_keyless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMINSPECTOR_API_KEY", KEY)
    prompt = build_prompt()
    body = request_body("glm-5.3", prompt, "low", None)
    assert body == request_body("glm-5.3", prompt, "low", None)
    assert len(body) == 424354
    assert hashlib.sha256(body).hexdigest() == CANONICAL_SHA
    text = body.decode("utf-8")
    assert "Authorization" not in text
    assert "Bearer" not in text
    assert "api_key" not in text
    assert "LLMINSPECTOR_API_KEY" not in text
    assert KEY not in text
    none_body = request_body("glm-5.3", prompt, "none", None)
    omitted = request_body("glm-5.3", prompt, None, None)
    empty = request_body("glm-5.3", prompt, "", None)
    assert b'"reasoning_effort":"none"' in none_body
    assert none_body != omitted
    assert b"reasoning_effort" not in omitted
    assert b"reasoning_effort" not in empty
    assert b"max_tokens" not in request_body("glm-5.3", prompt, "low", 0)
    assert b"max_tokens" not in request_body("glm-5.3", prompt, "low", None)
    assert b'"max_tokens":8' in request_body("m", "x", "low", 8)


def _sse(obj: object) -> str:
    return "data: " + json.dumps(obj)


def test_parse_sse_keeps_last_usage() -> None:
    events = [
        _sse(
            {
                "id": "early",
                "model": "early-model",
                "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": 1,
                    "input_tokens": 1,
                    "completion_tokens_details": {"reasoning_tokens": 9},
                },
            }
        ),
        _sse({"usage": None, "choices": [{"delta": {"content": ""}}]}),
        _sse(
            {
                "choices": [
                    {"delta": {"reasoning_content": "think-a", "reasoning": "ignore-me"}}
                ]
            }
        ),
        _sse({"choices": [{"delta": {"reasoning": "only-b"}}]}),
        _sse({"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}),
        _sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 115688,
                    "input_tokens": 115688,
                    "completion_tokens": 12,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        ),
        "data: [DONE]",
        _sse(
            {
                "id": "after-done",
                "usage": None,
                "choices": [{"delta": {"content": "!"}}],
            }
        ),
        "data: {not-json",
    ]
    parsed = parse_sse("\n".join(events).encode("utf-8"))
    assert parsed["content"] == "Hello world!"
    assert parsed["reasoning"] == "think-aonly-b"
    assert "ignore-me" not in parsed["reasoning"]
    assert parsed["finish_reason"] == "stop"
    assert parsed["id"] == "after-done"
    assert parsed["model_reported"] == "early-model"
    assert parsed["usage"]["prompt_tokens"] == 115688
    numbers = usage_numbers(parsed["usage"])
    assert numbers["prompt_tokens"] == 115688
    assert numbers["input_tokens"] == 115688
    assert numbers["reasoning_tokens"] == 0
    missing = usage_numbers({"prompt_tokens": 3})
    assert missing["reasoning_tokens"] is None
    assert missing["input_tokens"] is None
    assert missing["cached_tokens"] is None
    assert usage_numbers({"completion_tokens_details": {}})["reasoning_tokens"] is None


def test_tok_s_includes_reasoning_without_counting_it_twice() -> None:
    included = counted_output(
        {
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 22},
        }
    )
    assert included == 30
    separate = counted_output(
        {"output_tokens": 4, "completion_tokens_details": {"reasoning_tokens": 12}}
    )
    assert separate == 16
    watch = _StreamWatch()
    started = time.monotonic()
    watch.feed(
        "data: "
        + json.dumps({"choices": [{"delta": {"reasoning_content": "reason " * 40}}]})
        + "\n",
        started,
    )
    watch.feed(
        "data: " + json.dumps({"choices": [{"delta": {"content": "<svg></svg>"}}]}) + "\n",
        started,
    )
    live = call_metrics({}, watch.ttft_s, 2.0, watch.estimated, watch.reasoning_estimated)
    assert live["reasoning_tokens"] > 0
    assert live["output_tokens"] >= live["reasoning_tokens"]
    assert live["tok_s"] is not None and live["tok_s"] > 0
    final = call_metrics(
        {
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 22},
        },
        0.4,
        2.0,
        watch.estimated,
        watch.reasoning_estimated,
    )
    assert final["output_tokens"] == 30
    assert final["reasoning_tokens"] == 22
    assert final["tok_s"] == pytest.approx(30 / 1.6)


def test_classify_drop_and_neighbors() -> None:
    observed = verdict_from(DROP_JSON)
    assert observed["neighbor_before"] == "MISSING"
    assert classify(observed) == "DROP"
    still_drop = verdict_from(
        '{"start":"PRESENT","mid":"MISSING","end":"PRESENT",'
        '"neighbor_before":"PRESENT","neighbor_after":"PRESENT"}'
    )
    assert classify(still_drop) == "DROP"
    start_missing = verdict_from(
        '{"start":"MISSING","mid":"MISSING","end":"PRESENT",'
        '"neighbor_before":"MISSING","neighbor_after":"MISSING"}'
    )
    assert classify(start_missing) == "MID_MISSING"
    end_missing = verdict_from('{"start":"PRESENT","mid":"MISSING","end":"MISSING"}')
    assert classify(end_missing) == "MID_MISSING"
    assert classify(verdict_from('{"mid":"PRESENT"}')) == "PRESENT"
    assert classify(verdict_from("")) == "NO_VERDICT"
    prose = verdict_from("I do not see <<<MIDDLE>>>")
    assert prose["mid"] == "MISSING"
    assert prose["neighbor_before"] is None
    assert classify(prose) == "MID_MISSING"
    edged = dict(prose)
    edged["start"] = "PRESENT"
    edged["end"] = "PRESENT"
    assert classify(edged) == "DROP"
    fenced = "```json\n" + DROP_JSON + "\n```"
    assert classify(verdict_from(fenced)) == "DROP"
    assert verdict_from('{"middle":"missing"}')["mid"] == "MISSING"


def test_credentials_and_redact() -> None:
    with pytest.raises(ConfigError) as missing_key:
        resolve_api_key(None, {"OPENAI_API_KEY": KEY, "API_KEY": KEY})
    assert KEY not in str(missing_key.value)
    with pytest.raises(ConfigError) as missing_url:
        resolve_base_url(None, {"API_BASE": BASE, "LLMINSPECTOR_API_KEY": KEY})
    assert KEY not in str(missing_url.value)
    assert resolve_api_key("from-cli", {"LLMINSPECTOR_API_KEY": "from-env"}) == "from-cli"
    assert resolve_api_key("   ", {"LLMINSPECTOR_API_KEY": "from-env"}) == "from-env"
    with pytest.raises(ConfigError):
        resolve_api_key(None, {"OPENAI_API_KEY": KEY})
    assert resolve_base_url(BASE + "/", {}) == BASE
    assert (
        resolve_base_url("  ", {"LLMINSPECTOR_BASE_URL": "https://example.invalid/env/"})
        == "https://example.invalid/env"
    )
    with pytest.raises(ConfigError) as bad_scheme:
        resolve_base_url("example.invalid/v1", {"LLMINSPECTOR_API_KEY": KEY})
    assert KEY not in str(bad_scheme.value)
    with pytest.raises(ConfigError):
        resolve_base_url("ftp://example.invalid/v1", {})
    cleaned = redact(f"status=MISSING key={KEY}", KEY)
    assert KEY not in cleaned
    assert "MISSING" in cleaned
    assert redact("MISSING", "") == "MISSING"


def test_wait_seconds() -> None:
    assert wait_seconds(None, 1000.0) == 0
    assert wait_seconds(999.0, 1000.0) == pytest.approx(3.2)
    assert wait_seconds(0.0, 150.0) == 0


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as top:
        main(["--help"])
    assert top.value.code == 0
    top_text = capsys.readouterr().out
    assert "middle-drop" in top_text
    assert "pelican" in top_text
    assert "ui" in top_text
    _assert_public_text_clean(top_text)

    with pytest.raises(SystemExit) as sub:
        main(["middle-drop", "--help"])
    assert sub.value.code == 0
    sub_text = capsys.readouterr().out
    for flag in FLAGS:
        assert flag in sub_text
    assert "LLMINSPECTOR_API_KEY" in sub_text
    assert "LLMINSPECTOR_BASE_URL" in sub_text
    _assert_public_text_clean(sub_text)

    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    assert capsys.readouterr().out.strip() == "llminspect 0.1.0"

    assert main([]) == 2
    captured = capsys.readouterr()
    assert "scanners are not shipped yet" in captured.err


def test_cli_refuses_to_dial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("LLMINSPECTOR_API_KEY", raising=False)
    monkeypatch.delenv("LLMINSPECTOR_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("API_KEY", KEY)
    called = {"n": 0}

    def boom() -> httpx.Client:
        called["n"] += 1
        raise AssertionError("client factory called")

    monkeypatch.setattr("llminspector.middle_drop.build_client", boom)
    assert main(["middle-drop"]) == 2
    assert called["n"] == 0
    assert KEY not in capsys.readouterr().err

    monkeypatch.setenv("LLMINSPECTOR_API_KEY", KEY)
    monkeypatch.setenv("LLMINSPECTOR_BASE_URL", BASE)
    missing = tmp_path / "no-such-dir" / "out.jsonl"
    assert main(["middle-drop", "--out", str(missing)]) == 2
    assert called["n"] == 0
    assert KEY not in capsys.readouterr().err


def _options(
    tmp_path: Path,
    *,
    timeout: float = 5.0,
    out_name: str = "middle-drop.jsonl",
) -> MiddleDropOptions:
    return MiddleDropOptions(
        base_url=BASE,
        api_key=KEY,
        models=["glm-5.3"],
        efforts=["low"],
        iterations=1,
        filler_chars=32,
        timeout=timeout,
        max_tokens=None,
        min_interval=0.0,
        out=tmp_path / out_name,
    )


def _drop_sse() -> bytes:
    chunks = [
        {
            "id": "chatcmpl-test",
            "model": "glm-5.3",
            "choices": [
                {
                    "delta": {"content": DROP_JSON},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-test",
            "choices": [],
            "usage": {
                "prompt_tokens": 115688,
                "input_tokens": 115688,
                "completion_tokens": 24,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]
    lines = ["data: " + json.dumps(obj) for obj in chunks]
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_mock_transport_drop_and_timeout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_drop_sse())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    options = _options(tmp_path)
    records = run_middle_drop(options, client=client)
    client.close()

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == BASE + "/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["content-type"] == "application/json"
    expected = request_body("glm-5.3", build_prompt(32), "low", None)
    assert request.content == expected
    assert KEY.encode() not in request.content
    assert b"Authorization" not in request.content
    assert b"Bearer" not in request.content

    assert len(records) == 1
    record = records[0]
    assert record["verdict_class"] == "DROP"
    assert record["http"] == 200
    assert record["finish_reason"] == "stop"
    assert record["prompt_tokens"] == 115688
    assert record["input_tokens"] == 115688
    assert record["reasoning_tokens"] == 0
    assert record["timed_out"] is False
    captured = capsys.readouterr()
    assert KEY not in captured.out
    assert KEY not in captured.err
    assert "DROP_DETAIL" in captured.out
    assert "Authorization" not in captured.out
    written = Path(options.out).read_text(encoding="utf-8")
    assert KEY not in written
    assert "example.invalid" not in written
    assert "Authorization" not in written
    saved = json.loads(written)
    assert saved["verdict_class"] == "DROP"
    assert saved["request_sha256"] == record["request_sha256"]

    release = threading.Event()

    def blocking(request: httpx.Request) -> httpx.Response:
        release.wait(2.0)
        return httpx.Response(200, content=b"data: [DONE]\n")

    blocked = httpx.Client(transport=httpx.MockTransport(blocking))
    try:
        timed = run_middle_drop(
            _options(tmp_path, timeout=0.05, out_name="timeout.jsonl"),
            client=blocked,
        )
    finally:
        release.set()
        time.sleep(0.05)
        blocked.close()
    assert timed[0]["timed_out"] is True
    assert timed[0]["verdict_class"] != "DROP"
    assert timed[0]["http"] == 408
    again = capsys.readouterr()
    assert KEY not in again.out
    assert KEY not in again.err
