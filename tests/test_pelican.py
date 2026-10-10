"""Offline coverage for the pelican drawing command."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import httpx
import pytest

from llminspector.cli import main
from llminspector.middle_drop import ConfigError
from llminspector.pelican import (
    PROMPT,
    PelicanOptions,
    bind_gallery,
    extract_svg,
    request_body,
    run_pelican,
    svg_is_embeddable,
)

KEY = "pelican-test-key"
BASE = "https://example.invalid/v1"


def _svg(label: str = "bird") -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><desc>{label}</desc></svg>'


def _sse(svg: str) -> bytes:
    payload = {
        "choices": [{"delta": {"content": "```svg\n" + svg + "\n```"}, "finish_reason": "stop"}]
    }
    return ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n").encode()


def _options(tmp_path: Path, **overrides: object) -> PelicanOptions:
    values: dict[str, object] = {
        "base_url": BASE,
        "api_key": KEY,
        "models": ["deepseek-v4.1-flash"],
        "efforts": ["max"],
        "iterations": 1,
        "timeout": 5.0,
        "max_tokens": None,
        "concurrency": 4,
        "max_attempts": 3,
        "out": tmp_path / "pelican",
    }
    values.update(overrides)
    return PelicanOptions(**values)  # type: ignore[arg-type]


def _assert_public_text_clean(text: str) -> None:
    lowered = text.lower()
    for needle in ("mar" + "vin", "s" + "k-", "1.198" + "48845"):
        assert needle not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_request_is_one_completion_without_a_token_cap() -> None:
    body = json.loads(request_body("deepseek-v4.1-flash", "max", None))
    assert body["messages"] == [{"role": "user", "content": PROMPT}]
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["reasoning_effort"] == "max"
    assert body["stream"] is True
    assert "max_tokens" not in body
    assert "tools" not in body
    assert "temperature" not in body
    capped = json.loads(request_body("m", "max", 128))
    assert capped["max_tokens"] == 128


def test_extract_svg_from_a_fence_and_reject_active_markup() -> None:
    svg = _svg()
    assert extract_svg("sure\n```svg\n" + svg + "\n```\n") == svg
    assert extract_svg("no drawing here") is None
    unsafe = svg.replace("<desc>", "<script>")
    assert extract_svg(unsafe) == unsafe
    assert svg_is_embeddable(svg)
    assert not svg_is_embeddable(unsafe)


def test_batch_backs_off_on_429_then_writes_a_scrollable_page(tmp_path: Path) -> None:
    waits: list[float] = []
    seen: list[httpx.Request] = []
    state = {"n": 0}
    order = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        with order:
            state["n"] += 1
            call_n = state["n"]
        if call_n == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Too many concurrent requests.",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        label = "one" if call_n == 2 else "two"
        return httpx.Response(200, content=_sse(_svg(label)))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    options = _options(tmp_path, iterations=2, concurrency=4)
    samples = run_pelican(options, client=client, sleep=waits.append)
    client.close()

    assert [sample.ok for sample in samples] == [True, True]
    assert sorted(sample.attempts for sample in samples) == [1, 2]
    assert waits == [2.0]
    assert all(request.url.path.endswith("/chat/completions") for request in seen)
    assert KEY.encode() not in seen[0].content
    page = (tmp_path / "pelican" / "index.html").read_text(encoding="utf-8")
    assert page.lower().count("<svg") == 2
    assert "Run 1" not in page
    assert "run 1" in page and "run 2" in page
    assert KEY not in page
    assert (tmp_path / "pelican" / "deepseek-v4.1-flash-max-1.svg").is_file()
    assert (tmp_path / "pelican" / "deepseek-v4.1-flash-max-2.svg").is_file()


def test_unsafe_svg_is_saved_but_not_embedded(tmp_path: Path) -> None:
    unsafe = '<svg xmlns="http://www.w3.org/2000/svg"><script>bad</script></svg>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(unsafe))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    samples = run_pelican(_options(tmp_path), client=client, sleep=lambda _seconds: None)
    client.close()
    assert samples[0].ok
    page = (tmp_path / "pelican" / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in page
    assert "not embedded" in page
    assert unsafe in (tmp_path / "pelican" / "deepseek-v4.1-flash-max-1.svg").read_text()


def test_error_body_does_not_leak_the_key(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=f"failed key={KEY}".encode())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    options = _options(tmp_path, max_attempts=1)
    samples = run_pelican(options, client=client, sleep=lambda _seconds: None)
    client.close()
    assert samples[0].ok is False
    page = (tmp_path / "pelican" / "index.html").read_text(encoding="utf-8")
    assert KEY not in page
    assert "[REDACTED]" in page


def test_cli_help_and_refuses_to_dial(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as top:
        main(["pelican", "--help"])
    assert top.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--api-key",
        "--base-url",
        "--models",
        "--efforts",
        "--iterations",
        "--concurrency",
        "--max-tokens",
        "--no-serve",
    ):
        assert flag in text
    assert "LLMINSPECTOR_API_KEY" in text
    assert "LLMINSPECTOR_BASE_URL" in text
    _assert_public_text_clean(text)

    monkeypatch.delenv("LLMINSPECTOR_API_KEY", raising=False)
    monkeypatch.delenv("LLMINSPECTOR_BASE_URL", raising=False)
    called = {"n": 0}

    def boom() -> httpx.Client:
        called["n"] += 1
        raise AssertionError("client factory called")

    monkeypatch.setattr("llminspector.pelican.build_client", boom)
    assert main(["pelican", "--no-serve", "--models", "deepseek-v4.1-flash"]) == 2
    assert called["n"] == 0
    assert KEY not in capsys.readouterr().err

    monkeypatch.setenv("LLMINSPECTOR_API_KEY", KEY)
    monkeypatch.setenv("LLMINSPECTOR_BASE_URL", BASE)
    assert main(["pelican", "--no-serve"]) == 2
    assert called["n"] == 0
    missing = capsys.readouterr().err
    assert "missing model" in missing
    assert KEY not in missing


def test_gallery_serves_every_drawing(tmp_path: Path) -> None:
    out = tmp_path / "pelican"
    out.mkdir()
    (out / "index.html").write_text("<p>one</p><p>two</p>", encoding="utf-8")
    httpd = bind_gallery(out, "127.0.0.1", 0)
    port = httpd.server_address[1]
    try:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        html = urlopen(f"http://127.0.0.1:{port}/").read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert "one" in html and "two" in html


def test_missing_model_is_a_config_error(tmp_path: Path) -> None:
    options = _options(tmp_path, models=[])
    with pytest.raises(ConfigError):
        run_pelican(options, sleep=lambda _seconds: None)
