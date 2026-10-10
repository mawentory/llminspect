"""Local form: pelican drawings and context-marker probes on one page."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from llminspector.cli import main
from llminspector.middle_drop import MARK_MID, ConfigError, build_prompt
from llminspector.pelican import PROMPT
from llminspector.ui import UIApp, _marker_report, _public_call, bind_ui, parse_run_config

KEY = "unit-form-key-9f3a"
SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><rect width="8" height="8"/></svg>'
VERDICT = (
    '{"start":"PRESENT","mid":"PRESENT","end":"PRESENT",'
    '"neighbor_before":"PRESENT","neighbor_after":"PRESENT"}'
)
FIELD_IDS = (
    "base-url",
    "api-key",
    "model",
    "test-pelican",
    "test-marker",
    "pelican-runs",
    "marker-runs",
    "concurrency",
    "effort",
    "context-chars",
)


def _assert_public_text_clean(text: str) -> None:
    lowered = text.lower()
    for needle in ("mar" + "vin", "s" + "k-", "1.198" + "48845"):
        assert needle not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def _sse(text: str) -> bytes:
    payload = {
        "id": "chatcmpl-test",
        "model": "demo",
        "choices": [{"delta": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 64,
            "input_tokens": 64,
            "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return ("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode()


class ChatMock:
    def __init__(
        self,
        hold: threading.Event | None = None,
        hold_after: int | None = None,
    ) -> None:
        self.hold = hold
        self.hold_after = hold_after
        self.seen = 0
        self.bodies: list[dict[str, object]] = []
        self.auths: list[str] = []
        self.arrived = threading.Event()
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw)
                auth = self.headers.get("Authorization", "")
                with owner.lock:
                    owner.bodies.append(body)
                    owner.auths.append(auth)
                    owner.seen += 1
                    seen = owner.seen
                owner.arrived.set()
                if owner.hold is not None and (
                    owner.hold_after is None or seen > owner.hold_after
                ):
                    owner.hold.wait(30)
                content = body["messages"][0]["content"]
                if PROMPT in content:
                    reply = SVG
                elif MARK_MID in content:
                    reply = VERDICT
                else:
                    reply = "{}"
                data = _sse(reply)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        if self.hold is not None:
            self.hold.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class FormServer:
    def __init__(self) -> None:
        self.httpd = bind_ui("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def origin(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _wait_job(client: httpx.Client, origin: str, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 20
    view: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = client.get(f"{origin}/api/runs/{job_id}")
        view = response.json()
        if view.get("status") in {"done", "error", "stopped"}:
            return view
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {view}")


def test_parse_defaults_and_rejects_a_blank_choice() -> None:
    config = parse_run_config(
        {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key": "k",
            "model": "demo",
            "pelican": True,
        }
    )
    assert config.effort == "max"
    assert config.pelican_runs == 5
    assert config.marker_runs == 5
    assert config.concurrency == 4
    assert config.context_chars == 423588
    assert config.marker is False
    with pytest.raises(ConfigError, match="choose pelican"):
        parse_run_config(
            {
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "k",
                "model": "demo",
            }
        )


def test_marker_report_names_pass_and_fail() -> None:
    passed = _marker_report(
        [{"verdict_class": "PRESENT"}, {"verdict_class": "PRESENT"}],
        64,
        100,
        finished=True,
    )
    assert passed["outcome"] == "pass"
    assert passed["headline"].startswith("Passed")
    failed = _marker_report(
        [{"verdict_class": "DROP"}, {"verdict_class": "HTTP_ERROR"}],
        900_000,
        100,
        finished=True,
    )
    assert failed["outcome"] == "fail"
    assert "dropped the middle" in failed["headline"]
    assert "failed the call" in failed["headline"]
    stopped = _marker_report([], 64, 10, finished=False, stopped=True)
    assert stopped["outcome"] == "stopped"
    assert stopped["headline"].startswith("Stopped")


def test_notes_redact_the_key() -> None:
    app = UIApp()
    job: dict[str, object] = {"api_key": KEY, "log": []}
    app._note(job, f"upstream said {KEY}")
    assert job["log"] == ["upstream said [REDACTED]"]


def test_page_lists_the_form_fields() -> None:
    form = FormServer()
    try:
        page = httpx.get(form.origin + "/", timeout=5).text
    finally:
        form.close()
    for field_id in FIELD_IDS:
        assert f'id="{field_id}"' in page
    assert "Pelican" in page
    assert "Marker" in page
    assert "/api/runs" in page
    assert "900000" in page
    assert 'id="stop-button"' in page
    assert "markStopped" in page
    assert "localStorage" in page
    assert "Passed" in page
    assert "Failed" in page
    assert KEY not in page


def test_rejects_incomplete_runs() -> None:
    form = FormServer()
    try:
        client = httpx.Client(timeout=5)
        missing_key = client.post(
            form.origin + "/api/runs",
            json={"base_url": "http://127.0.0.1:9/v1", "model": "demo", "pelican": True},
        )
        missing_model = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": KEY,
                "pelican": True,
            },
        )
        neither = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": KEY,
                "model": "demo",
            },
        )
        bad_json = client.post(
            form.origin + "/api/runs",
            content=b"{",
            headers={"Content-Type": "application/json"},
        )
    finally:
        form.close()
    assert missing_key.status_code == 400
    assert "API key" in missing_key.json()["error"]
    assert missing_model.status_code == 400
    assert "model" in missing_model.json()["error"]
    assert neither.status_code == 400
    assert "pelican" in neither.json()["error"]
    assert bad_json.status_code == 400
    assert KEY not in missing_model.text


def test_second_start_is_a_conflict_while_the_first_is_in_flight() -> None:
    hold = threading.Event()
    mock = ChatMock(hold)
    form = FormServer()
    try:
        client = httpx.Client(timeout=5)
        first = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "effort": "max",
                "pelican": True,
                "marker": False,
                "pelican_runs": 1,
                "concurrency": 1,
            },
        )
        assert first.status_code == 202
        second = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "pelican": True,
                "pelican_runs": 1,
            },
        )
        assert second.status_code == 409
        assert KEY not in second.text
        assert mock.arrived.wait(5)
        hold.set()
        view = _wait_job(client, form.origin, first.json()["id"])
        assert view["status"] == "done"
        assert KEY not in json.dumps(view)
    finally:
        hold.set()
        form.close()
        mock.close()


def test_both_tests_finish_without_echoing_the_key() -> None:
    mock = ChatMock()
    form = FormServer()
    try:
        client = httpx.Client(timeout=5)
        started = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo-model",
                "effort": "max",
                "pelican": True,
                "marker": True,
                "pelican_runs": 1,
                "marker_runs": 1,
                "concurrency": 2,
                "context_chars": 64,
            },
        )
        assert started.status_code == 202
        view = _wait_job(client, form.origin, started.json()["id"])
        listed = client.get(form.origin + "/api/runs")
    finally:
        form.close()
        mock.close()

    assert view["status"] == "done"
    assert view["error"] is None
    pelican = view["pelican"]
    assert isinstance(pelican, list) and len(pelican) == 1
    assert pelican[0]["embed"] is True
    assert pelican[0]["svg"] == SVG
    marker = view["marker"]
    assert isinstance(marker, dict)
    records = marker["records"]
    assert isinstance(records, list) and len(records) == 1
    assert records[0]["verdict_class"] == "PRESENT"
    assert records[0]["http"] == 200
    assert records[0]["prompt_tokens"] == 64
    assert "context_chars=64" in str(marker["summary"])
    assert marker["outcome"] == "pass"
    assert str(marker["headline"]).startswith("Passed")
    calls = view["calls"]
    assert isinstance(calls, list) and calls
    assert any(call.get("ttft_s") is not None for call in calls)
    blob = json.dumps(view)
    assert KEY not in blob
    assert listed.status_code == 200
    assert KEY not in listed.text
    assert started.json()["id"] in {row["id"] for row in listed.json()["runs"]}

    assert mock.auths == [f"Bearer {KEY}", f"Bearer {KEY}"]
    pelican_body, marker_body = mock.bodies
    assert pelican_body["messages"] == [{"role": "user", "content": PROMPT}]
    assert pelican_body["reasoning_effort"] == "max"
    assert pelican_body["stream"] is True
    assert "max_tokens" not in pelican_body
    assert "tools" not in pelican_body
    assert "temperature" not in pelican_body
    marker_content = marker_body["messages"][0]["content"]
    assert isinstance(marker_content, str)
    assert marker_content == build_prompt(64)
    assert MARK_MID in marker_content
    assert marker_body["temperature"] == 0
    assert "max_tokens" not in marker_body


def test_ui_help_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["ui", "--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    assert "--host" in text
    assert "--port" in text
    assert "8787" in text
    assert "marker" in text
    _assert_public_text_clean(text)

    seen: dict[str, object] = {}

    def fake(host: str, port: int) -> int:
        seen["host"] = host
        seen["port"] = port
        return 0

    monkeypatch.setattr("llminspector.ui.serve_ui", fake)
    assert main(["ui", "--host", "127.0.0.1", "--port", "8791"]) == 0
    assert seen == {"host": "127.0.0.1", "port": 8791}


def test_public_call_refreshes_tok_s_while_streaming() -> None:
    row = {
        "state": "streaming",
        "output_tokens": 80,
        "reasoning_tokens": 60,
        "ttft_s": 0.5,
        "elapsed_s": 1.0,
        "_elapsed_at": time.monotonic() - 3.0,
    }
    view = _public_call(row, time.monotonic())
    assert view["reasoning_tokens"] == 60
    elapsed = float(view["elapsed_s"])
    assert elapsed > 3
    assert view["tok_s"] == pytest.approx(80 / (elapsed - 0.5))


def test_stop_frees_the_form_while_close_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    hold = threading.Event()
    release_close = threading.Event()
    mock = ChatMock(hold)
    form = FormServer()
    original = httpx.Client.close

    def slow_close(self: httpx.Client) -> None:
        release_close.wait(5)
        original(self)

    monkeypatch.setattr(httpx.Client, "close", slow_close)
    try:
        client = httpx.Client(timeout=5)
        started = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "pelican": True,
                "pelican_runs": 1,
                "concurrency": 1,
            },
        )
        assert started.status_code == 202
        job_id = started.json()["id"]
        assert mock.arrived.wait(5)
        stopped = client.post(f"{form.origin}/api/runs/{job_id}/stop")
        assert stopped.status_code == 202
        assert stopped.json()["status"] == "stopped"
        view = client.get(f"{form.origin}/api/runs/{job_id}").json()
        assert view["status"] == "stopped"
        fresh = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "pelican": True,
                "pelican_runs": 1,
                "concurrency": 1,
            },
        )
        assert fresh.status_code == 202
        assert fresh.json()["id"] != job_id
        assert KEY not in stopped.text
        assert KEY not in fresh.text
    finally:
        release_close.set()
        hold.set()
        mock.close()
        form.close()


def test_stop_keeps_a_finished_drawing_and_aborts_the_rest() -> None:
    hold = threading.Event()
    mock = ChatMock(hold, hold_after=1)
    form = FormServer()
    try:
        client = httpx.Client(timeout=5)
        started = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "pelican": True,
                "marker": False,
                "pelican_runs": 2,
                "concurrency": 1,
                "effort": "max",
            },
        )
        assert started.status_code == 202
        job_id = started.json()["id"]
        deadline = time.monotonic() + 10
        ready = False
        while time.monotonic() < deadline:
            view = client.get(f"{form.origin}/api/runs/{job_id}").json()
            drawings = view.get("pelican") or []
            if any(row.get("embed") for row in drawings):
                ready = True
                break
            time.sleep(0.05)
        assert ready
        stopped = client.post(f"{form.origin}/api/runs/{job_id}/stop")
        assert stopped.status_code == 202
        view = _wait_job(client, form.origin, job_id)
        assert view["status"] == "stopped"
        drawings = view["pelican"]
        assert isinstance(drawings, list)
        assert sum(1 for row in drawings if row.get("embed")) == 1
        calls = view["calls"]
        assert isinstance(calls, list)
        assert any(row.get("state") == "stopped" for row in calls)
        assert KEY not in json.dumps(view)
        missing = client.post(form.origin + "/api/runs/missing-run/stop")
        assert missing.status_code == 404
        again = client.post(f"{form.origin}/api/runs/{job_id}/stop")
        assert again.status_code == 409
        fresh = client.post(
            form.origin + "/api/runs",
            json={
                "base_url": mock.base_url,
                "api_key": KEY,
                "model": "demo",
                "pelican": True,
                "pelican_runs": 1,
                "concurrency": 1,
            },
        )
        assert fresh.status_code == 202
        hold.set()
        finished = _wait_job(client, form.origin, fresh.json()["id"])
        assert finished["status"] == "done"
        assert KEY not in json.dumps(finished)
    finally:
        hold.set()
        mock.close()
        form.close()
