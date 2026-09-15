"""Structural coverage of the shipped identity-findings report."""

from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "docs" / "identity-findings.html"


def _html() -> str:
    assert REPORT.is_file(), f"missing report: {REPORT}"
    return REPORT.read_text(encoding="utf-8")


def test_report_is_html_with_appended_sections() -> None:
    html = _html()
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert html.count("<h2") >= 6
    assert "2026-09-16 00:20Z" in html
    assert "2026-09-16 01:10Z" in html
    assert "2026-09-16 01:25Z" in html
    assert "2026-09-16 02:05Z" in html


def test_report_covers_required_identity_topics() -> None:
    html = _html()
    required = [
        "usage.prompt_tokens",
        "tokenizer differential",
        "JSD",
        "one-token",
        "arXiv:2607.10252",
        "The Tokenizer Is a Fingerprint",
        "arXiv:2407.15847",
        "Kimi-Vendor-Verifier",
        "same-vocab",
        "system prompt",
        "multi-host",
        "quantization",
        "reasoning",
        "EER",
        "aggregator",
        "no official analog found",
        "DeepSeek",
        "GLM",
        "false-positive",
        "uncertain",
        "misconfigured",
        "anthropic-version",
        "Messages",
        "think-mode",
    ]
    missing = [item for item in required if item not in html]
    assert missing == [], f"report missing required coverage: {missing}"


def test_kvv_is_identity_template_not_capability_substitute() -> None:
    html = _html()
    assert "first-party identity contract" in html
    assert "not a unique ID" in html or "not instead of it" in html
    assert "OCRBench" in html
    assert "Fireworks" in html


def test_report_accounts_for_misconfig_and_anthropic_stack_artifacts() -> None:
    html = _html()
    assert "Misconfiguration underperformance" in html
    assert "Visible Anthropic JSON/headers" in html
    assert "stack DNA" in html
    assert "chat_template_kwargs" in html
    assert "ant-rl" in html or "anthropic-ratelimit" in html
