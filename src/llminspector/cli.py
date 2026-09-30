"""CLI entry point. Parses arguments and dispatches subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llminspector import __version__

MIDDLE_DROP_HELP = (
    "Repeat one chat-completions body against an OpenAI-compatible endpoint. "
    "A DROP means start and end are PRESENT and mid is MISSING while the proxy "
    "can still bill the full prompt. Pass the key with --api-key or "
    "LLMINSPECTOR_API_KEY and the endpoint with --base-url or LLMINSPECTOR_BASE_URL. "
    "There is no default endpoint and the key is never printed."
)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llminspector",
        description="Identify which LLM an OpenAI-compatible API is actually serving.",
    )
    parser.add_argument("--version", action="version", version=f"llminspector {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    middle = subparsers.add_parser(
        "middle-drop",
        help="Repeat one chat-completions body and report a middle-context drop.",
        description=MIDDLE_DROP_HELP,
    )
    middle.add_argument(
        "--api-key",
        default=None,
        help="API key. Falls back to LLMINSPECTOR_API_KEY. Required. Never printed.",
    )
    middle.add_argument(
        "--base-url",
        default=None,
        help=(
            "Endpoint base URL. Falls back to LLMINSPECTOR_BASE_URL. Required. "
            "No default endpoint."
        ),
    )
    middle.add_argument(
        "--models",
        default="glm-5.3",
        help="Comma-separated model names. Empty entries are dropped.",
    )
    middle.add_argument(
        "--efforts",
        default="low",
        help="Comma-separated reasoning_effort values, passed through (none, max, ...).",
    )
    middle.add_argument("--iterations", type=int, default=30, help="Calls per model and effort.")
    middle.add_argument(
        "--filler-chars",
        type=int,
        default=423588,
        help="Filler characters placed around the document marker.",
    )
    middle.add_argument(
        "--timeout",
        type=float,
        default=150,
        help="Wall-clock seconds for each request.",
    )
    middle.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Completion cap. 0 omits the field.",
    )
    middle.add_argument(
        "--min-interval",
        type=float,
        default=4.2,
        help="Seconds between request starts.",
    )
    middle.add_argument(
        "--out",
        default="middle-drop.jsonl",
        help="JSONL path for one object per call.",
    )
    return parser


def _cmd_middle_drop(args: argparse.Namespace) -> int:
    from llminspector.middle_drop import (
        ConfigError,
        MiddleDropOptions,
        redact,
        resolve_api_key,
        resolve_base_url,
        run_middle_drop,
    )

    secret = ""
    try:
        api_key = resolve_api_key(args.api_key)
        secret = api_key
        base_url = resolve_base_url(args.base_url)
        options = MiddleDropOptions(
            base_url=base_url,
            api_key=api_key,
            models=_split_csv(args.models),
            efforts=_split_csv(args.efforts),
            iterations=args.iterations,
            filler_chars=args.filler_chars,
            timeout=float(args.timeout),
            max_tokens=args.max_tokens or None,
            min_interval=float(args.min_interval),
            out=Path(args.out),
        )
        records = run_middle_drop(options)
    except ConfigError as exc:
        print(redact(str(exc), secret), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {redact(str(exc), secret)}", file=sys.stderr)
        return 1
    return 0 if records else 2


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "middle-drop":
        return _cmd_middle_drop(args)
    parser.print_help()
    print(
        "\nllminspector is initialized. Tokenizer, behavioral, and stack scanners are not shipped yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
