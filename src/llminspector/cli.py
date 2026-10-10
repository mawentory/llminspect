"""CLI entry point. Parses arguments and dispatches subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llminspector import __version__

PELICAN_HELP = (
    "Ask for one SVG of a pelican riding a bicycle. Each call is a single "
    "chat completion with no tools. max_tokens is omitted unless --max-tokens "
    "is positive. A 429 lowers the in-flight cap by one, down to 1, then retries. "
    "Pass the key with --api-key or LLMINSPECTOR_API_KEY and the endpoint with "
    "--base-url or LLMINSPECTOR_BASE_URL. There is no default endpoint and the "
    "key is never printed."
)

MIDDLE_DROP_HELP = (
    "Repeat one chat-completions body against an OpenAI-compatible endpoint. "
    "A DROP means start and end are PRESENT and mid is MISSING while the proxy "
    "can still bill the full prompt. Pass the key with --api-key or "
    "LLMINSPECTOR_API_KEY and the endpoint with --base-url or LLMINSPECTOR_BASE_URL. "
    "There is no default endpoint and the key is never printed."
)

UI_HELP = (
    "Open a local form for a pelican drawing and a context-marker probe. "
    "Listens on 127.0.0.1 port 8787. Enter the API key, model, and base URL "
    "in the form. Choose pelican, marker, or both, with a run count for each "
    "and one concurrency cap. Marker also takes a context size in characters. "
    "The key is sent only to the endpoint and is not written to the page."
)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llminspect",
        description="Identify which LLM an OpenAI-compatible API is actually serving.",
    )
    parser.add_argument("--version", action="version", version=f"llminspect {__version__}")
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
            "Endpoint base URL. Falls back to LLMINSPECTOR_BASE_URL. Required. No default endpoint."
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
    pelican = subparsers.add_parser(
        "pelican",
        help="Draw a pelican on a bicycle and open the drawings.",
        description=PELICAN_HELP,
    )
    pelican.add_argument(
        "--api-key",
        default=None,
        help="API key. Falls back to LLMINSPECTOR_API_KEY. Required. Never printed.",
    )
    pelican.add_argument(
        "--base-url",
        default=None,
        help=(
            "Endpoint base URL. Falls back to LLMINSPECTOR_BASE_URL. Required. No default endpoint."
        ),
    )
    pelican.add_argument(
        "--models",
        default="",
        help="Comma-separated model names. Required. Empty entries are dropped.",
    )
    pelican.add_argument(
        "--efforts",
        default="max",
        help="Comma-separated reasoning_effort values. Default is max.",
    )
    pelican.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Calls per model and effort. Default is 5.",
    )
    pelican.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="In-flight calls. A 429 lowers this by one, down to 1.",
    )
    pelican.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Tries per call, including 429 retries.",
    )
    pelican.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Wall-clock seconds for each request.",
    )
    pelican.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Completion cap. 0 omits the field.",
    )
    pelican.add_argument(
        "--out",
        default="pelican",
        help="Directory for the SVG files and index.html.",
    )
    pelican.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local port for the gallery. Used unless --no-serve is set.",
    )
    pelican.add_argument(
        "--no-serve",
        action="store_true",
        help="Write the gallery and exit without listening.",
    )
    ui = subparsers.add_parser(
        "ui",
        help="Open the local form for pelican and marker tests.",
        description=UI_HELP,
    )
    ui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to listen on. Default is 127.0.0.1.",
    )
    ui.add_argument(
        "--port",
        type=int,
        default=8787,
        help="Port to listen on. Default is 8787.",
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


def _cmd_pelican(args: argparse.Namespace) -> int:
    from llminspector.middle_drop import ConfigError, redact, resolve_api_key, resolve_base_url
    from llminspector.pelican import PelicanOptions, run_pelican, serve_gallery

    secret = ""
    try:
        api_key = resolve_api_key(args.api_key)
        secret = api_key
        base_url = resolve_base_url(args.base_url)
        options = PelicanOptions(
            base_url=base_url,
            api_key=api_key,
            models=_split_csv(args.models),
            efforts=_split_csv(args.efforts),
            iterations=args.iterations,
            timeout=float(args.timeout),
            max_tokens=args.max_tokens or None,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            out=Path(args.out),
        )
        samples = run_pelican(options)
    except ConfigError as exc:
        print(redact(str(exc), secret), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {redact(str(exc), secret)}", file=sys.stderr)
        return 1
    if not any(sample.ok for sample in samples):
        return 2
    if args.no_serve:
        return 0
    return serve_gallery(options.out, port=args.port)


def _cmd_ui(args: argparse.Namespace) -> int:
    from llminspector.ui import serve_ui

    return serve_ui(args.host, args.port)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "middle-drop":
        return _cmd_middle_drop(args)
    if getattr(args, "command", None) == "pelican":
        return _cmd_pelican(args)
    if getattr(args, "command", None) == "ui":
        return _cmd_ui(args)
    parser.print_help()
    print(
        "\nllminspect is initialized. Tokenizer, behavioral, and stack scanners are not shipped yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
