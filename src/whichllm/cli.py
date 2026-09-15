"""CLI entry point. Scanners are not implemented yet."""

from __future__ import annotations

import argparse
import sys

from whichllm import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whichllm",
        description="Identify which LLM an OpenAI-compatible API is actually serving.",
    )
    parser.add_argument("--version", action="version", version=f"whichllm {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    print(
        "\nwhichllm is initialized. Tokenizer, behavioral, and stack scanners are not shipped yet.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
