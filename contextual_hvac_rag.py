"""Top-level CLI for additive offline tooling."""

from __future__ import annotations

import argparse

from eval.run import add_eval_subcommand


def main(argv: list[str] | None = None) -> int:
    """Dispatch the top-level CLI."""

    parser = argparse.ArgumentParser(prog="contextual-hvac-rag")
    subparsers = parser.add_subparsers(dest="command")
    add_eval_subcommand(subparsers)

    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
