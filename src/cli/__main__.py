"""CLI dispatcher — routes ``python -m src.cli <subcommand>`` to the
correct module (ingest, chunk, embed, retrieve, evaluate).

When run *without* a subcommand, defaults to **ingest** for
backwards-compatibility.
"""

from __future__ import annotations

import sys


_SUBCOMMANDS = {
    "ingest":   "src.cli.ingest",
    "chunk":    "src.cli.chunk",
    "embed":    "src.cli.embed",
    "retrieve": "src.cli.retrieve",
    "evaluate": "src.cli.evaluate",
}


def _usage() -> str:
    return (
        "usage: python -m src.cli <subcommand> [options]\n\n"
        "subcommands:\n"
        + "\n".join(f"  {k:<12} " for k in _SUBCOMMANDS)
    )


def main() -> int:
    # No args or first arg looks like a flag → default to ingest
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        from .ingest import main as _ingest_main
        return _ingest_main()

    sub = sys.argv[1]

    if sub in ("-h", "--help"):
        print(_usage())
        return 0

    if sub not in _SUBCOMMANDS:
        print(f"Unknown subcommand: {sub}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 1

    # Remove the subcommand token so the sub-module sees clean argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if sub == "ingest":
        from .ingest import main as _main
    elif sub == "chunk":
        from .chunk import main as _main
    elif sub == "embed":
        from .embed import main as _main
    elif sub == "retrieve":
        from .retrieve import main as _main
    elif sub == "evaluate":
        from .evaluate import main as _main
    else:
        return 1

    return _main()


sys.exit(main())
