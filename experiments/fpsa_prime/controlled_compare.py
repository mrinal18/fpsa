"""Compatibility entry point for the runtime-safe controlled comparison."""

from __future__ import annotations

from .controlled_compare_core import *  # noqa: F401,F403
from .controlled_compare_runtime import evaluate, parser, run


if __name__ == "__main__":
    run(parser().parse_args())
