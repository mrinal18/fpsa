"""Compatibility entry point for the portable FPSA-Prime verification harness.

The original numerical checks live in ``verification_core``.  The public
``run_verification`` and ``gmres_check`` symbols use portable pass criteria:
corrected GMRES must converge and remain stable as its budget grows, while the
exact failure pattern of the legacy v0 solver is recorded only as a diagnostic.
"""

from __future__ import annotations

import json

from . import verification_core as _core
from .verification_core import *  # noqa: F401,F403
from .verification_portable import gmres_check, run_verification


if __name__ == "__main__":
    args = _core.parser().parse_args()
    result = run_verification(
        output_dir=args.output_dir,
        device=args.device,
        quick=not args.full,
        run_tests=not args.skip_tests,
        run_training=args.run_training,
    )
    print(json.dumps(result, indent=2))
