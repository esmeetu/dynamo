#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate parity charts for parser stages from one common entrypoint.

Examples:
    python3 tests/parity/generate_parity_chart.py parser --html > tests/parity/parser/PARITY.html
    python3 tests/parity/generate_parity_chart.py reasoning --html > tests/parity/reasoning/PARITY.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage",
        choices=("parser", "reasoning"),
        help="Parity stage to render. Stage-specific flags follow this argument.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.stage == "parser":
        from tests.parity.parser import chart
    else:
        from tests.parity.reasoning import chart

    chart.main(rest)


if __name__ == "__main__":
    main()
