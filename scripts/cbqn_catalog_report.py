"""Emit a categorized report of imported CBQN cases."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from bqn_jax.cbqn_catalog import CATALOG
from bqn_jax.conformance import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        default="benchmarks/output/conformance/cbqn_catalog.json",
        help="where to write machine-readable catalog summary",
    )
    args = parser.parse_args()

    by_app = Counter(case.applicability for case in CATALOG)
    by_outcome = Counter(case.expected_outcome for case in CATALOG)
    by_source = Counter(case.source_file for case in CATALOG)

    print("CBQN import catalog")
    print("-------------------")
    print(f"total cases: {len(CATALOG)}")
    print("applicability:")
    for key in sorted(by_app):
        print(f"  - {key}: {by_app[key]}")
    print("expected outcomes:")
    for key in sorted(by_outcome):
        print(f"  - {key}: {by_outcome[key]}")
    print("source files:")
    for key in sorted(by_source):
        print(f"  - {key}: {by_source[key]}")

    write_json(
        Path(args.json_out),
        {
            "total_cases": len(CATALOG),
            "by_applicability": dict(sorted(by_app.items())),
            "by_expected_outcome": dict(sorted(by_outcome.items())),
            "by_source_file": dict(sorted(by_source.items())),
            "cases": [case.__dict__ for case in CATALOG],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
