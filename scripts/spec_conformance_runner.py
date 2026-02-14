"""Run section-level spec conformance suites tied to spec/index.html pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from bqn_jax.conformance import aggregate, run_spec_sections, stats_payload, write_json


def _build_markdown(rows) -> str:
    lines = [
        "# Spec Conformance Report",
        "",
        "| Section | URL | Run | Passed | Skipped | Failed | Errors | Pass Rate | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for section, stats in rows:
        rate = "n/a" if stats.pass_rate is None else f"{stats.pass_rate:.2f}%"
        lines.append(
            f"| `{section.page}` ({section.title}) | {section.url} | {stats.tests_run} | {stats.passed} | {stats.skipped} | {stats.failed} | {stats.errors} | {rate} | {stats.status} |"
        )

    overall = aggregate("spec", [stats for _, stats in rows])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total tests run: {overall.tests_run}",
            f"- Passed: {overall.passed}",
            f"- Skipped: {overall.skipped}",
            f"- Failures: {overall.failed}",
            f"- Errors: {overall.errors}",
            f"- Executable pass rate: {'n/a' if overall.pass_rate is None else f'{overall.pass_rate:.2f}%'}",
            f"- Status: `{overall.status}`",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests-dir",
        default="tests",
        help="directory containing unittest test files",
    )
    parser.add_argument(
        "--json-out",
        default="benchmarks/output/conformance/spec_conformance.json",
        help="where to write machine-readable conformance results",
    )
    parser.add_argument(
        "--markdown-out",
        default="benchmarks/output/conformance/spec_conformance.md",
        help="where to write markdown summary",
    )
    args = parser.parse_args()

    rows = run_spec_sections(tests_dir=Path(args.tests_dir))
    overall = aggregate("spec", [stats for _, stats in rows])

    report = _build_markdown(rows)
    print(report)

    write_json(
        Path(args.json_out),
        {
            "sections": [
                {
                    "page": section.page,
                    "title": section.title,
                    "url": section.url,
                    "stats": stats_payload([stats])[0],
                }
                for section, stats in rows
            ],
            "summary": stats_payload([overall])[0],
        },
    )
    Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown_out).write_text(report + "\n", encoding="utf-8")

    return 1 if overall.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
