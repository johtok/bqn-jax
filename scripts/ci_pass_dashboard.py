"""Generate CI pass-rate dashboard across key suite groups."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bqn_jax.conformance import (
    aggregate,
    run_patterns,
    run_spec_sections,
    stats_payload,
    stats_to_markdown_table,
    write_json,
)


def _suite_definitions() -> dict[str, tuple[str, ...]]:
    return {
        "runtime": (
            "test_runtime_language_semantics.py",
            "test_runtime_value_model.py",
            "test_system_surface.py",
            "test_inferred_properties.py",
            "test_primitive_modifier_matrix.py",
        ),
        "jax-ir": ("test_jax_ir_pipeline.py", "test_jax_ir_equivalence.py"),
        "diff-cbqn": ("test_cbqn_differential_matrix.py",),
    }


def _build_markdown(spec_summary, section_rows, group_rows) -> str:
    lines = [
        "# CI Pass-Rate Dashboard",
        "",
        "## Group Summary",
        "",
        stats_to_markdown_table([spec_summary, *group_rows]),
        "",
        "## Spec Section Breakdown",
        "",
        "| Section | URL | Run | Passed | Skipped | Failed | Errors | Pass Rate | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for section, stats in section_rows:
        rate = "n/a" if stats.pass_rate is None else f"{stats.pass_rate:.2f}%"
        lines.append(
            f"| `{section.page}` | {section.url} | {stats.tests_run} | {stats.passed} | {stats.skipped} | {stats.failed} | {stats.errors} | {rate} | {stats.status} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="tests", help="directory containing unittest test files")
    parser.add_argument(
        "--json-out",
        default="benchmarks/output/conformance/ci_dashboard.json",
        help="where to write machine-readable dashboard data",
    )
    parser.add_argument(
        "--markdown-out",
        default="benchmarks/output/conformance/ci_dashboard.md",
        help="where to write markdown dashboard output",
    )
    parser.add_argument(
        "--require-cbqn-diff",
        action="store_true",
        help="treat fully-skipped diff-cbqn suite as a failure",
    )
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir)
    section_rows = run_spec_sections(tests_dir=tests_dir)
    spec_summary = aggregate("spec", [stats for _, stats in section_rows])

    group_rows = []
    for name, patterns in _suite_definitions().items():
        group_rows.append(run_patterns(name, patterns, tests_dir=tests_dir))

    dashboard_md = _build_markdown(spec_summary, section_rows, group_rows)
    print(dashboard_md)

    Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown_out).write_text(dashboard_md + "\n", encoding="utf-8")
    write_json(
        Path(args.json_out),
        {
            "summary": stats_payload([spec_summary, *group_rows]),
            "spec_sections": [
                {
                    "page": section.page,
                    "title": section.title,
                    "url": section.url,
                    "stats": stats_payload([stats])[0],
                }
                for section, stats in section_rows
            ],
        },
    )

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(dashboard_md + "\n")

    fail = spec_summary.status == "fail" or any(row.status == "fail" for row in group_rows)
    if args.require_cbqn_diff:
        for row in group_rows:
            if row.name == "diff-cbqn" and row.status == "skipped":
                fail = True
                break

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
