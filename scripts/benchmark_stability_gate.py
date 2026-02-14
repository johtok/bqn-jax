"""Benchmark stability gate with explicit regression thresholds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys

from bqn_jax.conformance import write_json


def _load_bencharray_module():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    module_path = repo_root / "benchmarks" / "bencharray_recreated.py"
    spec = importlib.util.spec_from_file_location("bencharray_recreated_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class CaseThreshold:
    max_time_growth: float
    max_ns_per_element_growth: float
    max_ns_per_element: float


@dataclass(frozen=True)
class CaseMetrics:
    name: str
    mode: str
    size_kind: str
    sizes: list[int]
    seconds: list[float]
    ns_per_element: list[float]
    max_time_growth: float
    max_ns_per_element_growth: float


DEFAULT_CASE_THRESHOLDS = {
    "times_plus": CaseThreshold(max_time_growth=4.0, max_ns_per_element_growth=3.0, max_ns_per_element=30_000.0),
    "sum": CaseThreshold(max_time_growth=4.0, max_ns_per_element_growth=3.0, max_ns_per_element=60_000.0),
    "matrix_plus": CaseThreshold(max_time_growth=8.0, max_ns_per_element_growth=3.5, max_ns_per_element=50_000.0),
    "reshape_square": CaseThreshold(max_time_growth=8.0, max_ns_per_element_growth=4.0, max_ns_per_element=150_000.0),
    "sort_up": CaseThreshold(max_time_growth=6.0, max_ns_per_element_growth=4.0, max_ns_per_element=300_000.0),
}


def _growth(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    ratios = [values[i] / values[i - 1] for i in range(1, len(values)) if values[i - 1] > 0]
    if not ratios:
        return 1.0
    return max(ratios)


def _collect_metrics(case, *, flat_sizes: list[int], matrix_sizes: list[int], run_case) -> CaseMetrics:
    rows = run_case(case, flat_sizes=flat_sizes, matrix_sizes=matrix_sizes)
    sizes = [int(r.size) for r in rows]
    seconds = [float(r.seconds) for r in rows]
    ns_per_element = [float(r.seconds) * 1e9 / max(int(r.elements), 1) for r in rows]
    return CaseMetrics(
        name=case.name,
        mode=case.mode,
        size_kind=case.size_kind,
        sizes=sizes,
        seconds=seconds,
        ns_per_element=ns_per_element,
        max_time_growth=_growth(seconds),
        max_ns_per_element_growth=_growth(ns_per_element),
    )


def _render(metrics: list[CaseMetrics]) -> str:
    lines = [
        "| Case | Mode | Kind | max time growth | max ns/elem growth | max ns/elem |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row.name}` | `{row.mode}` | `{row.size_kind}` | {row.max_time_growth:.3f}x | {row.max_ns_per_element_growth:.3f}x | {max(row.ns_per_element):.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-min-exp", type=int, default=8, help="minimum exponent for flat sizes (2^exp)")
    parser.add_argument("--flat-max-exp", type=int, default=11, help="maximum exponent for flat sizes (2^exp)")
    parser.add_argument("--matrix-min-exp", type=int, default=3, help="minimum exponent for matrix side (2^exp)")
    parser.add_argument("--matrix-max-exp", type=int, default=6, help="maximum exponent for matrix side (2^exp)")
    parser.add_argument(
        "--json-out",
        default="benchmarks/output/conformance/benchmark_gate.json",
        help="where to write machine-readable gate results",
    )
    args = parser.parse_args()

    bencharray = _load_bencharray_module()
    by_name = {case.name: case for case in bencharray.CASES}
    selected = [by_name[name] for name in DEFAULT_CASE_THRESHOLDS]

    flat_sizes = bencharray._powers_of_two(args.flat_min_exp, args.flat_max_exp)
    matrix_sizes = bencharray._powers_of_two(args.matrix_min_exp, args.matrix_max_exp)
    metrics = [
        _collect_metrics(case, flat_sizes=flat_sizes, matrix_sizes=matrix_sizes, run_case=bencharray._run_case)
        for case in selected
    ]

    violations: list[str] = []
    for row in metrics:
        threshold = DEFAULT_CASE_THRESHOLDS[row.name]
        if row.max_time_growth > threshold.max_time_growth:
            violations.append(
                f"{row.name}: max time growth {row.max_time_growth:.3f}x exceeds {threshold.max_time_growth:.3f}x"
            )
        if row.max_ns_per_element_growth > threshold.max_ns_per_element_growth:
            violations.append(
                f"{row.name}: max ns/elem growth {row.max_ns_per_element_growth:.3f}x exceeds {threshold.max_ns_per_element_growth:.3f}x"
            )
        max_nspe = max(row.ns_per_element)
        if max_nspe > threshold.max_ns_per_element:
            violations.append(f"{row.name}: max ns/elem {max_nspe:.1f} exceeds {threshold.max_ns_per_element:.1f}")

    markdown = _render(metrics)
    print("# Benchmark Stability Gate")
    print()
    print(markdown)
    print()
    if violations:
        print("Violations:")
        for item in violations:
            print(f"- {item}")
    else:
        print("All benchmark thresholds satisfied.")

    write_json(
        Path(args.json_out),
        {
            "flat_exp_range": [args.flat_min_exp, args.flat_max_exp],
            "matrix_exp_range": [args.matrix_min_exp, args.matrix_max_exp],
            "thresholds": {
                key: {
                    "max_time_growth": value.max_time_growth,
                    "max_ns_per_element_growth": value.max_ns_per_element_growth,
                    "max_ns_per_element": value.max_ns_per_element,
                }
                for key, value in DEFAULT_CASE_THRESHOLDS.items()
            },
            "metrics": [asdict(row) for row in metrics],
            "violations": violations,
        },
    )

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
