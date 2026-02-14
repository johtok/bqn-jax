"""Regression gate for the top slowest bench-md kernels at n=10000.

The gate compares current results against a baseline snapshot and checks:
- normalized ratio growth against a stable reference case,
- absolute mean growth against baseline means,
- variance and sampling quality (CV and sample count),
- host benchmark contract consistency (backend / affinity / thread env).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class CaseKey:
    section: str
    name: str
    n: int


def _parse_case(case: str, *, n: int) -> CaseKey:
    section, sep, name = case.partition(":")
    if not sep or not section or not name:
        raise ValueError(f"Invalid case format {case!r}; expected 'section:name'")
    return CaseKey(section=section, name=name, n=n)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark file {path} must decode to a mapping")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_payload(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Benchmark file {path} does not contain a 'rows' list")
    return rows


def _row_index(rows: list[dict[str, Any]]) -> dict[CaseKey, dict[str, Any]]:
    index: dict[CaseKey, dict[str, Any]] = {}
    for row in rows:
        key = CaseKey(
            section=str(row["section"]),
            name=str(row["name"]),
            n=int(row["n"]),
        )
        index[key] = row
    return index


def _run_bench_md(results_path: Path, *, profile: str) -> None:
    cmd = [
        sys.executable,
        "benchmarks/monad_dyad_benchmarks.py",
        "--profile",
        profile,
        "--json-out",
        str(results_path),
    ]
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src{os.pathsep}{py_path}" if py_path else "src"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("JAX_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, env=env)


def _emit_baseline(
    results_path: Path,
    baseline_path: Path,
    *,
    n: int,
    top_k: int,
    reference_case: CaseKey,
) -> None:
    payload = _load_payload(results_path)
    rows = payload.get("rows", [])
    index = _row_index(rows)
    ref = index.get(reference_case)
    if ref is None:
        raise ValueError(f"Reference case {reference_case.section}:{reference_case.name} (n={n}) not found")
    ref_mean = float(ref["mean_ms"])
    if ref_mean <= 0:
        raise ValueError("Reference case mean_ms must be > 0")

    at_n = [row for row in rows if int(row["n"]) == n]
    if not at_n:
        raise ValueError(f"No rows found for n={n}")

    top = sorted(at_n, key=lambda row: float(row["mean_ms"]), reverse=True)[:top_k]
    baseline = {
        "source_results": str(results_path),
        "n": n,
        "reference_case": asdict(reference_case),
        "host_contract": {
            "backend": payload.get("host", {}).get("backend"),
            "active_cpu_affinity": payload.get("host", {}).get("active_cpu_affinity"),
            "thread_env": payload.get("host", {}).get("thread_env"),
        },
        "top_cases": [
            {
                "section": str(row["section"]),
                "name": str(row["name"]),
                "n": int(row["n"]),
                "baseline_mean_ms": float(row["mean_ms"]),
                "baseline_ratio": float(row["mean_ms"]) / ref_mean,
                "baseline_cv_pct": float(row.get("cv_pct", 0.0)),
                "baseline_samples": int(row.get("samples", 0)),
            }
            for row in top
        ],
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def _gate(
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    ratio_growth_limit: float,
    absolute_growth_limit: float,
    cv_growth_limit: float,
    max_cv_pct: float,
    min_sample_ratio: float,
    enforce_host_contract: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    n = int(baseline["n"])
    ref_case = CaseKey(
        section=str(baseline["reference_case"]["section"]),
        name=str(baseline["reference_case"]["name"]),
        n=n,
    )
    index = _row_index(rows)
    ref_row = index.get(ref_case)
    if ref_row is None:
        raise ValueError(f"Reference case missing in results: {ref_case.section}:{ref_case.name} (n={n})")
    ref_mean = float(ref_row["mean_ms"])
    if ref_mean <= 0:
        raise ValueError("Reference case mean_ms must be > 0 in results")

    report_rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for item in baseline["top_cases"]:
        case = CaseKey(section=str(item["section"]), name=str(item["name"]), n=int(item["n"]))
        row = index.get(case)
        if row is None:
            violations.append(f"missing case {case.section}:{case.name} (n={case.n}) in candidate results")
            continue

        baseline_mean = float(item["baseline_mean_ms"])
        baseline_ratio = float(item["baseline_ratio"])
        baseline_cv = float(item.get("baseline_cv_pct", 0.0))
        baseline_samples = int(item.get("baseline_samples", 0))
        current_mean = float(row["mean_ms"])
        current_cv = float(row.get("cv_pct", 0.0))
        current_samples = int(row.get("samples", 0))
        current_ratio = current_mean / ref_mean
        ratio_growth = current_ratio / baseline_ratio if baseline_ratio > 0 else float("inf")
        absolute_growth = current_mean / baseline_mean if baseline_mean > 0 else float("inf")
        cv_growth = current_cv / baseline_cv if baseline_cv > 0 else 1.0

        report_rows.append(
            {
                "section": case.section,
                "name": case.name,
                "n": case.n,
                "baseline_mean_ms": baseline_mean,
                "current_mean_ms": current_mean,
                "baseline_ratio": baseline_ratio,
                "current_ratio": current_ratio,
                "ratio_growth": ratio_growth,
                "absolute_growth": absolute_growth,
                "baseline_cv_pct": baseline_cv,
                "current_cv_pct": current_cv,
                "cv_growth": cv_growth,
                "baseline_samples": baseline_samples,
                "current_samples": current_samples,
            }
        )

        if ratio_growth > ratio_growth_limit:
            violations.append(
                f"{case.section}:{case.name} ratio growth {ratio_growth:.3f}x exceeds {ratio_growth_limit:.3f}x"
            )
        if absolute_growth > absolute_growth_limit:
            violations.append(
                f"{case.section}:{case.name} absolute growth {absolute_growth:.3f}x exceeds {absolute_growth_limit:.3f}x"
            )
        if current_cv > max(max_cv_pct, baseline_cv * cv_growth_limit):
            violations.append(
                f"{case.section}:{case.name} cv {current_cv:.2f}% exceeds thresholds "
                f"(max_cv={max_cv_pct:.2f}%, baseline_growth_limit={cv_growth_limit:.2f}x)"
            )
        if baseline_samples > 0 and current_samples < int(math.ceil(baseline_samples * min_sample_ratio)):
            violations.append(
                f"{case.section}:{case.name} samples {current_samples} below required "
                f"{math.ceil(baseline_samples * min_sample_ratio)}"
            )

    if enforce_host_contract:
        expected = baseline.get("host_contract", {})
        got = payload.get("host", {})
        exp_backend = expected.get("backend")
        got_backend = got.get("backend")
        if exp_backend and got_backend and exp_backend != got_backend:
            violations.append(f"host backend mismatch baseline={exp_backend!r} current={got_backend!r}")

        exp_aff = expected.get("active_cpu_affinity")
        got_aff = got.get("active_cpu_affinity")
        if exp_aff and got_aff and list(exp_aff) != list(got_aff):
            violations.append(f"host cpu affinity mismatch baseline={exp_aff} current={got_aff}")

        exp_threads = expected.get("thread_env")
        got_threads = got.get("thread_env")
        if isinstance(exp_threads, dict) and isinstance(got_threads, dict):
            for key, exp_val in exp_threads.items():
                cur_val = got_threads.get(key)
                if cur_val != exp_val:
                    violations.append(
                        f"thread env mismatch for {key}: baseline={exp_val!r} current={cur_val!r}"
                    )

    return report_rows, violations


def _print_report(
    *,
    results_path: Path,
    baseline_path: Path,
    ratio_growth_limit: float,
    absolute_growth_limit: float,
    report_rows: list[dict[str, Any]],
    violations: list[str],
) -> None:
    print("# bench-md top-k regression gate")
    print()
    print(f"- results: `{results_path}`")
    print(f"- baseline: `{baseline_path}`")
    print(f"- ratio growth limit: `{ratio_growth_limit:.3f}x`")
    print(f"- absolute growth limit: `{absolute_growth_limit:.3f}x`")
    print()
    print("| Case | Baseline ms | Current ms | Baseline ratio | Current ratio | Ratio growth | Absolute growth | Baseline CV | Current CV |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report_rows:
        case = f"{row['section']}:{row['name']}"
        print(
            f"| `{case}` | {row['baseline_mean_ms']:.6f} | {row['current_mean_ms']:.6f} |"
            f" {row['baseline_ratio']:.3f} | {row['current_ratio']:.3f} |"
            f" {row['ratio_growth']:.3f}x | {row['absolute_growth']:.3f}x |"
            f" {row['baseline_cv_pct']:.2f}% | {row['current_cv_pct']:.2f}% |"
        )
    print()
    if violations:
        print("violations:")
        for item in violations:
            print(f"- {item}")
    else:
        print("all thresholds satisfied")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="", help="bench-md JSON results path")
    parser.add_argument(
        "--baseline",
        default="benchmarks/baselines/bench_md_top5_baseline.json",
        help="baseline JSON for top-k regression checks",
    )
    parser.add_argument("--n", type=int, default=10000, help="problem size for baseline extraction")
    parser.add_argument("--top-k", type=int, default=5, help="number of slow cases to track in baseline")
    parser.add_argument(
        "--reference-case",
        default="modifier:constant",
        help="reference case for hardware-normalized ratios in format section:name",
    )
    parser.add_argument("--ratio-growth-limit", type=float, default=1.30, help="max allowed normalized ratio growth")
    parser.add_argument("--absolute-growth-limit", type=float, default=1.50, help="max allowed absolute mean growth")
    parser.add_argument("--cv-growth-limit", type=float, default=1.75, help="max allowed CV growth vs baseline")
    parser.add_argument("--max-cv-pct", type=float, default=20.0, help="hard CV ceiling for tracked cases")
    parser.add_argument("--min-sample-ratio", type=float, default=1.0, help="minimum current/baseline sample ratio")
    parser.add_argument(
        "--enforce-host-contract",
        action="store_true",
        default=False,
        help="fail when backend/affinity/thread contract differs from baseline",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="run benchmarks/monad_dyad_benchmarks.py to produce --results before gating",
    )
    parser.add_argument("--profile", default="stable", help="bench-md profile when --run-benchmark is used")
    parser.add_argument(
        "--emit-baseline",
        action="store_true",
        help="emit/update baseline from --results and exit",
    )
    parser.add_argument(
        "--json-out",
        default="benchmarks/output/conformance/bench_md_top5_gate.json",
        help="machine-readable gate report output path",
    )
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else Path("benchmarks/output/conformance/bench_md_candidate.json")
    baseline_path = Path(args.baseline)
    ref_case = _parse_case(args.reference_case, n=args.n)

    if args.run_benchmark:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        _run_bench_md(results_path, profile=args.profile)

    if not results_path.exists():
        raise SystemExit(f"Results file does not exist: {results_path}")

    if args.emit_baseline:
        _emit_baseline(results_path, baseline_path, n=args.n, top_k=args.top_k, reference_case=ref_case)
        print(f"Wrote baseline: {baseline_path}")
        return 0

    if not baseline_path.exists():
        raise SystemExit(f"Baseline file does not exist: {baseline_path}")

    payload = _load_payload(results_path)
    rows = _load_rows(results_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report_rows, violations = _gate(
        payload=payload,
        rows=rows,
        baseline=baseline,
        ratio_growth_limit=float(args.ratio_growth_limit),
        absolute_growth_limit=float(args.absolute_growth_limit),
        cv_growth_limit=float(args.cv_growth_limit),
        max_cv_pct=float(args.max_cv_pct),
        min_sample_ratio=float(args.min_sample_ratio),
        enforce_host_contract=bool(args.enforce_host_contract),
    )

    _print_report(
        results_path=results_path,
        baseline_path=baseline_path,
        ratio_growth_limit=float(args.ratio_growth_limit),
        absolute_growth_limit=float(args.absolute_growth_limit),
        report_rows=report_rows,
        violations=violations,
    )

    out_payload = {
        "results": str(results_path),
        "baseline": str(baseline_path),
        "ratio_growth_limit": float(args.ratio_growth_limit),
        "absolute_growth_limit": float(args.absolute_growth_limit),
        "cv_growth_limit": float(args.cv_growth_limit),
        "max_cv_pct": float(args.max_cv_pct),
        "min_sample_ratio": float(args.min_sample_ratio),
        "enforce_host_contract": bool(args.enforce_host_contract),
        "report_rows": report_rows,
        "violations": violations,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Wrote gate report: {out_path}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
