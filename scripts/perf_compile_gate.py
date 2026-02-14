"""Compile-time regression gate for benchmarks/perf_benchmarks.py outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON mapping")
    return payload


def _run_perf(results_path: Path, *, profile: str) -> None:
    cmd = [
        sys.executable,
        "benchmarks/perf_benchmarks.py",
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


def _compile_index(payload: dict[str, Any]) -> dict[str, float]:
    rows = payload.get("jax_rows")
    if not isinstance(rows, list):
        raise ValueError("payload has no jax_rows")
    out: dict[str, float] = {}
    for row in rows:
        name = str(row["name"])
        out[name] = float(row["compile_ms"])
    return out


def _emit_baseline(results_path: Path, baseline_path: Path) -> None:
    payload = _load_payload(results_path)
    baseline = {
        "source_results": str(results_path),
        "compile_ms_by_case": _compile_index(payload),
        "host_contract": {
            "backend": payload.get("host", {}).get("backend"),
            "thread_env": payload.get("host", {}).get("thread_env"),
        },
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def _gate(
    *,
    results_payload: dict[str, Any],
    baseline: dict[str, Any],
    growth_limit: float,
    enforce_host_contract: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    current = _compile_index(results_payload)
    report_rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for name, base_compile_ms in dict(baseline.get("compile_ms_by_case", {})).items():
        cur = current.get(name)
        if cur is None:
            violations.append(f"missing compile case in results: {name}")
            continue
        growth = cur / float(base_compile_ms) if float(base_compile_ms) > 0 else float("inf")
        report_rows.append(
            {
                "name": name,
                "baseline_compile_ms": float(base_compile_ms),
                "current_compile_ms": float(cur),
                "growth": float(growth),
            }
        )
        if growth > growth_limit:
            violations.append(f"{name} compile growth {growth:.3f}x exceeds {growth_limit:.3f}x")

    if enforce_host_contract:
        expected = dict(baseline.get("host_contract", {}))
        got = dict(results_payload.get("host", {}))
        exp_backend = expected.get("backend")
        got_backend = got.get("backend")
        if exp_backend and got_backend and exp_backend != got_backend:
            violations.append(f"backend mismatch baseline={exp_backend!r} current={got_backend!r}")

    return report_rows, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="benchmarks/output/conformance/perf_compile_candidate.json")
    parser.add_argument("--baseline", default="benchmarks/baselines/perf_compile_baseline.json")
    parser.add_argument("--growth-limit", type=float, default=1.5)
    parser.add_argument("--profile", default="full", help="perf benchmark profile when --run-benchmark is set")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--emit-baseline", action="store_true")
    parser.add_argument("--enforce-host-contract", action="store_true", default=False)
    parser.add_argument("--json-out", default="benchmarks/output/conformance/perf_compile_gate.json")
    args = parser.parse_args()

    results_path = Path(args.results)
    baseline_path = Path(args.baseline)
    if args.run_benchmark:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        _run_perf(results_path, profile=args.profile)

    if not results_path.exists():
        raise SystemExit(f"results not found: {results_path}")

    if args.emit_baseline:
        _emit_baseline(results_path, baseline_path)
        print(f"Wrote baseline: {baseline_path}")
        return 0

    if not baseline_path.exists():
        raise SystemExit(f"baseline not found: {baseline_path}")

    payload = _load_payload(results_path)
    baseline = _load_payload(baseline_path)
    report_rows, violations = _gate(
        results_payload=payload,
        baseline=baseline,
        growth_limit=float(args.growth_limit),
        enforce_host_contract=bool(args.enforce_host_contract),
    )

    print("# perf compile-time regression gate")
    print()
    print("| Case | Baseline compile ms | Current compile ms | Growth |")
    print("|---|---:|---:|---:|")
    for row in report_rows:
        print(
            f"| `{row['name']}` | {row['baseline_compile_ms']:.6f} | "
            f"{row['current_compile_ms']:.6f} | {row['growth']:.3f}x |"
        )
    print()
    if violations:
        print("violations:")
        for item in violations:
            print(f"- {item}")
    else:
        print("all thresholds satisfied")

    out = {
        "results": str(results_path),
        "baseline": str(baseline_path),
        "growth_limit": float(args.growth_limit),
        "report_rows": report_rows,
        "violations": violations,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote gate report: {out_path}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
