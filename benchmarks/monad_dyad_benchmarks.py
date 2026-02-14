"""Fast monadic/dyadic/modifier benchmark for optimization loops."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression
from _bench_utils import configure_cpu_affinity_from_env, host_metadata

PROFILE_CONFIG = {
    "quick": {
        "samples": 3,
        "warmup": 1,
        "target_sample_ms": 8.0,
        "min_repeats": 32,
        "cv_target_pct": 25.0,
        "max_samples": 7,
    },
    "full": {
        "samples": 5,
        "warmup": 2,
        "target_sample_ms": 20.0,
        "min_repeats": 64,
        "cv_target_pct": 18.0,
        "max_samples": 11,
    },
    "stable": {
        "samples": 9,
        "warmup": 3,
        "target_sample_ms": 60.0,
        "min_repeats": 192,
        "cv_target_pct": 10.0,
        "max_samples": 25,
    },
}


@dataclass(frozen=True)
class Case:
    section: str
    name: str
    expr: str
    arg_names: tuple[str, ...]
    constants: dict[str, object] | None = None


@dataclass(frozen=True)
class Row:
    section: str
    name: str
    expr: str
    n: int
    compile_mean_ms: float
    compile_p95_ms: float
    compile_min_ms: float
    compile_max_ms: float
    compile_samples: int
    mean_ms: float
    robust_mean_ms: float
    stdev_ms: float
    cv_pct: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    samples: int


def _block(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple):
        for item in value:
            _block(item)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    alpha = pos - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _trimmed_mean(values: list[float], *, trim: float = 0.2) -> float:
    if len(values) < 3:
        return _mean(values)
    ordered = sorted(values)
    cut = int(len(ordered) * trim)
    if cut <= 0 or (2 * cut) >= len(ordered):
        return _mean(values)
    core = ordered[cut : len(ordered) - cut]
    return _mean(core)


def _repeats_baseline(n: int) -> int:
    if n <= 10:
        return 120
    if n <= 1000:
        return 40
    return 10


def _calibrate_repeats(
    fn,
    args: tuple[object, ...],
    *,
    baseline_repeats: int,
    target_sample_ms: float,
    min_repeats: int,
    max_repeats: int = 200_000,
) -> int:
    trial_repeats = max(4, min_repeats // 4)
    start_ns = time.perf_counter_ns()
    for _ in range(trial_repeats):
        _block(fn(*args))
    elapsed_ns = time.perf_counter_ns() - start_ns
    per_call_ns = max(elapsed_ns / trial_repeats, 1_000.0)
    target_ns = max(target_sample_ms, 1.0) * 1e6
    dynamic_repeats = int(math.ceil(target_ns / per_call_ns))
    return int(max(baseline_repeats, min_repeats, min(dynamic_repeats, max_repeats)))


def _build_args(n: int) -> dict[str, object]:
    x = jnp.arange(1, n + 1, dtype=jnp.float32)
    y = x + 1.0
    mside = max(2, int(round(n**0.5)))
    total = mside * mside
    mat = jnp.reshape(jnp.arange(total, dtype=jnp.float32), (mside, mside))
    return {
        "x": x,
        "y": y,
        "xf": x / 3.0 - 0.5,
        "xs": x - n / 2.0,
        "shape2": jnp.asarray([mside, mside], dtype=jnp.int32),
        "flat": jnp.arange(total, dtype=jnp.float32),
        "take": jnp.asarray(max(1, n // 2), dtype=jnp.int32),
        "drop": jnp.asarray(max(1, n // 3), dtype=jnp.int32),
        "rot": jnp.asarray(max(1, n // 7), dtype=jnp.int32),
        "mat": mat,
        "x0": jnp.asarray(3.0, dtype=jnp.float32),
    }


def _cases() -> list[Case]:
    monadic = [
        Case("monadic", "plus", "+ x", ("x",)),
        Case("monadic", "minus", "- x", ("x",)),
        Case("monadic", "sign", "× xs", ("xs",)),
        Case("monadic", "reciprocal", "÷ x", ("x",)),
        Case("monadic", "floor", "⌊ xf", ("xf",)),
        Case("monadic", "ceil", "⌈ xf", ("xf",)),
        Case("monadic", "abs", "| xs", ("xs",)),
        Case("monadic", "sqrt", "√ x", ("x",)),
        Case("monadic", "reverse", "⌽ x", ("x",)),
        Case("monadic", "transpose", "⍉ mat", ("mat",)),
        Case("monadic", "shape", "≢ mat", ("mat",)),
        Case("monadic", "ravel", "⥊ mat", ("mat",)),
    ]
    dyadic = [
        Case("dyadic", "plus", "x + y", ("x", "y")),
        Case("dyadic", "minus", "x - y", ("x", "y")),
        Case("dyadic", "times", "x × y", ("x", "y")),
        Case("dyadic", "divide", "x ÷ y", ("x", "y")),
        Case("dyadic", "power", "x ⋆ 2", ("x",)),
        Case("dyadic", "lt", "x < y", ("x", "y")),
        Case("dyadic", "le", "x ≤ y", ("x", "y")),
        Case("dyadic", "gt", "x > y", ("x", "y")),
        Case("dyadic", "ge", "x ≥ y", ("x", "y")),
        Case("dyadic", "eq", "x = y", ("x", "y")),
        Case("dyadic", "join", "x ∾ y", ("x", "y")),
    ]
    modifiers = [
        Case("modifier", "fold_plus", "+´ x", ("x",)),
        Case("modifier", "constant", "3˙ x0", ("x0",)),
        Case("modifier", "self_swap", "2 F˜ 5", (), constants={"F": "+"}),
        Case("modifier", "atop", "F∘G x0", ("x0",), constants={"F": "+", "G": "-"}),
        Case("modifier", "over", "2 F○G 3", (), constants={"F": "+", "G": "-"}),
        Case("modifier", "before", "2 F⊸G 3", (), constants={"F": "+", "G": "-"}),
        Case("modifier", "after", "2 F⟜G 3", (), constants={"F": "+", "G": "-"}),
        Case("modifier", "valences", "2 F⊘G 3", (), constants={"F": "+", "G": "-"}),
    ]
    return monadic + dyadic + modifiers


def _run_case(
    case: Case,
    n: int,
    *,
    shape_specialized: bool,
    compile_batches: int,
    samples: int,
    warmup: int,
    target_sample_ms: float,
    min_repeats: int,
    cv_target_pct: float,
    max_samples: int,
) -> Row:
    payload = _build_args(n)
    args = tuple(payload[name] for name in case.arg_names)
    compile_rows: list[float] = []
    fn = None
    batches = max(1, compile_batches if shape_specialized else 1)
    for _ in range(batches):
        start_ns = time.perf_counter_ns()
        compiled = compile_expression(
            case.expr,
            arg_names=case.arg_names,
            constants=case.constants,
            shape_policy=ShapePolicy(kind="static"),
            use_cache=False,
        )
        fn = compiled.jit()
        _block(fn(*args))
        elapsed_ns = time.perf_counter_ns() - start_ns
        compile_rows.append(elapsed_ns / 1e6)

    if fn is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to build benchmark callable")

    for _ in range(max(0, warmup)):
        _block(fn(*args))

    baseline_repeats = _repeats_baseline(n)
    reps = _calibrate_repeats(
        fn,
        args,
        baseline_repeats=baseline_repeats,
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
    )

    rows: list[float] = []

    def _sample_once() -> None:
        start_ns = time.perf_counter_ns()
        for _ in range(reps):
            _block(fn(*args))
        elapsed_ns = time.perf_counter_ns() - start_ns
        rows.append((elapsed_ns / reps) / 1e6)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            _sample_once()
        while len(rows) < max_samples:
            mean_ms = _mean(rows)
            if mean_ms <= 0:
                break
            stdev_ms = _stddev(rows)
            cv_pct = (stdev_ms / mean_ms) * 100.0 if stdev_ms > 0 else 0.0
            if cv_pct <= cv_target_pct:
                break
            _sample_once()
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_ms = _mean(rows)
    stdev_ms = _stddev(rows)
    cv_pct = (stdev_ms / mean_ms) * 100.0 if mean_ms > 0 else 0.0
    return Row(
        section=case.section,
        name=case.name,
        expr=case.expr,
        n=n,
        compile_mean_ms=_mean(compile_rows),
        compile_p95_ms=_percentile(compile_rows, 0.95),
        compile_min_ms=min(compile_rows),
        compile_max_ms=max(compile_rows),
        compile_samples=len(compile_rows),
        mean_ms=mean_ms,
        robust_mean_ms=_trimmed_mean(rows),
        stdev_ms=stdev_ms,
        cv_pct=cv_pct,
        p50_ms=_percentile(rows, 0.50),
        p95_ms=_percentile(rows, 0.95),
        min_ms=min(rows),
        max_ms=max(rows),
        repeats=reps,
        samples=samples,
    )


def _print_noise_summary(rows: list[Row]) -> None:
    print()
    print("noise summary")
    print("n       median_cv(%)   p90_cv(%)   noisy_cases(cv>20%)")
    print("------  ------------   ---------   -------------------")
    for n in sorted({row.n for row in rows}):
        bucket = [row for row in rows if row.n == n]
        cv_values = sorted(row.cv_pct for row in bucket)
        median_cv = _percentile(cv_values, 0.50)
        p90_cv = _percentile(cv_values, 0.90)
        noisy = sum(1 for row in bucket if row.cv_pct > 20.0)
        print(f"{n:6d}  {median_cv:12.2f}   {p90_cv:9.2f}   {noisy:19d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIG),
        default="full",
        help="benchmark profile preset",
    )
    parser.add_argument("--ns", default="10,1000,10000", help="comma-separated sizes")
    parser.add_argument("--samples", type=int, default=None, help="timing samples")
    parser.add_argument("--warmup", type=int, default=None, help="warmup rounds before timing")
    parser.add_argument(
        "--target-sample-ms",
        type=float,
        default=None,
        help="target wall time per timing sample; repeats auto-calibrate to reach this",
    )
    parser.add_argument("--min-repeats", type=int, default=None, help="lower bound for calibrated repeats")
    parser.add_argument(
        "--cv-target",
        type=float,
        default=None,
        help="adaptive sampling target: add samples until cv <= target or max-samples is reached",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="upper bound for adaptive sampling",
    )
    parser.add_argument(
        "--shape-specialized",
        action="store_true",
        help=(
            "isolate compile-vs-steady metrics by running fixed-shape compile batches "
            "per case before steady-state timing"
        ),
    )
    parser.add_argument(
        "--compile-batches",
        type=int,
        default=5,
        help="number of fixed-shape compile batches when --shape-specialized is enabled",
    )
    parser.add_argument("--json-out", default="", help="optional path for machine-readable output")
    args = parser.parse_args()
    affinity_info = configure_cpu_affinity_from_env()

    profile = PROFILE_CONFIG[args.profile]
    samples = int(profile["samples"] if args.samples is None else args.samples)
    warmup = int(profile["warmup"] if args.warmup is None else args.warmup)
    target_sample_ms = float(profile["target_sample_ms"] if args.target_sample_ms is None else args.target_sample_ms)
    min_repeats = int(profile["min_repeats"] if args.min_repeats is None else args.min_repeats)
    cv_target_pct = float(profile["cv_target_pct"] if args.cv_target is None else args.cv_target)
    max_samples = int(profile["max_samples"] if args.max_samples is None else args.max_samples)
    if max_samples < samples:
        max_samples = samples

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    rows: list[Row] = []
    all_cases = _cases()

    print(f"sizes: {ns}")
    print(f"cases: monadic={sum(c.section=='monadic' for c in all_cases)}, dyadic={sum(c.section=='dyadic' for c in all_cases)}, modifier={sum(c.section=='modifier' for c in all_cases)}")
    print(
        "config:"
        f" profile={args.profile}, samples={samples}, warmup={warmup},"
        f" target_sample_ms={target_sample_ms:.1f}, min_repeats={min_repeats},"
        f" cv_target={cv_target_pct:.1f}%, max_samples={max_samples},"
        f" shape_specialized={bool(args.shape_specialized)},"
        f" compile_batches={int(args.compile_batches)}"
    )
    print(
        "host:"
        f" backend={jax.default_backend()}, affinity={affinity_info.get('active')},"
        f" thread_env={','.join(sorted(k for k in os.environ if k.endswith('_NUM_THREADS') or k in {'JAX_NUM_THREADS', 'XLA_FLAGS'})) or 'default'}"
    )
    print()
    for n in ns:
        print(f"n={n}")
        for case in all_cases:
            rows.append(
                _run_case(
                    case,
                    n,
                    shape_specialized=bool(args.shape_specialized),
                    compile_batches=max(1, int(args.compile_batches)),
                    samples=samples,
                    warmup=warmup,
                    target_sample_ms=target_sample_ms,
                    min_repeats=min_repeats,
                    cv_target_pct=cv_target_pct,
                    max_samples=max_samples,
                )
            )
        print(f"completed n={n}")

    print()
    print("section    case                  n      compile(ms)  mean(ms)   p95(ms)   cv(%)  reps")
    print("--------   ------------------  ------  -----------  ---------  --------  ------  -----")
    for row in rows:
        print(
            f"{row.section:8} {row.name:18} {row.n:6d}  {row.compile_mean_ms:11.4f}  "
            f"{row.mean_ms:9.4f}  {row.p95_ms:8.4f}  {row.cv_pct:6.2f}  {row.repeats:5d}"
        )
    _print_noise_summary(rows)

    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "profile": args.profile,
            "sizes": ns,
            "samples": samples,
            "warmup": warmup,
            "target_sample_ms": target_sample_ms,
            "min_repeats": min_repeats,
            "cv_target_pct": cv_target_pct,
            "max_samples": max_samples,
            "shape_specialized": bool(args.shape_specialized),
            "compile_batches": max(1, int(args.compile_batches)),
            "affinity": affinity_info,
            "host": host_metadata(),
            "rows": [asdict(row) for row in rows],
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {outpath}")


if __name__ == "__main__":
    main()
