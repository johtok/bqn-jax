"""Benchmark slice: typical numeric/data-science array workloads."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression
from _bench_utils import (
    block_until_ready,
    calibrate_repeats,
    configure_cpu_affinity_from_env,
    host_metadata,
    mean as _mean,
    percentile as _percentile,
    sample_adaptive_ms,
    stddev as _stddev,
)


PROFILE_PRESETS: dict[str, dict[str, float | int]] = {
    "quick": {"samples": 3, "warmup": 1, "repeat_scale": 0.25, "target_sample_ms": 12.0, "min_repeats": 8, "cv_target_pct": 25.0, "max_samples": 7},
    "full": {"samples": 7, "warmup": 3, "repeat_scale": 1.0, "target_sample_ms": 30.0, "min_repeats": 16, "cv_target_pct": 18.0, "max_samples": 11},
}


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    robust_mean_ms: float
    stdev_ms: float
    cv_pct: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class WorkloadRow:
    workload: str
    shape_policy: str
    compile_ms: float
    repeats: int
    warmup: int
    samples: int
    timing: TimingStats
    note: str


def _trimmed_mean(values: list[float], *, trim: float = 0.2) -> float:
    if len(values) < 3:
        return _mean(values)
    ordered = sorted(values)
    cut = int(len(ordered) * trim)
    if cut <= 0 or (2 * cut) >= len(ordered):
        return _mean(values)
    core = ordered[cut : len(ordered) - cut]
    return _mean(core)


def _summarize_ms(ms: list[float]) -> TimingStats:
    avg = _mean(ms)
    sd = _stddev(ms)
    cv_pct = (sd / avg) * 100.0 if avg > 0 else 0.0
    return TimingStats(
        mean_ms=avg,
        robust_mean_ms=_trimmed_mean(ms),
        stdev_ms=sd,
        cv_pct=cv_pct,
        p50_ms=_percentile(ms, 0.50),
        p95_ms=_percentile(ms, 0.95),
        min_ms=min(ms),
        max_ms=max(ms),
    )


def _scaled_repeats(base: int, scale: float) -> int:
    return max(1, int(round(base * scale)))


def _run_for_policy(
    kind: str,
    *,
    repeats_scale: float,
    warmup: int,
    samples: int,
    target_sample_ms: float,
    min_repeats: int,
    cv_target_pct: float,
    max_samples: int,
) -> list[WorkloadRow]:
    policy = ShapePolicy(kind=kind)
    out: list[WorkloadRow] = []

    x = jnp.linspace(-5.0, 5.0, 1_000_000, dtype=jnp.float32)
    y = jnp.linspace(0.0, 3.0, 1_000_000, dtype=jnp.float32)
    z = jnp.linspace(1.0, 4.0, 1_000_000, dtype=jnp.float32)

    # Workload 1: elementwise feature transform.
    t0 = time.perf_counter()
    elem = compile_expression(
        "((x × x) + y) ÷ (z + 1)",
        arg_names=("x", "y", "z"),
        shape_policy=policy,
        use_cache=False,
    )
    elem_jit = elem.jit()
    block_until_ready(elem_jit(x, y, z))
    compile_ms = (time.perf_counter() - t0) * 1e3
    elem_repeats = calibrate_repeats(
        elem_jit,
        (x, y, z),
        baseline_repeats=_scaled_repeats(30, repeats_scale),
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
    )
    elem_samples = sample_adaptive_ms(
        elem_jit,
        (x, y, z),
        repeats=elem_repeats,
        warmup=warmup,
        samples=samples,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )
    stats = _summarize_ms(elem_samples)
    out.append(
        WorkloadRow(
            workload="elementwise_transform",
            shape_policy=kind,
            compile_ms=compile_ms,
            repeats=elem_repeats,
            warmup=warmup,
            samples=len(elem_samples),
            timing=stats,
            note="large flat elementwise feature transform",
        )
    )

    # Workload 2: reduction.
    mat = jnp.reshape(jnp.arange(4096 * 512, dtype=jnp.float32), (4096, 512))
    t0 = time.perf_counter()
    red = compile_expression("0 +´ x", arg_names=("x",), shape_policy=policy, use_cache=False)
    red_jit = red.jit()
    block_until_ready(red_jit(mat))
    compile_ms = (time.perf_counter() - t0) * 1e3
    red_repeats = calibrate_repeats(
        red_jit,
        (mat,),
        baseline_repeats=_scaled_repeats(40, repeats_scale),
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
    )
    red_samples = sample_adaptive_ms(
        red_jit,
        (mat,),
        repeats=red_repeats,
        warmup=warmup,
        samples=samples,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )
    stats = _summarize_ms(red_samples)
    out.append(
        WorkloadRow(
            workload="batch_reduction_plus_insert",
            shape_policy=kind,
            compile_ms=compile_ms,
            repeats=red_repeats,
            warmup=warmup,
            samples=len(red_samples),
            timing=stats,
            note="reduction over batch-major matrix",
        )
    )

    # Workload 3: vmapped gradient of a scalar model.
    gx = jnp.linspace(-3.0, 3.0, 1_000_000, dtype=jnp.float32)
    gb = jnp.full_like(gx, 0.75)
    t0 = time.perf_counter()
    model = compile_expression("x × x + b × x", arg_names=("x", "b"), shape_policy=policy, use_cache=False)
    grad_fn = model.grad(argnums=0)
    grad_vmap = jax.vmap(grad_fn, in_axes=(0, 0), out_axes=0)
    block_until_ready(grad_vmap(gx, gb))
    compile_ms = (time.perf_counter() - t0) * 1e3
    grad_repeats = calibrate_repeats(
        grad_vmap,
        (gx, gb),
        baseline_repeats=_scaled_repeats(20, repeats_scale),
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
    )
    grad_samples = sample_adaptive_ms(
        grad_vmap,
        (gx, gb),
        repeats=grad_repeats,
        warmup=warmup,
        samples=samples,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )
    stats = _summarize_ms(grad_samples)
    out.append(
        WorkloadRow(
            workload="vmapped_grad",
            shape_policy=kind,
            compile_ms=compile_ms,
            repeats=grad_repeats,
            warmup=warmup,
            samples=len(grad_samples),
            timing=stats,
            note="per-element autodiff, usually memory-bandwidth sensitive",
        )
    )

    return out


def run_benchmarks(
    *,
    profile: str,
    repeat_scale: float,
    warmup: int,
    samples: int,
    target_sample_ms: float,
    min_repeats: int,
    cv_target_pct: float,
    max_samples: int,
) -> list[WorkloadRow]:
    print("Benchmark: JAX-backed compiled BQN subset")
    print(
        f"profile={profile} samples={samples} warmup={warmup} repeat_scale={repeat_scale:.3f} "
        f"target_sample_ms={target_sample_ms:.1f} min_repeats={min_repeats} "
        f"cv_target={cv_target_pct:.1f}% max_samples={max_samples}"
    )
    print()

    rows = _run_for_policy(
        "dynamic",
        repeats_scale=repeat_scale,
        warmup=warmup,
        samples=samples,
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )
    rows.extend(
        _run_for_policy(
            "static",
            repeats_scale=repeat_scale,
            warmup=warmup,
            samples=samples,
            target_sample_ms=target_sample_ms,
            min_repeats=min_repeats,
            cv_target_pct=cv_target_pct,
            max_samples=max_samples,
        )
    )

    by_name: dict[str, dict[str, WorkloadRow]] = {}
    for row in rows:
        by_name.setdefault(row.workload, {})[row.shape_policy] = row

    print("workload                      policy   compile(ms)   mean(ms)   p95(ms)")
    print("---------------------------  -------  -----------  ---------  --------")
    for row in rows:
        print(
            f"{row.workload:27}  {row.shape_policy:7}  {row.compile_ms:11.3f}  "
            f"{row.timing.mean_ms:9.3f}  {row.timing.p95_ms:8.3f}"
        )
    print()

    print("static/dynamic steady-state ratio")
    for workload, table in by_name.items():
        dyn = table.get("dynamic")
        sta = table.get("static")
        if dyn is None or sta is None or dyn.timing.mean_ms == 0:
            continue
        ratio = sta.timing.mean_ms / dyn.timing.mean_ms
        print(f"{workload:27}  {ratio:8.3f}x")
    print()

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default="quick", help="fixed benchmark profile")
    parser.add_argument("--samples", type=int, default=None, help="override sample count")
    parser.add_argument("--warmup", type=int, default=None, help="override warmup rounds")
    parser.add_argument("--target-sample-ms", type=float, default=None, help="target wall time per sample")
    parser.add_argument("--min-repeats", type=int, default=None, help="minimum repeats after calibration")
    parser.add_argument("--cv-target", type=float, default=None, help="adaptive sampling CV target percent")
    parser.add_argument("--max-samples", type=int, default=None, help="adaptive sampling cap")
    parser.add_argument("--repeat-scale", type=float, default=None, help="override repeat scale")
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path to write machine-readable workload results",
    )
    args = parser.parse_args()
    affinity_info = configure_cpu_affinity_from_env()

    profile = PROFILE_PRESETS[args.profile]
    samples = int(profile["samples"] if args.samples is None else args.samples)
    warmup = int(profile["warmup"] if args.warmup is None else args.warmup)
    target_sample_ms = float(profile["target_sample_ms"] if args.target_sample_ms is None else args.target_sample_ms)
    min_repeats = int(profile["min_repeats"] if args.min_repeats is None else args.min_repeats)
    cv_target_pct = float(profile["cv_target_pct"] if args.cv_target is None else args.cv_target)
    max_samples = int(profile["max_samples"] if args.max_samples is None else args.max_samples)
    repeat_scale = float(profile["repeat_scale"] if args.repeat_scale is None else args.repeat_scale)
    if max_samples < samples:
        max_samples = samples

    rows = run_benchmarks(
        profile=args.profile,
        repeat_scale=repeat_scale,
        warmup=warmup,
        samples=samples,
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )
    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "profile": args.profile,
            "samples": samples,
            "warmup": warmup,
            "target_sample_ms": target_sample_ms,
            "min_repeats": min_repeats,
            "cv_target_pct": cv_target_pct,
            "max_samples": max_samples,
            "repeat_scale": repeat_scale,
            "affinity": affinity_info,
            "host": host_metadata(),
            "rows": [asdict(row) for row in rows],
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON workload output: {outpath}")
