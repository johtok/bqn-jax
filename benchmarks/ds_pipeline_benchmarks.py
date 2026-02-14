"""Data-science pipeline benchmarks: inference, training, and autodiff scaling."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression, evaluate


@dataclass(frozen=True)
class TimingRow:
    section: str
    size: int
    engine: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    samples: int
    note: str


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


def _time_samples(fn, *args, repeats: int, samples: int, warmup: int = 1) -> tuple[float, float, float, float, float]:
    for _ in range(warmup):
        _block(fn(*args))
    rows: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repeats):
            _block(fn(*args))
        end = time.perf_counter()
        rows.append((end - start) * 1e3 / repeats)
    return (
        sum(rows) / len(rows),
        _percentile(rows, 0.50),
        _percentile(rows, 0.95),
        min(rows),
        max(rows),
    )


def _row(
    section: str,
    size: int,
    engine: str,
    stats: tuple[float, float, float, float, float],
    *,
    repeats: int,
    samples: int,
    note: str,
) -> TimingRow:
    return TimingRow(
        section=section,
        size=size,
        engine=engine,
        mean_ms=stats[0],
        p50_ms=stats[1],
        p95_ms=stats[2],
        min_ms=stats[3],
        max_ms=stats[4],
        repeats=repeats,
        samples=samples,
        note=note,
    )


def _bench_interpreter_vs_compiled(size: int, *, samples: int) -> list[TimingRow]:
    repeats = max(1, 30_000 // size)
    x = jnp.linspace(-1.0, 1.0, size, dtype=jnp.float32)
    w = jnp.asarray(1.5, dtype=jnp.float32)
    b = jnp.asarray(0.1, dtype=jnp.float32)
    env = {"x": x, "w": w, "b": b}

    source = "x × w + b"
    compiled = compile_expression(source, arg_names=("x", "w", "b"), shape_policy=ShapePolicy(kind="static"), use_cache=False)
    jitted = compiled.jit()

    interp_stats = _time_samples(lambda: evaluate(source, env=env), repeats=repeats, samples=samples)
    compiled_stats = _time_samples(lambda: jitted(x, w, b), repeats=repeats, samples=samples)

    return [
        _row(
            "inference_compare",
            size,
            "evaluate",
            interp_stats,
            repeats=repeats,
            samples=samples,
            note="interpreter evaluate(source, env)",
        ),
        _row(
            "inference_compare",
            size,
            "compiled_jit",
            compiled_stats,
            repeats=repeats,
            samples=samples,
            note="compile_expression(...).jit()",
        ),
    ]


def _bench_training_step(size: int, *, samples: int) -> TimingRow:
    repeats = max(1, 10_000 // size)
    x = jnp.linspace(-2.0, 2.0, size, dtype=jnp.float32)
    y = 0.5 * x * x - 0.3 * x + 0.2

    model = compile_expression("x × w + b", arg_names=("x", "w", "b"), use_cache=False)
    predict = jax.vmap(model, in_axes=(0, None, None), out_axes=0)

    def loss(params):
        w, b = params
        pred = predict(x, w, b)
        residual = pred - y
        return jnp.mean(residual * residual)

    value_grad = jax.value_and_grad(loss)
    step_fn = jax.jit(value_grad)
    params = (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32))
    stats = _time_samples(step_fn, params, repeats=repeats, samples=samples)
    return _row(
        "training_step",
        size,
        "jit(value_and_grad)",
        stats,
        repeats=repeats,
        samples=samples,
        note="one-end-to-end optimizer step kernel",
    )


def _bench_autodiff(size: int, *, samples: int) -> TimingRow:
    repeats = max(1, 20_000 // size)
    x = jnp.linspace(-3.0, 3.0, size, dtype=jnp.float32)
    b = jnp.full_like(x, 0.75)
    model = compile_expression("x × x + b × x", arg_names=("x", "b"), use_cache=False)
    grad_fn = model.grad(argnums=0)
    grad_vmap = jax.vmap(grad_fn, in_axes=(0, 0), out_axes=0)
    stats = _time_samples(grad_vmap, x, b, repeats=repeats, samples=samples)

    flops_per_elem = 3.0
    bytes_per_elem = 12.0  # x + b + output, float32
    intensity = flops_per_elem / bytes_per_elem
    note = f"autodiff workload; arithmetic_intensity={intensity:.3f} flops/byte => likely memory-bound"

    return _row(
        "autodiff",
        size,
        "vmap(grad)",
        stats,
        repeats=repeats,
        samples=samples,
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="512,2048,8192", help="comma-separated dataset sizes")
    parser.add_argument("--samples", type=int, default=3, help="timing samples per workload")
    parser.add_argument("--json-out", default="", help="optional output path")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    rows: list[TimingRow] = []

    print("Data-science pipeline benchmarks")
    print(f"sizes={sizes}, samples={args.samples}")
    print()

    for size in sizes:
        rows.extend(_bench_interpreter_vs_compiled(size, samples=args.samples))
        rows.append(_bench_training_step(size, samples=args.samples))
        rows.append(_bench_autodiff(size, samples=args.samples))

    print("section             size   engine              mean(ms)   p95(ms)")
    print("-----------------  -----  ------------------  ---------  --------")
    for row in rows:
        print(f"{row.section:17} {row.size:5d}  {row.engine:18}  {row.mean_ms:9.3f}  {row.p95_ms:8.3f}")
    print()

    inf_by_size: dict[int, dict[str, TimingRow]] = {}
    for row in rows:
        if row.section != "inference_compare":
            continue
        inf_by_size.setdefault(row.size, {})[row.engine] = row
    print("compiled_jit / evaluate inference speedup")
    for size in sizes:
        table = inf_by_size.get(size, {})
        eva = table.get("evaluate")
        jit = table.get("compiled_jit")
        if eva is None or jit is None or jit.mean_ms == 0:
            continue
        speedup = eva.mean_ms / jit.mean_ms
        print(f"n={size:5d}: {speedup:8.3f}x")

    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "sizes": sizes,
            "samples": args.samples,
            "rows": [asdict(row) for row in rows],
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {outpath}")


if __name__ == "__main__":
    main()
