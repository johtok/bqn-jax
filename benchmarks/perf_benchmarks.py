"""Performance benchmarks inspired by https://mlochbaum.github.io/BQN/implementation/perf.html."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_cache_stats, compile_expression
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


DEFAULT_FLAT_N = 1_000_000
DEFAULT_ROWS = 512
DEFAULT_COLS = 512
PROFILE_PRESETS: dict[str, dict[str, float | int]] = {
    "quick": {"samples": 3, "warmup": 1, "repeat_scale": 0.25, "target_sample_ms": 12.0, "min_repeats": 8, "cv_target_pct": 25.0, "max_samples": 7},
    "full": {"samples": 7, "warmup": 3, "repeat_scale": 1.0, "target_sample_ms": 30.0, "min_repeats": 16, "cv_target_pct": 18.0, "max_samples": 11},
}


@dataclass(frozen=True)
class JaxCase:
    name: str
    expression: str
    arg_names: tuple[str, ...]
    args: tuple[object, ...]
    repeats: int
    note: str


@dataclass(frozen=True)
class CbqnCase:
    name: str
    code: str
    note: str


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
class JaxRow:
    name: str
    note: str
    compile_ms: float
    repeats: int
    warmup: int
    samples: int
    timing: TimingStats


@dataclass(frozen=True)
class CbqnRow:
    name: str
    note: str
    repeats: int
    samples: int
    timing: TimingStats


def _scaled_repeats(base: int, scale: float) -> int:
    return max(1, int(round(base * scale)))


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


def _parse_cbqn_number(raw: str) -> float:
    line = raw.strip().splitlines()[-1].strip()
    return float(line.replace("¯", "-"))


def _find_cbqn_binary() -> str | None:
    configured = os.environ.get("CBQN_BIN") or os.environ.get("BQN_BIN")
    if configured:
        if os.path.sep in configured:
            return configured if os.path.exists(configured) else None
        resolved = shutil.which(configured)
        if resolved:
            return resolved
    for candidate in ("bqn", "cbqn"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _run_cbqn_timed(cbqn_bin: str, code: str) -> float:
    proc = subprocess.run(
        [cbqn_bin, "-p", code],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(stderr or f"CBQN exited with status {proc.returncode}")
    if not proc.stdout.strip():
        raise RuntimeError("CBQN produced no stdout for timed expression")
    return _parse_cbqn_number(proc.stdout)


def _build_jax_cases(flat_n: int, rows: int, cols: int, repeat_scale: float) -> list[JaxCase]:
    scalar_x = jnp.asarray(3.0, dtype=jnp.float32)
    scalar_y = jnp.asarray(1.0, dtype=jnp.float32)

    flat_x = jnp.arange(flat_n, dtype=jnp.float32)
    flat_y = flat_x + 1.0

    matrix_x = jnp.reshape(jnp.arange(rows * cols, dtype=jnp.float32), (rows, cols))
    matrix_y = matrix_x + 1.0

    return [
        JaxCase(
            name="scalar_chain",
            expression="(x × x) + y",
            arg_names=("x", "y"),
            args=(scalar_x, scalar_y),
            repeats=_scaled_repeats(20_000, repeat_scale),
            note="scalar arithmetic (atom-sized values)",
        ),
        JaxCase(
            name="flat_elementwise",
            expression="(x × x) + y",
            arg_names=("x", "y"),
            args=(flat_x, flat_y),
            repeats=_scaled_repeats(20, repeat_scale),
            note="flat list elementwise arithmetic",
        ),
        JaxCase(
            name="flat_reduce_plus",
            expression="0 +´ x",
            arg_names=("x",),
            args=(flat_x,),
            repeats=_scaled_repeats(16, repeat_scale),
            note="flat list reduction (0 +´)",
        ),
        JaxCase(
            name="matrix_elementwise",
            expression="(x × x) + y",
            arg_names=("x", "y"),
            args=(matrix_x, matrix_y),
            repeats=_scaled_repeats(20, repeat_scale),
            note="multidimensional elementwise arithmetic",
        ),
        JaxCase(
            name="matrix_reduce_major",
            expression="0 +´ x",
            arg_names=("x",),
            args=(matrix_x,),
            repeats=_scaled_repeats(16, repeat_scale),
            note="multidimensional reduction over major axis",
        ),
    ]


def _build_cbqn_cases(flat_n: int, rows: int, cols: int, repeat_scale: float) -> list[CbqnCase]:
    matrix_count = rows * cols
    return [
        CbqnCase(
            name="scalar_chain",
            code=f"{_scaled_repeats(120000, repeat_scale)} 1⊸{{𝕩×𝕩+𝕨}}•_timed 3",
            note="scalar arithmetic (atom-sized values)",
        ),
        CbqnCase(
            name="flat_elementwise",
            code=f"{_scaled_repeats(30, repeat_scale)} (1+↕{flat_n})⊸{{𝕩×𝕩+𝕨}}•_timed ↕{flat_n}",
            note="flat list elementwise arithmetic",
        ),
        CbqnCase(
            name="flat_reduce_plus",
            code=f"{_scaled_repeats(30, repeat_scale)} 0⊸+´•_timed ↕{flat_n}",
            note="flat list reduction (0 +´)",
        ),
        CbqnCase(
            name="matrix_elementwise",
            code=(
                f"{_scaled_repeats(20, repeat_scale)} ({rows}‿{cols}⥊(1+↕{matrix_count}))⊸{{𝕩×𝕩+𝕨}}•_timed "
                f"{rows}‿{cols}⥊↕{matrix_count}"
            ),
            note="multidimensional elementwise arithmetic",
        ),
        CbqnCase(
            name="matrix_reduce_major",
            code=f"{_scaled_repeats(20, repeat_scale)} {{+´˘⍉𝕩}}•_timed {rows}‿{cols}⥊↕{matrix_count}",
            note="multidimensional reduction over major axis",
        ),
    ]


def _print_section(title: str) -> None:
    print(title)
    print("-" * len(title))


def run_jax_benchmarks(
    flat_n: int,
    rows: int,
    cols: int,
    repeat_scale: float,
    *,
    warmup: int,
    samples: int,
    target_sample_ms: float,
    min_repeats: int,
    cv_target_pct: float,
    max_samples: int,
) -> list[JaxRow]:
    _print_section("JAX-backed bqn-jax")
    out: list[JaxRow] = []
    for case in _build_jax_cases(flat_n, rows, cols, repeat_scale):
        t0 = time.perf_counter()
        compiled = compile_expression(
            case.expression,
            arg_names=case.arg_names,
            shape_policy=ShapePolicy(kind="static"),
            use_cache=False,
        )
        jitted = compiled.jit()
        block_until_ready(jitted(*case.args))
        compile_ms = (time.perf_counter() - t0) * 1e3

        repeats = calibrate_repeats(
            jitted,
            case.args,
            baseline_repeats=case.repeats,
            target_sample_ms=target_sample_ms,
            min_repeats=min_repeats,
        )
        sample_rows = sample_adaptive_ms(
            jitted,
            case.args,
            repeats=repeats,
            warmup=warmup,
            samples=samples,
            cv_target_pct=cv_target_pct,
            max_samples=max_samples,
        )
        stats = _summarize_ms(sample_rows)
        row = JaxRow(
            name=case.name,
            note=case.note,
            compile_ms=compile_ms,
            repeats=repeats,
            warmup=warmup,
            samples=len(sample_rows),
            timing=stats,
        )
        out.append(row)
        print(
            f"{row.name:20} compile {row.compile_ms:8.3f} ms  "
            f"steady {row.timing.mean_ms:8.3f} ms (p95 {row.timing.p95_ms:8.3f}, cv {row.timing.cv_pct:5.2f}%)  {row.note}"
        )
    print()
    return out


def run_cbqn_benchmarks(
    cbqn_bin: str,
    flat_n: int,
    rows: int,
    cols: int,
    repeat_scale: float,
    *,
    samples: int,
) -> list[CbqnRow]:
    _print_section(f"CBQN ({cbqn_bin})")
    out: list[CbqnRow] = []
    for case in _build_cbqn_cases(flat_n, rows, cols, repeat_scale):
        sample_rows = [_run_cbqn_timed(cbqn_bin, case.code) for _ in range(samples)]
        stats = _summarize_ms(sample_rows)
        row = CbqnRow(
            name=case.name,
            note=case.note,
            repeats=1,
            samples=samples,
            timing=stats,
        )
        out.append(row)
        print(f"{row.name:20} steady {row.timing.mean_ms:8.3f} ms (p95 {row.timing.p95_ms:8.3f})  {row.note}")
    print()
    return out


def _relative_summary(jax_rows: list[JaxRow], cbqn_rows: list[CbqnRow]) -> list[tuple[str, float]]:
    cbqn_map = {row.name: row.timing.mean_ms for row in cbqn_rows}
    result: list[tuple[str, float]] = []
    for row in jax_rows:
        cbqn_ms = cbqn_map.get(row.name)
        if cbqn_ms is None:
            continue
        ratio = cbqn_ms / row.timing.mean_ms if row.timing.mean_ms > 0 else float("inf")
        result.append((row.name, ratio))
    return result


def _print_summary(jax_rows: list[JaxRow], cbqn_rows: list[CbqnRow]) -> None:
    if not cbqn_rows:
        return
    _print_section("Relative speed (CBQN/JAX steady-state)")
    for name, ratio in _relative_summary(jax_rows, cbqn_rows):
        print(f"{name:20} {ratio:9.3f}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run performance benchmarks inspired by "
            "https://mlochbaum.github.io/BQN/implementation/perf.html"
        )
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_PRESETS), default="quick", help="fixed benchmark profile")
    parser.add_argument("--samples", type=int, default=None, help="override timing sample count")
    parser.add_argument("--warmup", type=int, default=None, help="override warmup rounds")
    parser.add_argument("--target-sample-ms", type=float, default=None, help="target wall time per sample")
    parser.add_argument("--min-repeats", type=int, default=None, help="minimum repeats after calibration")
    parser.add_argument("--cv-target", type=float, default=None, help="adaptive sampling CV target percent")
    parser.add_argument("--max-samples", type=int, default=None, help="adaptive sampling cap")
    parser.add_argument(
        "--repeat-scale",
        type=float,
        default=None,
        help="override repeat scaling factor",
    )
    parser.add_argument(
        "--no-cbqn",
        action="store_true",
        help="Skip CBQN runs even if a CBQN binary is available.",
    )
    parser.add_argument("--flat-n", type=int, default=DEFAULT_FLAT_N, help="flat-array length for flat cases")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="row count for matrix cases")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="column count for matrix cases")
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path to write machine-readable benchmark results",
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

    print("BQN perf-inspired benchmark suite")
    print("Based on categories from https://mlochbaum.github.io/BQN/implementation/perf.html")
    print(
        f"config: profile={args.profile}, samples={samples}, warmup={warmup}, "
        f"flat_n={args.flat_n}, rows={args.rows}, cols={args.cols}, repeat_scale={repeat_scale:.3f}, "
        f"target_sample_ms={target_sample_ms:.1f}, min_repeats={min_repeats}, "
        f"cv_target={cv_target_pct:.1f}%, max_samples={max_samples}"
    )
    print(f"host: backend={jax.default_backend()}, affinity={affinity_info.get('active')}")
    print()

    jax_rows = run_jax_benchmarks(
        args.flat_n,
        args.rows,
        args.cols,
        repeat_scale,
        warmup=warmup,
        samples=samples,
        target_sample_ms=target_sample_ms,
        min_repeats=min_repeats,
        cv_target_pct=cv_target_pct,
        max_samples=max_samples,
    )

    cbqn_rows: list[CbqnRow] = []
    cbqn_bin = _find_cbqn_binary()
    if args.no_cbqn:
        print("CBQN section skipped by --no-cbqn.")
    elif cbqn_bin is None:
        print("CBQN section skipped: no `bqn`/`cbqn` binary found.")
    else:
        cbqn_rows = run_cbqn_benchmarks(
            cbqn_bin,
            args.flat_n,
            args.rows,
            args.cols,
            repeat_scale,
            samples=samples,
        )

    if cbqn_rows:
        _print_summary(jax_rows, cbqn_rows)

    if args.json_out:
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "config": {
                "profile": args.profile,
                "samples": samples,
                "warmup": warmup,
                "target_sample_ms": target_sample_ms,
                "min_repeats": min_repeats,
                "cv_target_pct": cv_target_pct,
                "max_samples": max_samples,
                "flat_n": args.flat_n,
                "rows": args.rows,
                "cols": args.cols,
                "repeat_scale": repeat_scale,
                "cbqn_bin": cbqn_bin if cbqn_rows else None,
            },
            "affinity": affinity_info,
            "host": host_metadata(),
            "compile_cache": compile_cache_stats(),
            "jax_rows": [asdict(row) for row in jax_rows],
            "cbqn_rows": [asdict(row) for row in cbqn_rows],
            "relative_cbqn_over_jax": dict(_relative_summary(jax_rows, cbqn_rows)),
        }
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON benchmark output: {outpath}")


if __name__ == "__main__":
    main()
