"""Benchmark monadic, dyadic, and modifier expressions at fixed sizes."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp

from bqn_jax import evaluate
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


N_DEFAULT = (10, 1000, 10000)
PROFILE_CONFIG: dict[str, dict[str, float | int]] = {
    "quick": {"samples": 3, "warmup": 1, "repeat_scale": 0.25, "max_case_seconds": 2.0, "target_sample_ms": 10.0, "min_repeats": 8, "cv_target_pct": 25.0, "max_samples": 7},
    "full": {"samples": 7, "warmup": 2, "repeat_scale": 1.0, "max_case_seconds": 20.0, "target_sample_ms": 20.0, "min_repeats": 12, "cv_target_pct": 18.0, "max_samples": 11},
}


@dataclass(frozen=True)
class BenchCase:
    section: str
    name: str
    expression: str
    env_builder: Callable[[int], dict[str, object]]


@dataclass(frozen=True)
class BenchRow:
    section: str
    name: str
    expression: str
    n: int
    status: str
    first_call_ms: float | None
    mean_ms: float | None
    stdev_ms: float | None
    cv_pct: float | None
    p50_ms: float | None
    p95_ms: float | None
    min_ms: float | None
    max_ms: float | None
    repeats: int
    warmup: int
    samples: int
    error: str | None



def _sizes_from_arg(raw: str) -> tuple[int, ...]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("at least one size must be provided")
    return tuple(out)


def _repeats_for_n(n: int, repeat_scale: float) -> int:
    if n <= 10:
        base = 120
    elif n <= 1000:
        base = 40
    else:
        base = 12
    return max(1, int(round(base * repeat_scale)))


def _base_env(n: int) -> dict[str, object]:
    n = int(n)
    side = max(2, int(round(math.sqrt(float(n)))))
    total = side * side

    x = jnp.arange(1, n + 1, dtype=jnp.float32)
    y = x + 1.0
    xf = x / 3.0 - 0.5
    xs = x - float(n) / 2.0

    xi = jnp.arange(n, dtype=jnp.int32)
    binv = jnp.mod(xi, 2)
    cls = jnp.mod(xi, max(2, min(8, n)))
    cnt = jnp.mod(xi, 3)

    yq = jnp.mod((xi * 5 + 1), max(4, 2 * n)).astype(jnp.int32)
    xq = jnp.mod((xi * 3 + 2), max(4, 2 * n)).astype(jnp.int32)

    idx_len = max(1, n // 4)
    idx = jnp.mod(jnp.arange(idx_len, dtype=jnp.int32) * 7 + 1, n)

    mat = jnp.reshape(jnp.arange(total, dtype=jnp.float32), (side, side))
    shape2 = jnp.asarray([side, side], dtype=jnp.int32)
    xshape = jnp.arange(total, dtype=jnp.float32)

    take = max(1, n // 2)
    drop = max(1, n // 3)
    rot = max(1, n // 7)

    # Boxed vectors/lists for depth/grouping/sort semantics.
    bx = [jnp.asarray(1.0), jnp.asarray([2.0, 3.0], dtype=jnp.float32)]
    bx2 = [jnp.asarray(1.0), jnp.asarray([2.0, 3.0], dtype=jnp.float32)]

    pat = jnp.asarray([1, 2], dtype=jnp.int32)
    keys_up = jnp.asarray([2, 4, 6], dtype=jnp.int32)
    keys_dn = jnp.asarray([6, 4, 2], dtype=jnp.int32)

    x0 = jnp.asarray(3.0, dtype=jnp.float32)

    def _scalar_int(v: object) -> int:
        arr = jnp.asarray(v)
        return int(arr.item()) if arr.ndim == 0 else int(arr.reshape(()).item())

    env: dict[str, object] = {
        "n": jnp.asarray(n, dtype=jnp.int32),
        "x": x,
        "y": y,
        "xf": xf,
        "xs": xs,
        "xq": xq,
        "yq": yq,
        "binv": binv,
        "cls": cls,
        "cnt": cnt,
        "idx": idx,
        "mat": mat,
        "shape2": shape2,
        "xshape": xshape,
        "take": jnp.asarray(take, dtype=jnp.int32),
        "drop": jnp.asarray(drop, dtype=jnp.int32),
        "rot": jnp.asarray(rot, dtype=jnp.int32),
        "fill": jnp.asarray(0.0, dtype=jnp.float32),
        "pick": jnp.asarray(min(2, n - 1), dtype=jnp.int32),
        "axes": jnp.asarray([1, 0], dtype=jnp.int32),
        "bx": bx,
        "bx2": bx2,
        "pat": pat,
        "keys_up": keys_up,
        "keys_dn": keys_dn,
        "x0": x0,
    }

    # Modifier/combinator helpers.
    env.update(
        {
            "F": lambda *a: a[0] + 1 if len(a) == 1 else a[0] + a[1],
            "G": lambda *a: 2 * a[0] if len(a) == 1 else 10 * a[0] + a[1],
            "A": lambda a: a + 10,
            "B": lambda a: a + 100,
            "Sel": lambda a: _scalar_int(a) % 2,
            "M": lambda w, x: w - x,
            "P": lambda w, x: w + x,
            "Inv": lambda a: a + 1,
            "Explode": lambda a: (_ for _ in ()).throw(ValueError("boom")),
        }
    )
    return env


def _all_cases() -> list[BenchCase]:
    env = _base_env

    monadic = [
        ("plus", "+ x"),
        ("minus", "- x"),
        ("times_sign", "× xs"),
        ("divide_recip", "÷ x"),
        ("power_exp", "⋆ xf"),
        ("not", "¬ binv"),
        ("reverse", "⌽ x"),
        ("transpose", "⍉ mat"),
        ("range", "↕ n"),
        ("ravel", "⥊ mat"),
        ("floor", "⌊ xf"),
        ("ceil", "⌈ xf"),
        ("abs", "| xs"),
        ("sqrt", "√ x"),
        ("shape", "≢ mat"),
        ("rank", "= mat"),
        ("length", "≠ x"),
        ("left", "⊣ x"),
        ("right", "⊢ x"),
    ]

    dyadic = [
        ("plus", "x + y"),
        ("minus", "x - y"),
        ("times", "x × y"),
        ("divide", "x ÷ y"),
        ("power", "x ⋆ 2"),
        ("lt", "x < y"),
        ("le", "x ≤ y"),
        ("gt", "x > y"),
        ("ge", "x ≥ y"),
        ("eq", "x = y"),
        ("ne", "x ≠ y"),
        ("reshape", "shape2 ⥊ xshape"),
        ("floor", "x ⌊ y"),
        ("ceil", "x ⌈ y"),
        ("modulus", "x | y"),
        ("root", "2 √ x"),
        ("left", "x ⊣ y"),
        ("right", "x ⊢ y"),
        ("take", "take ↑ x"),
        ("drop", "drop ↓ x"),
        ("rotate", "rot ⌽ x"),
        ("transpose_axes", "axes ⍉ mat"),
        ("join", "x ∾ y"),
        ("replicate", "cnt / x"),
        ("pick", "pick ⊑ x"),
        ("select", "idx ⊏ x"),
    ]

    modifiers = [
        ("fold", "+´ x"),
        ("scan", "+` x"),
        ("each", "F¨ x"),
        ("cells", "F˘ mat"),
        ("table", "x F⌜ y"),
        ("insert", "F˝ x"),
        ("insert_init", "0 F˝ x"),
        ("constant", "3˙ x0"),
        ("self_swap_dyad", "2 M˜ 5"),
        ("undo", "2 (+⁼) 5"),
        ("atop", "F∘G x0"),
        ("over", "2 M○G 3"),
        ("before", "2 F⊸G 3"),
        ("after", "2 F⟜G 3"),
        ("valences", "2 F⊘G 3"),
        ("choose", "Sel◶A‿B x0"),
    ]

    rows: list[BenchCase] = []
    rows.extend(BenchCase("monadic", name, expr, env) for name, expr in monadic)
    rows.extend(BenchCase("dyadic", name, expr, env) for name, expr in dyadic)
    rows.extend(BenchCase("modifier", name, expr, env) for name, expr in modifiers)
    return rows


def _run_case(
    case: BenchCase,
    n: int,
    *,
    samples: int,
    warmup: int,
    repeat_scale: float,
    max_case_seconds: float,
    target_sample_ms: float,
    min_repeats: int,
    cv_target_pct: float,
    max_samples: int,
) -> BenchRow:
    baseline_repeats = _repeats_for_n(n, repeat_scale)
    env = case.env_builder(n)
    started = time.perf_counter()

    def _call():
        out = evaluate(case.expression, env=env)
        block_until_ready(out)
        return out

    try:
        t0 = time.perf_counter()
        first = _call()
        first_ms = (time.perf_counter() - t0) * 1e3
        if time.perf_counter() - started > max_case_seconds:
            raise TimeoutError(f"case timed out after {max_case_seconds:.2f}s")

        for _ in range(warmup):
            _call()
            if time.perf_counter() - started > max_case_seconds:
                raise TimeoutError(f"case timed out after {max_case_seconds:.2f}s")

        repeats = calibrate_repeats(
            lambda: _call(),
            (),
            baseline_repeats=baseline_repeats,
            target_sample_ms=target_sample_ms,
            min_repeats=min_repeats,
        )
        per_call_ms: list[float] = []

        def _sample_once() -> None:
            start = time.perf_counter()
            for _ in range(repeats):
                _call()
                if time.perf_counter() - started > max_case_seconds:
                    raise TimeoutError(f"case timed out after {max_case_seconds:.2f}s")
            elapsed = time.perf_counter() - start
            per_call_ms.append((elapsed / repeats) * 1e3)
            if time.perf_counter() - started > max_case_seconds:
                raise TimeoutError(f"case timed out after {max_case_seconds:.2f}s")

        for _ in range(samples):
            _sample_once()
        while len(per_call_ms) < max_samples:
            m = _mean(per_call_ms)
            if m <= 0:
                break
            sd = _stddev(per_call_ms)
            cv_pct = (sd / m) * 100.0 if sd > 0 else 0.0
            if cv_pct <= cv_target_pct:
                break
            _sample_once()

        mean_ms = _mean(per_call_ms)
        stdev_ms = _stddev(per_call_ms)
        cv_pct = (stdev_ms / mean_ms) * 100.0 if mean_ms > 0 else 0.0
        return BenchRow(
            section=case.section,
            name=case.name,
            expression=case.expression,
            n=n,
            status="ok",
            first_call_ms=first_ms,
            mean_ms=mean_ms,
            stdev_ms=stdev_ms,
            cv_pct=cv_pct,
            p50_ms=_percentile(per_call_ms, 0.50),
            p95_ms=_percentile(per_call_ms, 0.95),
            min_ms=min(per_call_ms),
            max_ms=max(per_call_ms),
            repeats=repeats,
            warmup=warmup,
            samples=len(per_call_ms),
            error=None,
        )
    except Exception as err:  # pragma: no cover - benchmark resilience
        return BenchRow(
            section=case.section,
            name=case.name,
            expression=case.expression,
            n=n,
            status="error",
            first_call_ms=None,
            mean_ms=None,
            stdev_ms=None,
            cv_pct=None,
            p50_ms=None,
            p95_ms=None,
            min_ms=None,
            max_ms=None,
            repeats=baseline_repeats,
            warmup=warmup,
            samples=samples,
            error=str(err),
        )


def _print_summary(rows: list[BenchRow]) -> None:
    print("valence benchmark summary")
    print("section    case                  n      mean(ms)   p95(ms)   cv(%)   status")
    print("--------   ------------------  ------  ---------  --------  ------  ------")
    for row in rows:
        mean_text = "-" if row.mean_ms is None else f"{row.mean_ms:9.4f}"
        p95_text = "-" if row.p95_ms is None else f"{row.p95_ms:8.4f}"
        cv_text = "-" if row.cv_pct is None else f"{row.cv_pct:6.2f}"
        print(f"{row.section:8} {row.name:18} {row.n:6d}  {mean_text:>9}  {p95_text:>8}  {cv_text:>6}  {row.status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CONFIG),
        default="quick",
        help="fixed benchmark profile presets",
    )
    parser.add_argument("--ns", default="10,1000,10000", help="comma-separated n sizes")
    parser.add_argument(
        "--sections",
        default="monadic,dyadic,modifier",
        help="comma-separated subset of sections",
    )
    parser.add_argument("--samples", type=int, default=None, help="timing samples per case")
    parser.add_argument("--warmup", type=int, default=None, help="warmup rounds before timing")
    parser.add_argument("--target-sample-ms", type=float, default=None, help="target wall time per sample")
    parser.add_argument("--min-repeats", type=int, default=None, help="minimum repeats after calibration")
    parser.add_argument("--cv-target", type=float, default=None, help="adaptive sampling CV target percent")
    parser.add_argument("--max-samples", type=int, default=None, help="adaptive sampling cap")
    parser.add_argument(
        "--repeat-scale",
        type=float,
        default=None,
        help="scale factor for internal repeat counts (defaults from --profile)",
    )
    parser.add_argument(
        "--max-case-seconds",
        type=float,
        default=None,
        help="soft timeout per benchmark case (seconds)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path for machine-readable output",
    )
    args = parser.parse_args()
    affinity_info = configure_cpu_affinity_from_env()

    profile = PROFILE_CONFIG[args.profile]
    samples = int(profile["samples"] if args.samples is None else args.samples)
    warmup = int(profile["warmup"] if args.warmup is None else args.warmup)
    target_sample_ms = float(profile["target_sample_ms"] if args.target_sample_ms is None else args.target_sample_ms)
    min_repeats = int(profile["min_repeats"] if args.min_repeats is None else args.min_repeats)
    cv_target_pct = float(profile["cv_target_pct"] if args.cv_target is None else args.cv_target)
    max_samples = int(profile["max_samples"] if args.max_samples is None else args.max_samples)
    repeat_scale = float(profile["repeat_scale"] if args.repeat_scale is None else args.repeat_scale)
    max_case_seconds = float(
        profile["max_case_seconds"] if args.max_case_seconds is None else args.max_case_seconds
    )
    if max_samples < samples:
        max_samples = samples

    ns = _sizes_from_arg(args.ns)
    wanted_sections = {part.strip() for part in args.sections.split(",") if part.strip()}
    valid = {"monadic", "dyadic", "modifier"}
    unknown = wanted_sections - valid
    if unknown:
        raise SystemExit(f"Unknown sections: {sorted(unknown)}")

    all_cases = [case for case in _all_cases() if case.section in wanted_sections]
    rows: list[BenchRow] = []

    print(f"sizes: {ns}")
    print(
        "profile: "
        f"{args.profile} (samples={samples}, warmup={warmup}, repeat_scale={repeat_scale:.3f}, "
        f"target_sample_ms={target_sample_ms:.1f}, min_repeats={min_repeats}, "
        f"cv_target={cv_target_pct:.1f}%, max_samples={max_samples}, "
        f"max_case_seconds={max_case_seconds:.2f})"
    )
    print(f"sections: {sorted(wanted_sections)}")
    print(f"host: backend={jax.default_backend()}, affinity={affinity_info.get('active')}")
    print()

    for n in ns:
        print(f"n={n}")
        for case in all_cases:
            row = _run_case(
                case,
                n,
                samples=samples,
                warmup=warmup,
                repeat_scale=repeat_scale,
                max_case_seconds=max_case_seconds,
                target_sample_ms=target_sample_ms,
                min_repeats=min_repeats,
                cv_target_pct=cv_target_pct,
                max_samples=max_samples,
            )
            rows.append(row)
        print(f"completed n={n} ({len(all_cases)} cases)")

    print()
    _print_summary(rows)

    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "sizes": list(ns),
            "profile": args.profile,
            "sections": sorted(wanted_sections),
            "samples": samples,
            "warmup": warmup,
            "target_sample_ms": target_sample_ms,
            "min_repeats": min_repeats,
            "cv_target_pct": cv_target_pct,
            "max_samples": max_samples,
            "repeat_scale": repeat_scale,
            "max_case_seconds": max_case_seconds,
            "affinity": affinity_info,
            "host": host_metadata(),
            "rows": [asdict(row) for row in rows],
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {outpath}")


if __name__ == "__main__":
    main()
