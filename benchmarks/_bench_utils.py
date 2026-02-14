"""Shared benchmark runtime helpers."""

from __future__ import annotations

import math
import os
import platform
import time
from typing import Any

import jax

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JAX_NUM_THREADS",
    "XLA_FLAGS",
)


def configure_cpu_affinity_from_env() -> dict[str, Any]:
    requested = os.environ.get("BQN_JAX_BENCH_CPU_AFFINITY", "").strip()
    info: dict[str, Any] = {"requested": requested or None, "applied": False, "active": None}
    if not requested:
        return info
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        return info

    cpus = _parse_affinity_spec(requested)
    if not cpus:
        return info
    try:
        os.sched_setaffinity(0, cpus)
        info["applied"] = True
        info["active"] = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception:
        info["applied"] = False
    return info


def _parse_affinity_spec(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_raw, hi_raw = token.split("-", 1)
            lo = int(lo_raw.strip())
            hi = int(hi_raw.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
            continue
        out.add(int(token))
    return out


def thread_env_snapshot() -> dict[str, str]:
    return {name: os.environ[name] for name in THREAD_ENV_VARS if name in os.environ}


def host_metadata() -> dict[str, Any]:
    active_affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            active_affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except Exception:
            active_affinity = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": getattr(jax, "__version__", "unknown"),
        "backend": jax.default_backend(),
        "cpu_count": os.cpu_count(),
        "active_cpu_affinity": active_affinity,
        "thread_env": thread_env_snapshot(),
    }


def block_until_ready(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple):
        for item in value:
            block_until_ready(item)
        return
    if isinstance(value, list):
        for item in value:
            block_until_ready(item)


def percentile(values: list[float], q: float) -> float:
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def calibrate_repeats(
    fn,
    args: tuple[object, ...],
    *,
    baseline_repeats: int,
    target_sample_ms: float,
    min_repeats: int,
    max_repeats: int = 200_000,
) -> int:
    trial = max(4, min_repeats // 4)
    start_ns = time.perf_counter_ns()
    for _ in range(trial):
        block_until_ready(fn(*args))
    elapsed_ns = time.perf_counter_ns() - start_ns
    per_call_ns = max(elapsed_ns / trial, 1_000.0)
    target_ns = max(target_sample_ms, 1.0) * 1e6
    dynamic = int(math.ceil(target_ns / per_call_ns))
    return int(max(baseline_repeats, min_repeats, min(dynamic, max_repeats)))


def sample_adaptive_ms(
    fn,
    args: tuple[object, ...],
    *,
    repeats: int,
    warmup: int,
    samples: int,
    cv_target_pct: float,
    max_samples: int,
) -> list[float]:
    for _ in range(max(0, warmup)):
        block_until_ready(fn(*args))

    rows: list[float] = []

    def _once() -> None:
        start_ns = time.perf_counter_ns()
        for _ in range(repeats):
            block_until_ready(fn(*args))
        elapsed_ns = time.perf_counter_ns() - start_ns
        rows.append((elapsed_ns / repeats) / 1e6)

    for _ in range(samples):
        _once()
    while len(rows) < max_samples:
        m = mean(rows)
        if m <= 0:
            break
        s = stddev(rows)
        cv_pct = (s / m) * 100.0 if s > 0 else 0.0
        if cv_pct <= cv_target_pct:
            break
        _once()
    return rows
