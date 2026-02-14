"""Profile boxed/list-heavy interpreter paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp

from bqn_jax import evaluate


@dataclass(frozen=True)
class Case:
    name: str
    expr: str
    repeats: int


@dataclass(frozen=True)
class Row:
    name: str
    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    samples: int


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


def _run_case(case: Case, n: int, *, samples: int) -> Row:
    base = jnp.arange(n, dtype=jnp.int32)
    env = {
        "idx": jnp.mod(jnp.arange(max(1, n // 4), dtype=jnp.int32) * 5 + 1, n),
        "classes": jnp.mod(base, max(2, min(32, n))),
        "boxed": [jnp.asarray(1), jnp.asarray([2, 3], dtype=jnp.int32), jnp.asarray([4, 5], dtype=jnp.int32)],
        "boxed_sort": [
            [jnp.asarray(1), jnp.asarray([2, 3], dtype=jnp.int32)],
            [jnp.asarray(1), jnp.asarray([2, 2], dtype=jnp.int32)],
            jnp.asarray(0),
        ],
        "x": base,
    }

    rows: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(case.repeats):
            evaluate(case.expr, env=env)
        end = time.perf_counter()
        rows.append((end - start) * 1e3 / case.repeats)
    return Row(
        name=case.name,
        n=n,
        mean_ms=sum(rows) / len(rows),
        p50_ms=_percentile(rows, 0.50),
        p95_ms=_percentile(rows, 0.95),
        min_ms=min(rows),
        max_ms=max(rows),
        repeats=case.repeats,
        samples=samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", default="16,128,1024", help="comma-separated sizes")
    parser.add_argument("--samples", type=int, default=3, help="timing samples")
    parser.add_argument("--json-out", default="", help="optional output JSON")
    args = parser.parse_args()

    ns = [int(x.strip()) for x in args.ns.split(",") if x.strip()]
    cases = [
        Case("boxed_merge", "> boxed", repeats=300),
        Case("boxed_sort", "∧ boxed_sort", repeats=150),
        Case("group_boxed", "⊔ classes", repeats=200),
        Case("select_boxed", "idx ⊏ boxed", repeats=200),
    ]

    rows: list[Row] = []
    print("Boxed path benchmark")
    for n in ns:
        for case in cases:
            row = _run_case(case, n, samples=args.samples)
            rows.append(row)
            print(f"{case.name:12} n={n:4d} mean={row.mean_ms:8.3f}ms p95={row.p95_ms:8.3f}ms")

    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "sizes": ns,
            "samples": args.samples,
            "rows": [asdict(row) for row in rows],
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {outpath}")


if __name__ == "__main__":
    main()
