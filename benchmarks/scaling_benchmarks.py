"""Scaling benchmarks over exponentially increasing workload sizes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression


SizeKind = Literal["flat", "matrix"]


@dataclass(frozen=True)
class ScalingSpec:
    name: str
    note: str
    kind: SizeKind
    expression: str
    arg_names: tuple[str, ...]
    target_elements_per_timing: int
    build_jax_args: Callable[[int], tuple[object, ...]]
    build_cbqn_code: Callable[[int, int], str]


@dataclass(frozen=True)
class ScalingRow:
    size: int
    size_label: str
    elements: int
    repeats: int
    seconds: float


def _block(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple):
        for item in value:
            _block(item)


def _timeit(fn, *args: object, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        _block(fn(*args))
    start = time.perf_counter()
    for _ in range(repeats):
        _block(fn(*args))
    end = time.perf_counter()
    return (end - start) / repeats


def _parse_cbqn_number(raw: str) -> float:
    line = raw.strip().splitlines()[-1].strip()
    # CBQN uses `¯` for negative exponents.
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


def _elements_for_size(kind: SizeKind, size: int) -> int:
    if kind == "flat":
        return size
    return size * size


def _label_for_size(kind: SizeKind, size: int) -> str:
    exp = size.bit_length() - 1
    if kind == "flat":
        return f"2^{exp}"
    return f"2^{exp}x2^{exp}"


def _repeats_for_elements(elements: int, target_elements_per_timing: int) -> int:
    repeats = target_elements_per_timing // max(elements, 1)
    return max(3, min(100, repeats))


def _build_specs() -> list[ScalingSpec]:
    def build_flat_xy(n: int) -> tuple[object, ...]:
        x = jnp.arange(n, dtype=jnp.float32)
        return (x, x + 1.0)

    def build_flat_x(n: int) -> tuple[object, ...]:
        return (jnp.arange(n, dtype=jnp.float32),)

    def build_matrix_xy(side: int) -> tuple[object, ...]:
        count = side * side
        x = jnp.reshape(jnp.arange(count, dtype=jnp.float32), (side, side))
        return (x, x + 1.0)

    def build_matrix_x(side: int) -> tuple[object, ...]:
        count = side * side
        x = jnp.reshape(jnp.arange(count, dtype=jnp.float32), (side, side))
        return (x,)

    return [
        ScalingSpec(
            name="flat_elementwise",
            note="flat list elementwise arithmetic",
            kind="flat",
            expression="(x × x) + y",
            arg_names=("x", "y"),
            target_elements_per_timing=20_000_000,
            build_jax_args=build_flat_xy,
            build_cbqn_code=lambda n, repeats: f"{repeats} (1+↕{n})⊸{{𝕩×𝕩+𝕨}}•_timed ↕{n}",
        ),
        ScalingSpec(
            name="flat_reduce_plus",
            note="flat list reduction (0 +´)",
            kind="flat",
            expression="0 +´ x",
            arg_names=("x",),
            target_elements_per_timing=10_000_000,
            build_jax_args=build_flat_x,
            build_cbqn_code=lambda n, repeats: f"{repeats} 0⊸+´•_timed ↕{n}",
        ),
        ScalingSpec(
            name="matrix_elementwise",
            note="multidimensional elementwise arithmetic",
            kind="matrix",
            expression="(x × x) + y",
            arg_names=("x", "y"),
            target_elements_per_timing=12_000_000,
            build_jax_args=build_matrix_xy,
            build_cbqn_code=(
                lambda side, repeats: (
                    f"{repeats} ({side}‿{side}⥊(1+↕{side * side}))⊸{{𝕩×𝕩+𝕨}}•_timed "
                    f"{side}‿{side}⥊↕{side * side}"
                )
            ),
        ),
        ScalingSpec(
            name="matrix_reduce_major",
            note="multidimensional reduction over major axis",
            kind="matrix",
            expression="0 +´ x",
            arg_names=("x",),
            target_elements_per_timing=8_000_000,
            build_jax_args=build_matrix_x,
            build_cbqn_code=lambda side, repeats: f"{repeats} {{+´˘⍉𝕩}}•_timed {side}‿{side}⥊↕{side * side}",
        ),
    ]


def _print_section(title: str) -> None:
    print(title)
    print("-" * len(title))


def _print_rows(engine: str, rows: list[ScalingRow]) -> None:
    print(engine)
    print(f"{'size':>11} {'elements':>12} {'repeats':>8} {'time(ms)':>11} {'growth':>8}")
    prev: float | None = None
    for row in rows:
        growth = "-" if prev is None else f"{row.seconds / prev:7.2f}x"
        print(
            f"{row.size_label:>11} "
            f"{row.elements:12d} "
            f"{row.repeats:8d} "
            f"{row.seconds * 1e3:11.4f} "
            f"{growth:>8}"
        )
        prev = row.seconds
    print()


def _print_relative(cbqn_rows: list[ScalingRow], jax_rows: list[ScalingRow]) -> None:
    jax_by_size = {row.size: row.seconds for row in jax_rows}
    print("CBQN/JAX")
    print(f"{'size':>11} {'ratio':>10}")
    for row in cbqn_rows:
        jax_sec = jax_by_size.get(row.size)
        if jax_sec is None:
            continue
        ratio = row.seconds / jax_sec if jax_sec > 0 else float("inf")
        print(f"{row.size_label:>11} {ratio:10.3f}x")
    print()


def _run_jax_scaling(spec: ScalingSpec, sizes: list[int]) -> list[ScalingRow]:
    compiled = compile_expression(
        spec.expression,
        arg_names=spec.arg_names,
        shape_policy=ShapePolicy(kind="dynamic"),
    )
    jitted = compiled.jit()

    rows: list[ScalingRow] = []
    for size in sizes:
        args = spec.build_jax_args(size)
        _block(jitted(*args))  # compile for the current shape before timing
        elements = _elements_for_size(spec.kind, size)
        repeats = _repeats_for_elements(elements, spec.target_elements_per_timing)
        sec = _timeit(jitted, *args, repeats=repeats, warmup=2)
        rows.append(
            ScalingRow(
                size=size,
                size_label=_label_for_size(spec.kind, size),
                elements=elements,
                repeats=repeats,
                seconds=sec,
            )
        )
    return rows


def _run_cbqn_scaling(spec: ScalingSpec, sizes: list[int], cbqn_bin: str) -> list[ScalingRow]:
    rows: list[ScalingRow] = []
    for size in sizes:
        elements = _elements_for_size(spec.kind, size)
        repeats = _repeats_for_elements(elements, spec.target_elements_per_timing)
        code = spec.build_cbqn_code(size, repeats)
        sec = _run_cbqn_timed(cbqn_bin, code)
        rows.append(
            ScalingRow(
                size=size,
                size_label=_label_for_size(spec.kind, size),
                elements=elements,
                repeats=repeats,
                seconds=sec,
            )
        )
    return rows


def _powers_of_two(min_exp: int, max_exp: int) -> list[int]:
    if min_exp > max_exp:
        raise ValueError("minimum exponent cannot be greater than maximum exponent")
    return [1 << exp for exp in range(min_exp, max_exp + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scaling benchmarks across exponentially increasing workload sizes."
    )
    parser.add_argument("--flat-min-exp", type=int, default=10, help="minimum exponent for flat sizes (2^exp)")
    parser.add_argument("--flat-max-exp", type=int, default=20, help="maximum exponent for flat sizes (2^exp)")
    parser.add_argument("--matrix-min-exp", type=int, default=4, help="minimum exponent for matrix side (2^exp)")
    parser.add_argument("--matrix-max-exp", type=int, default=9, help="maximum exponent for matrix side (2^exp)")
    parser.add_argument(
        "--no-cbqn",
        action="store_true",
        help="Skip CBQN runs even if a CBQN binary is available.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path to write machine-readable scaling results",
    )
    args = parser.parse_args()

    flat_sizes = _powers_of_two(args.flat_min_exp, args.flat_max_exp)
    matrix_sizes = _powers_of_two(args.matrix_min_exp, args.matrix_max_exp)

    print("Scaling benchmark suite (exponential workloads)")
    print(
        f"flat sizes: 2^{args.flat_min_exp} .. 2^{args.flat_max_exp}; "
        f"matrix side sizes: 2^{args.matrix_min_exp} .. 2^{args.matrix_max_exp}"
    )
    print()

    cbqn_bin = None if args.no_cbqn else _find_cbqn_binary()
    if args.no_cbqn:
        print("CBQN disabled by --no-cbqn.")
        print()
    elif cbqn_bin is None:
        print("CBQN not found; running JAX-only scaling benchmarks.")
        print()

    specs = _build_specs()
    payload_rows: list[dict[str, object]] = []
    for spec in specs:
        sizes = flat_sizes if spec.kind == "flat" else matrix_sizes
        _print_section(f"{spec.name}: {spec.note}")

        jax_rows = _run_jax_scaling(spec, sizes)
        _print_rows("JAX-backed bqn-jax", jax_rows)
        payload_rows.append(
            {
                "spec": spec.name,
                "note": spec.note,
                "engine": "jax",
                "rows": [asdict(row) for row in jax_rows],
            }
        )

        if cbqn_bin is not None:
            cbqn_rows = _run_cbqn_scaling(spec, sizes, cbqn_bin)
            _print_rows(f"CBQN ({cbqn_bin})", cbqn_rows)
            _print_relative(cbqn_rows, jax_rows)
            payload_rows.append(
                {
                    "spec": spec.name,
                    "note": spec.note,
                    "engine": "cbqn",
                    "rows": [asdict(row) for row in cbqn_rows],
                }
            )

    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "flat_exp_range": [args.flat_min_exp, args.flat_max_exp],
                "matrix_exp_range": [args.matrix_min_exp, args.matrix_max_exp],
                "cbqn_bin": cbqn_bin,
            },
            "results": payload_rows,
        }
        outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON scaling output: {outpath}")


if __name__ == "__main__":
    main()
