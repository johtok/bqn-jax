"""bencharray-inspired benchmark subset for bqn-jax."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

import jax.numpy as jnp

from bqn_jax import ShapePolicy, compile_expression, evaluate

SizeKind = Literal["flat", "matrix"]
EvalMode = Literal["compiled", "evaluate"]


@dataclass(frozen=True)
class BenchCase:
    group: str
    name: str
    source: str
    mode: EvalMode
    size_kind: SizeKind
    target_elements_per_timing: int
    arg_names: tuple[str, ...]
    build_inputs: Callable[[int], tuple[object, ...] | dict[str, object]]


@dataclass(frozen=True)
class BenchRow:
    group: str
    case: str
    source: str
    mode: EvalMode
    size: int
    size_label: str
    elements: int
    repeats: int
    seconds: float


def _sequence(n: int, *, mul: int, add: int, mod: int) -> jnp.ndarray:
    idx = jnp.arange(n, dtype=jnp.int32)
    return jnp.mod(idx * mul + add, mod)


def _build_flat_xy(n: int) -> tuple[object, ...]:
    x = _sequence(n, mul=73, add=19, mod=104_729).astype(jnp.float32)
    y = _sequence(n, mul=97, add=29, mod=104_729).astype(jnp.float32)
    return x, y


def _build_flat_x(n: int) -> tuple[object, ...]:
    x = _sequence(n, mul=67, add=13, mod=1_000_003).astype(jnp.float32)
    return (x,)


def _build_matrix_xy(side: int) -> tuple[object, ...]:
    total = side * side
    base = _sequence(total, mul=37, add=11, mod=65_521).astype(jnp.float32)
    x = jnp.reshape(base, (side, side))
    return x, x + 1.0


def _build_group_env(n: int) -> dict[str, object]:
    classes = max(4, int(n**0.5))
    x = _sequence(n, mul=19, add=5, mod=classes)
    return {"x": x}


def _build_replicate_env(n: int) -> dict[str, object]:
    x = _sequence(n, mul=43, add=17, mod=2048)
    m = jnp.mod(jnp.arange(n, dtype=jnp.int32), 3)
    return {"x": x, "m": m}


def _build_reshape_env(side: int) -> dict[str, object]:
    total = side * side
    x = _sequence(total, mul=59, add=7, mod=4096)
    s = jnp.asarray([side, side], dtype=jnp.int32)
    return {"x": x, "s": s}


def _build_reverse_env(n: int) -> dict[str, object]:
    return {"x": _sequence(n, mul=71, add=3, mod=1_048_573)}


def _build_search_env(n: int) -> dict[str, object]:
    q = max(1, n // 8)
    x = _sequence(q, mul=113, add=31, mod=2 * n + 1)
    y = _sequence(n, mul=89, add=23, mod=2 * n + 1)
    return {"x": x, "y": y}


def _build_select_env(n: int) -> dict[str, object]:
    picks = max(1, n // 4)
    x = _sequence(n, mul=47, add=13, mod=2_097_169)
    i = jnp.mod(jnp.arange(picks, dtype=jnp.int32) * 131 + 7, n)
    return {"x": x, "i": i}


def _build_sort_env(n: int) -> dict[str, object]:
    x = _sequence(n, mul=149, add=41, mod=9_973)
    return {"x": x}


def _build_transpose_env(side: int) -> dict[str, object]:
    total = side * side
    x = _sequence(total, mul=83, add=27, mod=50_021)
    return {"x": jnp.reshape(x, (side, side))}


CASES: tuple[BenchCase, ...] = (
    BenchCase(
        group="arith",
        name="times_plus",
        source="(x × x) + y",
        mode="compiled",
        size_kind="flat",
        target_elements_per_timing=3_000_000,
        arg_names=("x", "y"),
        build_inputs=_build_flat_xy,
    ),
    BenchCase(
        group="fold",
        name="sum",
        source="0 +´ x",
        mode="compiled",
        size_kind="flat",
        target_elements_per_timing=2_000_000,
        arg_names=("x",),
        build_inputs=_build_flat_x,
    ),
    BenchCase(
        group="group",
        name="group_monadic",
        source="⊔ x",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=600_000,
        arg_names=(),
        build_inputs=_build_group_env,
    ),
    BenchCase(
        group="replicate",
        name="mask_replicate",
        source="m / x",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=1_000_000,
        arg_names=(),
        build_inputs=_build_replicate_env,
    ),
    BenchCase(
        group="reshape",
        name="reshape_square",
        source="s ⥊ x",
        mode="evaluate",
        size_kind="matrix",
        target_elements_per_timing=1_500_000,
        arg_names=(),
        build_inputs=_build_reshape_env,
    ),
    BenchCase(
        group="reverse",
        name="reverse_flat",
        source="⌽ x",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=1_500_000,
        arg_names=(),
        build_inputs=_build_reverse_env,
    ),
    BenchCase(
        group="search",
        name="member_of",
        source="x ∊ y",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=700_000,
        arg_names=(),
        build_inputs=_build_search_env,
    ),
    BenchCase(
        group="select",
        name="index_select",
        source="i ⊏ x",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=1_000_000,
        arg_names=(),
        build_inputs=_build_select_env,
    ),
    BenchCase(
        group="sort",
        name="sort_up",
        source="∧ x",
        mode="evaluate",
        size_kind="flat",
        target_elements_per_timing=200_000,
        arg_names=(),
        build_inputs=_build_sort_env,
    ),
    BenchCase(
        group="transpose",
        name="transpose_square",
        source="⍉ x",
        mode="evaluate",
        size_kind="matrix",
        target_elements_per_timing=1_500_000,
        arg_names=(),
        build_inputs=_build_transpose_env,
    ),
    BenchCase(
        group="arith",
        name="matrix_plus",
        source="x + y",
        mode="compiled",
        size_kind="matrix",
        target_elements_per_timing=2_000_000,
        arg_names=("x", "y"),
        build_inputs=_build_matrix_xy,
    ),
)


def _elements(kind: SizeKind, size: int) -> int:
    if kind == "flat":
        return size
    return size * size


def _size_label(kind: SizeKind, size: int) -> str:
    exp = size.bit_length() - 1
    if kind == "flat":
        return f"2^{exp}"
    return f"2^{exp}x2^{exp}"


def _powers_of_two(min_exp: int, max_exp: int) -> list[int]:
    if min_exp > max_exp:
        raise ValueError("minimum exponent cannot be greater than maximum exponent")
    return [1 << exp for exp in range(min_exp, max_exp + 1)]


def _repeats(elements: int, target_elements_per_timing: int) -> int:
    value = target_elements_per_timing // max(elements, 1)
    return max(2, min(20, value))


def _block(value: object) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple):
        for item in value:
            _block(item)


def _timeit_payload(run_once: Callable[[object], object], payload: object, *, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        _block(run_once(payload))
    start = time.perf_counter()
    for _ in range(repeats):
        _block(run_once(payload))
    end = time.perf_counter()
    return (end - start) / repeats


def _runner(case: BenchCase) -> Callable[[object], object]:
    if case.mode == "compiled":
        compiled = compile_expression(
            case.source,
            arg_names=case.arg_names,
            shape_policy=ShapePolicy(kind="dynamic"),
        )
        jitted = compiled.jit()
        return lambda payload: jitted(*payload)

    return lambda payload: evaluate(case.source, env=payload)


def _print_case_rows(case: BenchCase, rows: list[BenchRow]) -> None:
    print(f"{case.name} ({case.mode})  {case.source}")
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


def _run_case(case: BenchCase, *, flat_sizes: list[int], matrix_sizes: list[int]) -> list[BenchRow]:
    sizes = flat_sizes if case.size_kind == "flat" else matrix_sizes
    run_once = _runner(case)
    rows: list[BenchRow] = []

    for size in sizes:
        payload = case.build_inputs(size)
        _block(run_once(payload))  # Compile/warm up once per size.
        elements = _elements(case.size_kind, size)
        reps = _repeats(elements, case.target_elements_per_timing)
        sec = _timeit_payload(run_once, payload, repeats=reps, warmup=1)
        rows.append(
            BenchRow(
                group=case.group,
                case=case.name,
                source=case.source,
                mode=case.mode,
                size=size,
                size_label=_size_label(case.size_kind, size),
                elements=elements,
                repeats=reps,
                seconds=sec,
            )
        )
    return rows


def _group_map() -> dict[str, list[BenchCase]]:
    groups: dict[str, list[BenchCase]] = {}
    for case in CASES:
        groups.setdefault(case.group, []).append(case)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreated subset of mlochbaum/bencharray for bqn-jax.",
    )
    parser.add_argument(
        "mode_or_groups",
        nargs="*",
        help="Use `list`, `measure [groups...]`, or pass groups directly.",
    )
    parser.add_argument("--flat-min-exp", type=int, default=10, help="minimum exponent for flat sizes (2^exp)")
    parser.add_argument("--flat-max-exp", type=int, default=16, help="maximum exponent for flat sizes (2^exp)")
    parser.add_argument("--matrix-min-exp", type=int, default=4, help="minimum exponent for matrix side (2^exp)")
    parser.add_argument("--matrix-max-exp", type=int, default=7, help="maximum exponent for matrix side (2^exp)")
    parser.add_argument(
        "--outdir",
        default="benchmarks/output/bencharray_recreated",
        help="directory where JSON timing output is written",
    )
    args = parser.parse_args()

    groups = _group_map()
    known_groups = sorted(groups)

    items = args.mode_or_groups
    mode = "measure"
    selected: list[str]
    if not items:
        selected = known_groups
    elif items[0] == "list":
        mode = "list"
        selected = known_groups
    elif items[0] == "measure":
        selected = items[1:] or known_groups
    else:
        selected = items

    unknown = [name for name in selected if name not in groups]
    if unknown:
        raise SystemExit(f"Unknown benchmark groups: {', '.join(unknown)}")

    if mode == "list":
        print("Available groups:")
        for name in known_groups:
            print(name)
        return

    flat_sizes = _powers_of_two(args.flat_min_exp, args.flat_max_exp)
    matrix_sizes = _powers_of_two(args.matrix_min_exp, args.matrix_max_exp)

    print("bencharray-recreated (bqn-jax subset)")
    print("Inspired by https://github.com/mlochbaum/bencharray")
    print(f"groups: {', '.join(selected)}")
    print(
        f"flat sizes: 2^{args.flat_min_exp}..2^{args.flat_max_exp}; "
        f"matrix side sizes: 2^{args.matrix_min_exp}..2^{args.matrix_max_exp}"
    )
    print()

    all_rows: list[BenchRow] = []
    for group in selected:
        print(group)
        print("-" * len(group))
        for case in groups[group]:
            rows = _run_case(case, flat_sizes=flat_sizes, matrix_sizes=matrix_sizes)
            all_rows.extend(rows)
            _print_case_rows(case, rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outpath = outdir / f"timings-{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "source": "bencharray_recreated.py",
        "groups": selected,
        "flat_exp_range": [args.flat_min_exp, args.flat_max_exp],
        "matrix_exp_range": [args.matrix_min_exp, args.matrix_max_exp],
        "rows": [asdict(row) for row in all_rows],
    }
    outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote timing output: {outpath}")


if __name__ == "__main__":
    main()
