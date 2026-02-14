"""JAX-oriented lowering, tracing, and transform helpers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import numbers
import os
from pathlib import Path
import pickle
import platform
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal
import weakref

import jax
from jax import lax
import jax.numpy as jnp

from .ast import Assign, Block, Call, Char, Export, Expr, Infix, Member, Mod1, Mod2, Name, Nothing, Null, Number, Prefix, String, Train, Vector
from .errors import (
    BQNError,
    BQNParseError,
    BQNRuntimeError,
    BQNShapeError,
    BQNTypeError,
    BQNUnsupportedError,
    classify_runtime_exception,
)
from .parser import ParseError, parse

_PRIMITIVE_FUNCTION_NAMES = {
    "+",
    "-",
    "×",
    "÷",
    "⋆",
    "!",
    "<",
    "≤",
    ">",
    "≥",
    "=",
    "≠",
    "≡",
    "≢",
    "⊣",
    "⊢",
    "↕",
    "⥊",
    "⌊",
    "⌈",
    "|",
    "√",
    "⌽",
    "⍉",
    "∾",
    "↑",
    "↓",
    "⊑",
    "⊏",
    "⊐",
    "⊒",
    "⊔",
    "≍",
    "/",
    "¬",
    "∧",
    "∨",
    "⍋",
    "⍒",
    "∊",
    "⍷",
}

_SUPPORTED_PREFIX = {"+", "-", "×", "÷", "¬", "⌊", "⌈", "|", "√", "=", "≠", "≢", "≡", "↕", "⥊", "⌽", "⍉", "⊣", "⊢"}
_SUPPORTED_INFIX = {
    "+",
    "-",
    "×",
    "÷",
    "⋆",
    "√",
    "⌊",
    "⌈",
    "|",
    "<",
    "≤",
    ">",
    "≥",
    "=",
    "≠",
    "∧",
    "∨",
    "⥊",
    "⊣",
    "⊢",
    "∾",
    "↑",
    "↓",
    "⌽",
    "⍉",
}
_SUPPORTED_FOLD = {"+", "×", "⌊", "⌈", "∧", "∨"}

_SYSTEM_CONSTANTS = {
    "•pi": jnp.pi,
    "•e": jnp.e,
    "•inf": jnp.inf,
    "•nan": jnp.nan,
    "•i": 1j,
    "•true": 1,
    "•false": 0,
}

_MISSING = object()
_USE_INT_POWER_FAST_PATH = os.environ.get("BQN_JAX_DISABLE_INT_POWER_FAST_PATH", "0") != "1"
_USE_PRIMITIVE_FOLD_FAST_PATH = os.environ.get("BQN_JAX_DISABLE_PRIMITIVE_FOLD_FAST_PATH", "0") != "1"
_USE_PERSISTENT_IR_CACHE = os.environ.get("BQN_JAX_DISABLE_PERSISTENT_IR_CACHE", "0") != "1"
_PERSISTENT_IR_CACHE_DIR = Path(os.environ.get("BQN_JAX_IR_CACHE_DIR", ".bqn_jax_cache/ir"))
_PERSISTENT_IR_CACHE_VERSION = "v1"
_PARSE_CACHE_MAX = max(1, int(os.environ.get("BQN_JAX_PARSE_CACHE_MAX", "512")))
_PRELOWER_CACHE_MAX = max(1, int(os.environ.get("BQN_JAX_PRELOWER_CACHE_MAX", "1024")))
_COMPILED_IR_CACHE: dict[tuple[object, ...], "JaxIR"] = {}
_COMPILED_IR_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0, "disk_hits": 0, "disk_misses": 0}
_PRELOWER_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}
_TRANSFORM_HELPER_CACHE: dict[tuple[object, ...], object] = {}
_TRANSFORM_HELPER_STATS: dict[str, int] = {"hits": 0, "misses": 0}
_GRAD_KERNEL_CACHE: dict[tuple[object, ...], object] = {}
_IR_CONST_VALUE_CACHE: dict[int, tuple[object | None, ...]] = {}
_IR_CONST_VALUE_CACHE_REFS: dict[int, weakref.ref] = {}


def _unsupported_runtime_only(feature: str) -> BQNUnsupportedError:
    return BQNUnsupportedError(
        f"Runtime-only feature in interpreter path: {feature}. Use evaluate() for this construct."
    )


def _unsupported_backend(feature: str) -> BQNUnsupportedError:
    return BQNUnsupportedError(f"Unsupported in JAX IR backend: {feature}")


@lru_cache(maxsize=_PARSE_CACHE_MAX)
def _parse_cached(source: str) -> Expr:
    return parse(source)


def _rewrite_aliases(constants: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    aliases: list[tuple[str, str]] = []
    for name in sorted(constants):
        value = constants[name]
        if isinstance(value, str) and value in _PRIMITIVE_FUNCTION_NAMES:
            aliases.append((name, value))
    return tuple(aliases)


def _prelower_fingerprint(
    source: str,
    arg_names: tuple[str, ...],
    rewrite_aliases: tuple[tuple[str, str], ...],
    optimize: bool,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(source.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(repr(arg_names).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(repr(rewrite_aliases).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(b"1" if optimize else b"0")
    return digest.hexdigest()


@lru_cache(maxsize=_PRELOWER_CACHE_MAX)
def _prepare_expr_cached(
    fingerprint: str,
    source: str,
    arg_names: tuple[str, ...],
    rewrite_aliases: tuple[tuple[str, str], ...],
    optimize: bool,
) -> Expr:
    # `fingerprint` participates in cache keying for quick identity and auditability.
    _ = fingerprint
    expr = _parse_cached(source)
    alias_map = dict(rewrite_aliases)
    rewritten = _rewrite_for_ir(expr, constants=alias_map, arg_names=frozenset(arg_names))
    if optimize:
        return _optimize_for_ir(rewritten)
    return rewritten


def _is_primitive_function_name(name: str, *, constants: dict[str, object], arg_names: frozenset[str]) -> bool:
    if name in arg_names:
        return False
    if name in _PRIMITIVE_FUNCTION_NAMES:
        return True
    alias = constants.get(name)
    return isinstance(alias, str) and alias in _PRIMITIVE_FUNCTION_NAMES


def _is_function_expr(expr: Expr, *, constants: dict[str, object], arg_names: frozenset[str]) -> bool:
    if isinstance(expr, Name):
        return _is_primitive_function_name(expr.value, constants=constants, arg_names=arg_names)
    if isinstance(expr, (Mod1, Mod2)):
        return True
    if isinstance(expr, Train):
        if len(expr.parts) == 2:
            return _is_function_expr(expr.parts[0], constants=constants, arg_names=arg_names) and _is_function_expr(
                expr.parts[1], constants=constants, arg_names=arg_names
            )
        if len(expr.parts) == 3:
            return (
                _is_function_expr(expr.parts[0], constants=constants, arg_names=arg_names)
                and _is_function_expr(expr.parts[1], constants=constants, arg_names=arg_names)
                and _is_function_expr(expr.parts[2], constants=constants, arg_names=arg_names)
            )
        return True
    return False


def _is_value_expr(expr: Expr, *, constants: dict[str, object], arg_names: frozenset[str]) -> bool:
    return not _is_function_expr(expr, constants=constants, arg_names=arg_names)


def _split_train_trailing_arg(
    expr: Train, *, constants: dict[str, object], arg_names: frozenset[str]
) -> tuple[Expr, Expr] | None:
    if len(expr.parts) == 2:
        first, second = expr.parts
        if _is_function_expr(first, constants=constants, arg_names=arg_names) and _is_value_expr(
            second, constants=constants, arg_names=arg_names
        ):
            return first, second
        return None

    if len(expr.parts) == 3:
        first, second, third = expr.parts
        if (
            _is_function_expr(first, constants=constants, arg_names=arg_names)
            and _is_function_expr(second, constants=constants, arg_names=arg_names)
            and _is_value_expr(third, constants=constants, arg_names=arg_names)
        ):
            return Train(parts=(first, second)), third
    return None


def _rewrite_for_ir(expr: Expr, *, constants: dict[str, object], arg_names: frozenset[str]) -> Expr:
    """Desugar supported function notation into primitive expression forms.

    This keeps lowering focused on numeric expression nodes while allowing
    compile-time use of train/modifier syntax in a restricted subset.
    """

    if isinstance(expr, Name):
        if expr.value in arg_names:
            return expr
        alias = constants.get(expr.value)
        if isinstance(alias, str) and alias in _PRIMITIVE_FUNCTION_NAMES:
            return Name(value=alias)
        return expr

    if isinstance(expr, Vector):
        return Vector(items=tuple(_rewrite_for_ir(item, constants=constants, arg_names=arg_names) for item in expr.items))

    if isinstance(expr, Prefix):
        return Prefix(op=expr.op, right=_rewrite_for_ir(expr.right, constants=constants, arg_names=arg_names))

    if isinstance(expr, Infix):
        return Infix(
            op=expr.op,
            left=_rewrite_for_ir(expr.left, constants=constants, arg_names=arg_names),
            right=_rewrite_for_ir(expr.right, constants=constants, arg_names=arg_names),
        )

    if isinstance(expr, Mod1):
        return Mod1(op=expr.op, operand=_rewrite_for_ir(expr.operand, constants=constants, arg_names=arg_names))

    if isinstance(expr, Mod2):
        if isinstance(expr.right, Train):
            split = _split_train_trailing_arg(expr.right, constants=constants, arg_names=arg_names)
            if split is not None:
                right_func, right_arg = split
                rewritten_func = Mod2(
                    op=expr.op,
                    left=_rewrite_for_ir(expr.left, constants=constants, arg_names=arg_names),
                    right=_rewrite_for_ir(right_func, constants=constants, arg_names=arg_names),
                )
                rewritten_arg = _rewrite_for_ir(right_arg, constants=constants, arg_names=arg_names)
                return _rewrite_apply_for_ir(rewritten_func, rewritten_arg)
        return Mod2(
            op=expr.op,
            left=_rewrite_for_ir(expr.left, constants=constants, arg_names=arg_names),
            right=_rewrite_for_ir(expr.right, constants=constants, arg_names=arg_names),
        )

    if isinstance(expr, Train):
        if len(expr.parts) == 2:
            first_raw, second_raw = expr.parts
            if _is_value_expr(first_raw, constants=constants, arg_names=arg_names) and isinstance(second_raw, Mod2):
                if isinstance(second_raw.right, Train):
                    split = _split_train_trailing_arg(second_raw.right, constants=constants, arg_names=arg_names)
                    if split is not None:
                        right_func, right_arg = split
                        rewritten_left = _rewrite_for_ir(first_raw, constants=constants, arg_names=arg_names)
                        rewritten_func = Mod2(
                            op=second_raw.op,
                            left=_rewrite_for_ir(second_raw.left, constants=constants, arg_names=arg_names),
                            right=_rewrite_for_ir(right_func, constants=constants, arg_names=arg_names),
                        )
                        rewritten_right = _rewrite_for_ir(right_arg, constants=constants, arg_names=arg_names)
                        return _rewrite_apply_for_ir(rewritten_func, rewritten_right, left=rewritten_left)

        rewritten_parts = tuple(_rewrite_for_ir(part, constants=constants, arg_names=arg_names) for part in expr.parts)

        if len(rewritten_parts) == 2:
            first, second = rewritten_parts
            if _is_function_expr(first, constants=constants, arg_names=arg_names) and _is_value_expr(
                second, constants=constants, arg_names=arg_names
            ):
                return _rewrite_apply_for_ir(first, second)
            return Train(parts=rewritten_parts)

        if len(rewritten_parts) == 3:
            first, second, third = rewritten_parts
            if (
                _is_function_expr(first, constants=constants, arg_names=arg_names)
                and _is_function_expr(second, constants=constants, arg_names=arg_names)
                and _is_value_expr(third, constants=constants, arg_names=arg_names)
            ):
                return _rewrite_apply_for_ir(Train(parts=(first, second)), third)
            if (
                _is_value_expr(first, constants=constants, arg_names=arg_names)
                and _is_function_expr(second, constants=constants, arg_names=arg_names)
                and _is_value_expr(third, constants=constants, arg_names=arg_names)
            ):
                return _rewrite_apply_for_ir(second, third, left=first)
            return Train(parts=rewritten_parts)

        return Train(parts=rewritten_parts)

    if isinstance(expr, Call):
        rewritten_func = _rewrite_for_ir(expr.func, constants=constants, arg_names=arg_names)
        rewritten_right = _rewrite_for_ir(expr.right, constants=constants, arg_names=arg_names)
        rewritten_left = None if expr.left is None else _rewrite_for_ir(expr.left, constants=constants, arg_names=arg_names)
        return _rewrite_apply_for_ir(rewritten_func, rewritten_right, left=rewritten_left)

    return expr


def _is_number_expr(expr: Expr, value: float | int) -> bool:
    if not isinstance(expr, Number):
        return False
    return expr.value == value


def _fold_prefix_literal(op: str, right: Number) -> Number | None:
    x = right.value
    try:
        if op == "+":
            return Number(value=+x)
        if op == "-":
            return Number(value=-x)
        if op == "×":
            if x > 0:
                return Number(value=1)
            if x < 0:
                return Number(value=-1)
            return Number(value=0)
        if op == "÷":
            return Number(value=1 / x)
        if op == "⌊":
            return Number(value=float(x) // 1 if isinstance(x, float) else x)
        if op == "⌈":
            if isinstance(x, float):
                ix = int(x)
                return Number(value=ix if ix == x else ix + (1 if x > 0 else 0))
            return Number(value=x)
        if op == "|":
            return Number(value=abs(x))
        if op == "√":
            return Number(value=float(x) ** 0.5)
        if op == "¬":
            return Number(value=1 if x == 0 else 0)
    except Exception:
        return None
    return None


def _fold_infix_literals(op: str, left: Number, right: Number) -> Number | None:
    w = left.value
    x = right.value
    try:
        if op == "+":
            return Number(value=w + x)
        if op == "-":
            return Number(value=w - x)
        if op == "×":
            return Number(value=w * x)
        if op == "÷":
            return Number(value=w / x)
        if op == "⋆":
            return Number(value=w**x)
        if op == "⌊":
            return Number(value=min(w, x))
        if op == "⌈":
            return Number(value=max(w, x))
        if op == "|":
            return Number(value=x % w)
        if op == "<":
            return Number(value=1 if w < x else 0)
        if op == "≤":
            return Number(value=1 if w <= x else 0)
        if op == ">":
            return Number(value=1 if w > x else 0)
        if op == "≥":
            return Number(value=1 if w >= x else 0)
        if op == "=":
            return Number(value=1 if w == x else 0)
        if op == "≠":
            return Number(value=1 if w != x else 0)
    except Exception:
        return None
    return None


def _is_fold_identity_literal(op: str, expr: Expr) -> bool:
    if not isinstance(expr, Number):
        return False
    value = expr.value
    if op in {"+", "∨"}:
        return value == 0
    if op in {"×", "∧"}:
        return value == 1
    return False


def _optimize_for_ir(expr: Expr) -> Expr:
    """Lightweight algebraic simplifications for IR lowering."""
    if isinstance(expr, Vector):
        return Vector(items=tuple(_optimize_for_ir(item) for item in expr.items))

    if isinstance(expr, Prefix):
        right = _optimize_for_ir(expr.right)
        if isinstance(right, Number):
            folded = _fold_prefix_literal(expr.op, right)
            if folded is not None:
                return folded
        return Prefix(op=expr.op, right=right)

    if isinstance(expr, Infix):
        left = _optimize_for_ir(expr.left)
        right = _optimize_for_ir(expr.right)

        if expr.op.startswith("´"):
            fold_op = expr.op[1:]
            if fold_op in _SUPPORTED_FOLD and _is_fold_identity_literal(fold_op, left):
                # Canonicalize identity-seeded fold-init to plain fold.
                return Call(
                    func=Mod1(op="´", operand=Name(value=fold_op)),
                    right=right,
                )
            return Infix(op=expr.op, left=left, right=right)

        if isinstance(left, Number) and isinstance(right, Number):
            folded = _fold_infix_literals(expr.op, left, right)
            if folded is not None:
                return folded

        if expr.op == "+":
            if _is_number_expr(left, 0):
                return right
            if _is_number_expr(right, 0):
                return left
        if expr.op == "-":
            if _is_number_expr(right, 0):
                return left
        if expr.op == "×":
            if _is_number_expr(left, 0) or _is_number_expr(right, 0):
                return Number(value=0)
            if _is_number_expr(left, 1):
                return right
            if _is_number_expr(right, 1):
                return left
        if expr.op == "÷":
            if _is_number_expr(right, 1):
                return left
        if expr.op == "⋆":
            if _is_number_expr(right, 1):
                return left
            if _is_number_expr(right, 0):
                return Number(value=1)
            if _is_number_expr(right, 2):
                return Infix(op="×", left=left, right=left)
            if _is_number_expr(left, 1):
                return Number(value=1)
        if expr.op == "⊣":
            return left
        if expr.op == "⊢":
            return right

        return Infix(op=expr.op, left=left, right=right)

    if isinstance(expr, Mod1):
        return Mod1(op=expr.op, operand=_optimize_for_ir(expr.operand))

    if isinstance(expr, Mod2):
        return Mod2(op=expr.op, left=_optimize_for_ir(expr.left), right=_optimize_for_ir(expr.right))

    if isinstance(expr, Train):
        return Train(parts=tuple(_optimize_for_ir(part) for part in expr.parts))

    if isinstance(expr, Call):
        return Call(
            func=_optimize_for_ir(expr.func),
            right=_optimize_for_ir(expr.right),
            left=None if expr.left is None else _optimize_for_ir(expr.left),
        )

    return expr


def _rewrite_apply_for_ir(func: Expr, right: Expr, *, left: Expr | None = None) -> Expr:
    """Rewrite an application node into lowerable numeric-expression forms."""

    if isinstance(func, Name):
        if func.value not in _PRIMITIVE_FUNCTION_NAMES:
            raise _unsupported_backend(f"function-valued name {func.value!r}")
        if left is None:
            return Prefix(op=func.value, right=right)
        return Infix(op=func.value, left=left, right=right)

    if isinstance(func, Train):
        if len(func.parts) == 2:
            left_func, right_func = func.parts
            if left is None:
                right_result = _rewrite_apply_for_ir(right_func, right)
            else:
                right_result = _rewrite_apply_for_ir(right_func, right, left=left)
            return _rewrite_apply_for_ir(left_func, right_result)
        if len(func.parts) == 3:
            left_func, center_func, right_func = func.parts
            if left is None:
                left_result = _rewrite_apply_for_ir(left_func, right)
                right_result = _rewrite_apply_for_ir(right_func, right)
            else:
                left_result = _rewrite_apply_for_ir(left_func, right, left=left)
                right_result = _rewrite_apply_for_ir(right_func, right, left=left)
            return _rewrite_apply_for_ir(center_func, right_result, left=left_result)
        raise _unsupported_backend("train lowering beyond 2- and 3-train forms")

    if isinstance(func, Mod2):
        left_func = func.left
        right_func = func.right

        if func.op == "∘":
            composed = _rewrite_apply_for_ir(right_func, right) if left is None else _rewrite_apply_for_ir(
                right_func, right, left=left
            )
            return _rewrite_apply_for_ir(left_func, composed)

        if func.op == "○":
            if left is None:
                composed = _rewrite_apply_for_ir(right_func, right)
                return _rewrite_apply_for_ir(left_func, composed)
            left_transformed = _rewrite_apply_for_ir(right_func, left)
            right_transformed = _rewrite_apply_for_ir(right_func, right)
            return _rewrite_apply_for_ir(left_func, right_transformed, left=left_transformed)

        if func.op == "⊸":
            before = _rewrite_apply_for_ir(left_func, right) if left is None else _rewrite_apply_for_ir(
                left_func, left
            )
            return _rewrite_apply_for_ir(right_func, right, left=before)

        if func.op == "⟜":
            after = _rewrite_apply_for_ir(right_func, right)
            if left is None:
                return _rewrite_apply_for_ir(left_func, after, left=right)
            return _rewrite_apply_for_ir(left_func, after, left=left)

        if func.op == "⊘":
            if left is None:
                return _rewrite_apply_for_ir(left_func, right)
            return _rewrite_apply_for_ir(right_func, right, left=left)

        raise _unsupported_backend(f"2-modifier {func.op!r}")

    if isinstance(func, Mod1):
        if func.op == "˜":
            if left is None:
                return _rewrite_apply_for_ir(func.operand, right, left=right)
            return _rewrite_apply_for_ir(func.operand, left, left=right)

        if func.op == "˙":
            return func.operand

        if func.op == "´":
            # Preserve fold call form recognized by lowering.
            return Call(func=func, right=right, left=left)

        raise _unsupported_backend(f"1-modifier {func.op!r}")

    # Leave unhandled calls for the lowerer to report with existing errors.
    return Call(func=func, right=right, left=left)


@dataclass(frozen=True)
class IRNode:
    """Single SSA-like IR node for compiled numeric subset execution."""

    id: int
    op: str
    inputs: tuple[int, ...] = ()
    value: object | None = None
    name: str | None = None


@dataclass(frozen=True)
class JaxIR:
    """Lowered IR container."""

    nodes: tuple[IRNode, ...]
    output: int
    arg_names: tuple[str, ...]


@dataclass(frozen=True)
class ShapePolicy:
    """Explicit shape policy for transform wrappers.

    - `dynamic`: no call-to-call shape lock.
    - `static`: first call locks argument shapes and future calls must match.
    """

    kind: Literal["dynamic", "static"] = "dynamic"
    static_argnums: tuple[int, ...] = ()


class _Lowerer:
    def __init__(self, *, arg_names: tuple[str, ...], constants: dict[str, object]) -> None:
        self.arg_names = arg_names
        self.arg_set = set(arg_names)
        self.constants = constants
        self.nodes: list[IRNode] = []
        self._arg_nodes: dict[str, int] = {}
        self._expr_cache: dict[Expr, int] = {}

    def _add(self, op: str, *, inputs: tuple[int, ...] = (), value: object | None = None, name: str | None = None) -> int:
        node_id = len(self.nodes)
        self.nodes.append(IRNode(id=node_id, op=op, inputs=inputs, value=value, name=name))
        return node_id

    def _arg(self, name: str) -> int:
        if name in self._arg_nodes:
            return self._arg_nodes[name]
        idx = self._add("arg", name=name)
        self._arg_nodes[name] = idx
        return idx

    def _cache_expr(self, expr: Expr, node_id: int) -> int:
        self._expr_cache[expr] = node_id
        return node_id

    def lower_expr(self, expr: Expr) -> int:
        if expr in self._expr_cache:
            return self._expr_cache[expr]

        if isinstance(expr, Number):
            return self._cache_expr(expr, self._add("const", value=expr.value))

        if isinstance(expr, Char):
            return self._cache_expr(expr, self._add("const", value=ord(expr.value)))

        if isinstance(expr, String):
            return self._cache_expr(expr, self._add("const", value=[ord(ch) for ch in expr.value]))

        if isinstance(expr, Null):
            return self._cache_expr(expr, self._add("const", value=0))

        if isinstance(expr, Name):
            if expr.value in self.arg_set:
                return self._cache_expr(expr, self._arg(expr.value))
            if expr.value in self.constants:
                return self._cache_expr(expr, self._add("const", value=self.constants[expr.value]))
            if expr.value == "i":
                return self._cache_expr(expr, self._add("const", value=1j))
            if expr.value in _SYSTEM_CONSTANTS:
                return self._cache_expr(expr, self._add("const", value=_SYSTEM_CONSTANTS[expr.value]))
            if expr.value in _PRIMITIVE_FUNCTION_NAMES:
                raise _unsupported_backend(f"function-valued name {expr.value!r}")
            if expr.value.startswith("•"):
                raise _unsupported_runtime_only(f"system value/function {expr.value!r}")
            raise _unsupported_backend(f"name {expr.value!r}")

        if isinstance(expr, Member):
            raise _unsupported_runtime_only("namespace member access")

        if isinstance(expr, Vector):
            item_ids = tuple(self.lower_expr(item) for item in expr.items)
            return self._cache_expr(expr, self._add("vector", inputs=item_ids))

        if isinstance(expr, Prefix):
            if expr.op not in _SUPPORTED_PREFIX:
                raise _unsupported_backend(f"monadic primitive {expr.op!r}")
            right_id = self.lower_expr(expr.right)
            return self._cache_expr(expr, self._add(f"prefix:{expr.op}", inputs=(right_id,)))

        if isinstance(expr, Infix):
            if expr.op.startswith("´"):
                fold_op = expr.op[1:]
                if fold_op not in _SUPPORTED_FOLD:
                    raise _unsupported_backend(f"fold primitive {fold_op!r}")
                left_id = self.lower_expr(expr.left)
                right_id = self.lower_expr(expr.right)
                return self._cache_expr(expr, self._add(f"fold_init:{fold_op}", inputs=(left_id, right_id)))

            if expr.op not in _SUPPORTED_INFIX:
                raise _unsupported_backend(f"dyadic primitive {expr.op!r}")
            left_id = self.lower_expr(expr.left)
            right_id = self.lower_expr(expr.right)
            return self._cache_expr(expr, self._add(f"infix:{expr.op}", inputs=(left_id, right_id)))

        if isinstance(expr, Call):
            if isinstance(expr.func, Block):
                raise _unsupported_runtime_only("block call")
            if isinstance(expr.func, Member):
                raise _unsupported_runtime_only("namespace member call")
            if isinstance(expr.func, Name) and expr.func.value.startswith("•"):
                raise _unsupported_runtime_only(f"system call {expr.func.value!r}")
            if isinstance(expr.func, Mod1) and expr.func.op == "´" and isinstance(expr.func.operand, Name):
                fold_op = expr.func.operand.value
                if fold_op not in _SUPPORTED_FOLD:
                    raise _unsupported_backend(f"fold primitive {fold_op!r}")
                right_id = self.lower_expr(expr.right)
                if expr.left is None:
                    return self._cache_expr(expr, self._add(f"fold:{fold_op}", inputs=(right_id,)))
                left_id = self.lower_expr(expr.left)
                return self._cache_expr(expr, self._add(f"fold_init:{fold_op}", inputs=(left_id, right_id)))

            raise _unsupported_backend("call forms beyond fold-call lowering")

        if isinstance(expr, Nothing):
            raise _unsupported_runtime_only("nothing literal ·")
        if isinstance(expr, (Assign, Export, Block)):
            raise _unsupported_runtime_only(type(expr).__name__)
        if isinstance(expr, (Mod1, Mod2, Train)):
            raise _unsupported_backend(f"node type {type(expr).__name__}")

        raise _unsupported_backend(f"node type {type(expr).__name__}")


def _as_array(value):
    if isinstance(value, jnp.ndarray):
        return value
    return jnp.asarray(value)


def _as_python_int_scalar(value, *, where: str) -> int:
    arr = _as_array(value)
    if arr.ndim != 0:
        raise BQNTypeError(f"{where} requires a scalar integer argument")
    scalar = arr.item()
    if isinstance(scalar, numbers.Integral):
        return int(scalar)
    if isinstance(scalar, numbers.Real):
        real = float(scalar)
        if real.is_integer():
            return int(real)
    raise BQNTypeError(f"{where} requires an integer argument")


def _as_integer_array(value, *, where: str):
    arr = _as_array(value)
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return arr
    raise BQNTypeError(f"{where} requires integer arguments")


def _shape_from_left(value) -> tuple[int, ...]:
    arr = _as_array(value)
    if arr.ndim == 0:
        dims = [arr.item()]
    elif arr.ndim == 1:
        dims = [arr[i].item() for i in range(arr.shape[0])]
    else:
        raise BQNShapeError("⥊ left argument must be a scalar or rank-1 shape vector")

    shape: list[int] = []
    for dim in dims:
        if isinstance(dim, numbers.Integral):
            out = int(dim)
        elif isinstance(dim, numbers.Real):
            real = float(dim)
            if not real.is_integer():
                raise BQNTypeError("⥊ dimensions must be integers")
            out = int(real)
        else:
            raise BQNTypeError("⥊ dimensions must be integers")
        if out < 0:
            raise BQNShapeError("⥊ dimensions must be non-negative")
        shape.append(out)
    return tuple(shape)


def _fill_cell_for_take(arr):
    if arr.ndim == 0:
        return jnp.asarray(0, dtype=arr.dtype)
    return jnp.zeros(tuple(int(d) for d in arr.shape[1:]), dtype=arr.dtype)


def _array_fill_block(fill_cell, count: int):
    fill = _as_array(fill_cell)
    if count <= 0:
        return jnp.broadcast_to(fill, (0, *fill.shape))
    return jnp.broadcast_to(fill, (count, *fill.shape))


def _concat(left, right):
    left_arr = _as_array(left)
    right_arr = _as_array(right)
    if left_arr.ndim == 0:
        left_arr = jnp.reshape(left_arr, (1,))
    if right_arr.ndim == 0:
        right_arr = jnp.reshape(right_arr, (1,))
    if left_arr.ndim != right_arr.ndim:
        raise BQNShapeError("∾ requires matching ranks after scalar promotion")
    return jnp.concatenate((left_arr, right_arr), axis=0)


def _take(count: int, value):
    n = abs(count)
    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    fill_cell = _fill_cell_for_take(arr)

    if count >= 0:
        taken = arr[:n]
        missing = n - int(taken.shape[0])
        if missing <= 0:
            return taken
        fill = _array_fill_block(fill_cell, missing)
        return jnp.concatenate((taken, fill), axis=0)

    taken = arr[-n:] if n > 0 else arr[:0]
    missing = n - int(taken.shape[0])
    if missing <= 0:
        return taken
    fill = _array_fill_block(fill_cell, missing)
    return jnp.concatenate((fill, taken), axis=0)


def _drop(count: int, value):
    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    if count >= 0:
        return arr[count:]
    return arr[:count]


def _rotate(count: int, value):
    arr = _as_array(value)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 0:
        return arr
    n = count % int(arr.shape[0])
    return jnp.concatenate((arr[n:], arr[:n]), axis=0)


def _transpose(value):
    arr = _as_array(value)
    if arr.ndim <= 1:
        return arr
    return jnp.transpose(arr)


def _transpose_axes(left, right):
    arr = _as_array(right)
    if arr.ndim <= 1:
        return arr

    axes_arr = _as_array(left)
    if axes_arr.ndim == 0:
        axis = _as_python_int_scalar(axes_arr, where="⍉")
        if arr.ndim != 2:
            raise BQNShapeError("Scalar left argument for ⍉ requires rank-2 right argument")
        if axis == 0:
            return arr
        if axis in {1, -1}:
            return jnp.transpose(arr)
        raise BQNShapeError("Scalar left argument for ⍉ must be 0 or 1")

    if axes_arr.ndim != 1:
        raise BQNShapeError("⍉ left argument must be a scalar or rank-1 axis vector")
    axes = [_as_python_int_scalar(axes_arr[i], where="⍉") for i in range(axes_arr.shape[0])]
    if len(axes) != arr.ndim:
        raise BQNShapeError("⍉ axis vector length must match right rank")

    norm_axes: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise BQNShapeError("⍉ axis out of bounds")
        norm_axes.append(axis)
    if len(set(norm_axes)) != len(norm_axes):
        raise BQNShapeError("⍉ axis vector must be a permutation")
    return jnp.transpose(arr, axes=tuple(norm_axes))


def _rank_of(value) -> int:
    return _as_array(value).ndim


def _shape_of(value) -> tuple[int, ...]:
    return tuple(int(d) for d in _as_array(value).shape)


def _depth_of(value) -> int:
    arr = _as_array(value)
    return 0 if arr.ndim == 0 else 1


def _length_of(value) -> int:
    shape = _shape_of(value)
    if not shape:
        return 1
    return int(shape[0])


def _pack_vector(items: list[object]):
    arrays = [_as_array(item) for item in items]
    first_shape = arrays[0].shape
    if not all(arr.shape == first_shape for arr in arrays):
        raise BQNTypeError("Numeric IR vector packing requires homogeneous element shapes")
    return jnp.stack(arrays, axis=0)


def _promote_binary_pair(w: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if w.dtype == x.dtype:
        return w, x
    dtype = jnp.result_type(w, x)
    if w.dtype == dtype:
        return w, lax.convert_element_type(x, dtype)
    if x.dtype == dtype:
        return lax.convert_element_type(w, dtype), x
    return lax.convert_element_type(w, dtype), lax.convert_element_type(x, dtype)


def _unary_reciprocal_array(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / x


def _unary_sign_array(x: jnp.ndarray) -> jnp.ndarray:
    return lax.sign(x)


def _unary_floor_array(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.floor(x)


def _unary_ceil_array(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.ceil(x)


def _unary_abs_array(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.abs(x)


def _infix_add_eval_array(w, x):
    if w.dtype == x.dtype and jnp.issubdtype(w.dtype, jnp.floating):
        return w + x
    ww, xx = _promote_binary_pair(w, x)
    return lax.add(ww, xx)


def _infix_sub_eval_array(w, x):
    ww, xx = _promote_binary_pair(w, x)
    return lax.sub(ww, xx)


def _infix_mul_eval_array(w, x):
    ww, xx = _promote_binary_pair(w, x)
    return lax.mul(ww, xx)


def _infix_cmp_eval_array(cmp_op, w, x):
    if w.dtype == x.dtype:
        return lax.convert_element_type(cmp_op(w, x), jnp.int32)
    ww, xx = _promote_binary_pair(w, x)
    return lax.convert_element_type(cmp_op(ww, xx), jnp.int32)


def _static_integer_scalar_or_none(value) -> int | None:
    arr = _as_array(value)
    if arr.ndim != 0:
        return None
    try:
        scalar = arr.item()
    except Exception:
        return None
    if isinstance(scalar, numbers.Integral):
        return int(scalar)
    if isinstance(scalar, numbers.Real):
        real = float(scalar)
        if real.is_integer():
            return int(real)
    return None


def _nonnegative_int_power_array(base: jnp.ndarray, exponent: int) -> jnp.ndarray:
    if exponent == 0:
        return jnp.ones_like(base)
    if exponent == 1:
        return base
    if exponent == 2:
        return jnp.square(base)

    result = jnp.ones_like(base)
    factor = base
    power = exponent
    while power > 0:
        if power & 1:
            result = result * factor
        power >>= 1
        if power:
            factor = factor * factor
    return result


def _infix_power_eval_array(w, x):
    if not _USE_INT_POWER_FAST_PATH:
        return w**x
    exponent = _static_integer_scalar_or_none(x)
    if exponent in {0, 1, 2}:
        return _nonnegative_int_power_array(w, exponent)
    return w**x


def _prefix_rank_eval_array(x):
    return jnp.asarray(_rank_of(x), dtype=jnp.int32)


def _prefix_length_eval_array(x):
    return jnp.asarray(_length_of(x), dtype=jnp.int32)


def _prefix_shape_eval_array(x):
    return jnp.asarray(_shape_of(x), dtype=jnp.int32)


def _prefix_depth_eval_array(x):
    return jnp.asarray(_depth_of(x), dtype=jnp.int32)


def _prefix_range_eval_array(x):
    n = _as_python_int_scalar(x, where="↕")
    if n < 0:
        raise BQNShapeError("↕ requires a non-negative integer")
    return jnp.arange(n)


def _prefix_reverse_eval_array(x):
    if x.ndim == 0:
        return x
    return jnp.flip(x, axis=0)


_PREFIX_ARRAY_DISPATCH: dict[str, callable] = {
    "+": jnp.conjugate,
    "-": lambda x: -x,
    "×": _unary_sign_array,
    "÷": _unary_reciprocal_array,
    "¬": lambda x: 1 - x,
    "⌊": _unary_floor_array,
    "⌈": _unary_ceil_array,
    "|": _unary_abs_array,
    "√": jnp.sqrt,
    "=": _prefix_rank_eval_array,
    "≠": _prefix_length_eval_array,
    "≢": _prefix_shape_eval_array,
    "≡": _prefix_depth_eval_array,
    "↕": _prefix_range_eval_array,
    "⥊": jnp.ravel,
    "⌽": _prefix_reverse_eval_array,
    "⍉": _transpose,
    "⊣": lambda x: x,
    "⊢": lambda x: x,
}


def _prefix_eval_array(op: str, x):
    fn = _PREFIX_ARRAY_DISPATCH.get(op)
    if fn is None:
        raise _unsupported_backend(f"prefix op {op!r}")
    return fn(x)


def _prefix_eval(op: str, right):
    return _prefix_eval_array(op, _as_array(right))


def _infix_div_eval_array(w, x):
    return w / x


def _infix_root_eval_array(w, x):
    return x ** (1 / w)


def _infix_floor_eval_array(w, x):
    return jnp.minimum(w, x)


def _infix_ceil_eval_array(w, x):
    return jnp.maximum(w, x)


def _infix_mod_eval_array(w, x):
    return jnp.mod(x, w)


def _infix_lt_eval_array(w, x):
    return _infix_cmp_eval_array(lax.lt, w, x)


def _infix_le_eval_array(w, x):
    return _infix_cmp_eval_array(lax.le, w, x)


def _infix_gt_eval_array(w, x):
    return _infix_cmp_eval_array(lax.gt, w, x)


def _infix_ge_eval_array(w, x):
    return _infix_cmp_eval_array(lax.ge, w, x)


def _infix_eq_eval_array(w, x):
    return _infix_cmp_eval_array(lax.eq, w, x)


def _infix_ne_eval_array(w, x):
    return _infix_cmp_eval_array(lax.ne, w, x)


def _infix_lcm_eval_array(w, x):
    return jnp.lcm(_as_integer_array(w, where="∧"), _as_integer_array(x, where="∧"))


def _infix_gcd_eval_array(w, x):
    return jnp.gcd(_as_integer_array(w, where="∨"), _as_integer_array(x, where="∨"))


def _infix_reshape_eval_array(w, x):
    return jnp.reshape(x, _shape_from_left(w))


def _infix_left_eval_array(w, x):
    del x
    return w


def _infix_right_eval_array(w, x):
    del w
    return x


def _infix_take_eval_array(w, x):
    count = _as_python_int_scalar(w, where="↑")
    return _take(count, x)


def _infix_drop_eval_array(w, x):
    count = _as_python_int_scalar(w, where="↓")
    return _drop(count, x)


def _infix_rotate_eval_array(w, x):
    count = _as_python_int_scalar(w, where="⌽")
    return _rotate(count, x)


_INFIX_ARRAY_DISPATCH: dict[str, callable] = {
    "+": _infix_add_eval_array,
    "-": _infix_sub_eval_array,
    "×": _infix_mul_eval_array,
    "÷": _infix_div_eval_array,
    "⋆": _infix_power_eval_array,
    "√": _infix_root_eval_array,
    "⌊": _infix_floor_eval_array,
    "⌈": _infix_ceil_eval_array,
    "|": _infix_mod_eval_array,
    "<": _infix_lt_eval_array,
    "≤": _infix_le_eval_array,
    ">": _infix_gt_eval_array,
    "≥": _infix_ge_eval_array,
    "=": _infix_eq_eval_array,
    "≠": _infix_ne_eval_array,
    "∧": _infix_lcm_eval_array,
    "∨": _infix_gcd_eval_array,
    "⥊": _infix_reshape_eval_array,
    "⊣": _infix_left_eval_array,
    "⊢": _infix_right_eval_array,
    "∾": _concat,
    "↑": _infix_take_eval_array,
    "↓": _infix_drop_eval_array,
    "⌽": _infix_rotate_eval_array,
    "⍉": _transpose_axes,
}


def _infix_eval_array(op: str, w, x):
    fn = _INFIX_ARRAY_DISPATCH.get(op)
    if fn is None:
        raise _unsupported_backend(f"infix op {op!r}")
    return fn(w, x)


def _infix_eval(op: str, left, right):
    return _infix_eval_array(op, _as_array(left), _as_array(right))


def _fold_identity(op: str):
    if op == "+":
        return jnp.asarray(0)
    if op == "×":
        return jnp.asarray(1)
    if op == "∧":
        return jnp.asarray(1, dtype=jnp.int32)
    if op == "∨":
        return jnp.asarray(0, dtype=jnp.int32)
    if op == "⌊":
        return jnp.asarray(jnp.inf)
    if op == "⌈":
        return jnp.asarray(-jnp.inf)
    return _MISSING


def _fold_eval(op: str, value, *, init=_MISSING):
    arr = _as_array(value)
    if arr.ndim == 0:
        raise BQNShapeError("Fold input must have rank at least 1")

    length = int(arr.shape[0])
    if length == 0:
        if init is not _MISSING:
            return init
        identity = _fold_identity(op)
        if identity is _MISSING:
            raise BQNShapeError(f"Fold of empty array has no known identity for op {op!r}")
        return identity

    if _USE_PRIMITIVE_FOLD_FAST_PATH:
        # Fast paths for associative primitive folds: map to vectorized reducers.
        reduced = _MISSING
        if op == "+":
            reduced = jnp.sum(arr, axis=0, dtype=arr.dtype)
        elif op == "×":
            reduced = jnp.prod(arr, axis=0, dtype=arr.dtype)
        elif op == "⌊":
            reduced = jnp.min(arr, axis=0)
        elif op == "⌈":
            reduced = jnp.max(arr, axis=0)
        elif op == "∧":
            ints = _as_integer_array(arr, where="∧")
            reduced = lax.associative_scan(jnp.lcm, ints, axis=0)[-1]
        elif op == "∨":
            ints = _as_integer_array(arr, where="∨")
            reduced = lax.associative_scan(jnp.gcd, ints, axis=0)[-1]
        if reduced is not _MISSING:
            if init is _MISSING:
                return reduced
            return _infix_eval(op, init, reduced)

    if init is _MISSING:
        acc = arr[length - 1]
    else:
        acc = _infix_eval(op, init, arr[length - 1])
    for i in range(length - 2, -1, -1):
        acc = _infix_eval(op, arr[i], acc)
    return acc


@lru_cache(maxsize=1024)
def _decode_node_op(op: str) -> tuple[str, str]:
    head, sep, tail = op.partition(":")
    if sep:
        return head, tail
    return op, ""


def _drop_ir_const_cache(cache_key: int) -> None:
    _IR_CONST_VALUE_CACHE.pop(cache_key, None)
    _IR_CONST_VALUE_CACHE_REFS.pop(cache_key, None)


def _const_values_for_ir(ir: JaxIR) -> tuple[object | None, ...]:
    cache_key = id(ir)
    ref = _IR_CONST_VALUE_CACHE_REFS.get(cache_key)
    if ref is not None and ref() is ir:
        cached = _IR_CONST_VALUE_CACHE.get(cache_key)
        if cached is not None:
            return cached
    elif ref is not None:
        _drop_ir_const_cache(cache_key)

    const_values: list[object | None] = [None] * len(ir.nodes)
    for node in ir.nodes:
        if node.op != "const":
            continue
        value = node.value
        const_values[node.id] = value if isinstance(value, jnp.ndarray) else _as_array(value)
    frozen = tuple(const_values)
    _IR_CONST_VALUE_CACHE[cache_key] = frozen
    _IR_CONST_VALUE_CACHE_REFS[cache_key] = weakref.ref(ir, lambda _ref, k=cache_key: _drop_ir_const_cache(k))
    return frozen


def evaluate_ir(ir: JaxIR, args: tuple[object, ...]) -> object:
    """Execute lowered IR in pure JAX operations."""
    if len(args) != len(ir.arg_names):
        raise BQNTypeError(f"Expected {len(ir.arg_names)} arguments, got {len(args)}")

    arg_values = {name: arg for name, arg in zip(ir.arg_names, args, strict=True)}
    values: list[object] = [None] * len(ir.nodes)
    const_values = _const_values_for_ir(ir)

    for node in ir.nodes:
        op = node.op
        if op == "arg":
            if node.name is None:
                raise BQNRuntimeError("IR arg node is missing a name")
            values[node.id] = _as_array(arg_values[node.name])
            continue

        if op == "const":
            values[node.id] = const_values[node.id]
            continue

        if op == "vector":
            items = [values[idx] for idx in node.inputs]
            if not items:
                values[node.id] = jnp.asarray([])
            else:
                values[node.id] = _pack_vector(items)
            continue

        kind, payload = _decode_node_op(op)
        if kind == "prefix":
            values[node.id] = _prefix_eval_array(payload, values[node.inputs[0]])
            continue

        if kind == "infix":
            values[node.id] = _infix_eval_array(payload, values[node.inputs[0]], values[node.inputs[1]])
            continue

        if kind == "fold_init":
            values[node.id] = _fold_eval(payload, values[node.inputs[1]], init=values[node.inputs[0]])
            continue

        if kind == "fold":
            values[node.id] = _fold_eval(payload, values[node.inputs[0]])
            continue

        raise BQNRuntimeError(f"Unknown IR op {op!r}")

    out = values[ir.output]
    if out is None:
        raise BQNRuntimeError("IR output node was not produced")
    return out


@dataclass
class CompiledExpression:
    """Callable wrapper around lowered IR with optional JAX transforms."""

    ir: JaxIR
    source: str | None = None
    shape_policy: ShapePolicy = field(default_factory=ShapePolicy)
    _static_signature: tuple[tuple[int, ...], ...] | None = None
    _static_arg_ids: tuple[int, ...] | None = None
    _jit_cache: dict[tuple[int, ...], object] = field(default_factory=dict, init=False, repr=False)
    _jit_shape_cache: dict[tuple[tuple[int, ...], ...], object] = field(default_factory=dict, init=False, repr=False)
    _last_jit_signature: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False)
    _last_jit_arg_ids: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _last_jit_fn: object | None = field(default=None, init=False, repr=False)
    _call_ir: object = field(default=None, init=False, repr=False)
    _grad_cache: dict[object, object] = field(default_factory=dict, init=False, repr=False)
    _vmap_cache: dict[tuple[str, str], object] = field(default_factory=dict, init=False, repr=False)
    _transform_stats: dict[str, int] = field(
        default_factory=lambda: {"jit_hits": 0, "jit_misses": 0, "grad_hits": 0, "grad_misses": 0, "vmap_hits": 0, "vmap_misses": 0},
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        ir = self.ir

        def _call_ir(*args):
            return evaluate_ir(ir, args)

        self._call_ir = _call_ir

    def _resolve_call_args(self, *args, **kwargs) -> tuple[object, ...]:
        if args and kwargs:
            raise BQNTypeError("Use either positional or keyword arguments, not both")

        if kwargs:
            missing = [name for name in self.ir.arg_names if name not in kwargs]
            extra = [name for name in kwargs if name not in self.ir.arg_names]
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if extra:
                    details.append(f"extra={extra}")
                raise BQNTypeError(f"Keyword arguments do not match IR signature ({', '.join(details)})")
            return tuple(kwargs[name] for name in self.ir.arg_names)

        if len(args) != len(self.ir.arg_names):
            raise BQNTypeError(f"Expected {len(self.ir.arg_names)} arguments, got {len(args)}")
        return args

    def _shape_signature(self, values: tuple[object, ...]) -> tuple[tuple[int, ...], ...]:
        signature: list[tuple[int, ...]] = []
        for value in values:
            quick = self._value_shape_fast(value)
            if quick is not None:
                signature.append(quick)
                continue
            arr = _as_array(value)
            signature.append(tuple(int(d) for d in arr.shape))
        return tuple(signature)

    @staticmethod
    def _value_shape_fast(value: object) -> tuple[int, ...] | None:
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return tuple(int(d) for d in shape)
            except Exception:  # pragma: no cover - defensive
                pass
        if isinstance(value, numbers.Number):
            return ()
        return None

    @staticmethod
    def _contains_tracer(values: tuple[object, ...]) -> bool:
        tracer_type = jax.core.Tracer
        for value in values:
            if isinstance(value, tracer_type):
                return True
        return False

    def _shape_signature_matches(self, values: tuple[object, ...], expected: tuple[tuple[int, ...], ...]) -> bool:
        if len(values) != len(expected):
            return False
        for value, exp_shape in zip(values, expected, strict=True):
            quick = self._value_shape_fast(value)
            if quick is None:
                return False
            if quick != exp_shape:
                return False
        return True

    @staticmethod
    def _arg_ids(values: tuple[object, ...]) -> tuple[int, ...]:
        n = len(values)
        if n == 1:
            return (id(values[0]),)
        if n == 2:
            return (id(values[0]), id(values[1]))
        if n == 3:
            return (id(values[0]), id(values[1]), id(values[2]))
        return tuple(id(value) for value in values)

    @staticmethod
    def _values_match_arg_ids(values: tuple[object, ...], cached_ids: tuple[int, ...] | None) -> bool:
        if cached_ids is None:
            return False
        n = len(values)
        if len(cached_ids) != n:
            return False
        if n == 1:
            return cached_ids[0] == id(values[0])
        if n == 2:
            return cached_ids[0] == id(values[0]) and cached_ids[1] == id(values[1])
        if n == 3:
            return cached_ids[0] == id(values[0]) and cached_ids[1] == id(values[1]) and cached_ids[2] == id(values[2])
        for idx, value in enumerate(values):
            if cached_ids[idx] != id(value):
                return False
        return True

    def _arg_ids_match(self, values: tuple[object, ...]) -> bool:
        if not self._values_match_arg_ids(values, self._static_arg_ids):
            return False
        if self.shape_policy.kind == "static" and self._static_signature is not None:
            # Guard against Python object-id reuse by requiring that the fast
            # shape signature still matches the static contract.
            return self._shape_signature_matches(values, self._static_signature)
        return True

    def _enforce_shape_policy(self, values: tuple[object, ...]) -> None:
        if self.shape_policy.kind != "static":
            return
        if self._static_signature is None:
            arg_ids = self._arg_ids(values)
            self._static_signature = self._shape_signature(values)
            self._static_arg_ids = arg_ids
            return
        if self._arg_ids_match(values):
            return
        arg_ids = self._arg_ids(values)
        if self._shape_signature_matches(values, self._static_signature):
            self._static_arg_ids = arg_ids
            return
        signature = self._shape_signature(values)
        if self._static_signature != signature:
            raise BQNShapeError(
                f"Static shape policy violation: expected {self._static_signature}, got {signature}"
            )
        self._static_arg_ids = arg_ids

    def _call_positional(self, *args):
        return self._call_ir(*args)

    def _call_wrapped(self, fn, values: tuple[object, ...]):
        try:
            return fn(*values)
        except BQNError:
            raise
        except Exception as err:  # pragma: no cover - defensive
            raise classify_runtime_exception(err) from err

    def _call_entry(self, fn, args: tuple[object, ...], kwargs: dict[str, object]):
        if not kwargs and len(args) == len(self.ir.arg_names):
            if self.shape_policy.kind != "static" or not self._contains_tracer(args):
                self._enforce_shape_policy(args)
            try:
                return fn(*args)
            except BQNError:
                raise
            except Exception as err:  # pragma: no cover - defensive
                raise classify_runtime_exception(err) from err
        values = self._resolve_call_args(*args, **kwargs)
        if self.shape_policy.kind != "static" or not self._contains_tracer(values):
            self._enforce_shape_policy(values)
        return self._call_wrapped(fn, values)

    def _grad_kernel_cache_key(self, *, argnums: int) -> tuple[object, ...]:
        node_key = tuple(
            (node.op, node.inputs, node.name, _freeze_for_cache(node.value))
            for node in self.ir.nodes
        )
        return (
            "grad_kernel",
            self.ir.arg_names,
            self.ir.output,
            node_key,
            argnums,
            tuple(self.shape_policy.static_argnums),
        )

    def __call__(self, *args, **kwargs):
        return self._call_entry(self._call_ir, args, kwargs)

    def trace(self, *args, **kwargs):
        """Emit jaxpr for the current expression under sample inputs."""
        values = self._resolve_call_args(*args, **kwargs)
        return jax.make_jaxpr(self._call_ir)(*values)

    def transform_cache_stats(self) -> dict[str, float | int]:
        stats = dict(self._transform_stats)
        hits = stats["jit_hits"] + stats["grad_hits"] + stats["vmap_hits"]
        misses = stats["jit_misses"] + stats["grad_misses"] + stats["vmap_misses"]
        total = hits + misses
        stats["total_hits"] = hits
        stats["total_misses"] = misses
        stats["hit_rate"] = float(hits / total) if total else 0.0
        return stats

    def jit(self):
        """Return a JIT-compiled callable honoring this expression's shape policy."""
        key = tuple(self.shape_policy.static_argnums)
        cached = self._jit_cache.get(key)
        if cached is not None:
            self._transform_stats["jit_hits"] += 1
            return cached

        self._transform_stats["jit_misses"] += 1
        arg_count = len(self.ir.arg_names)
        static_policy = self.shape_policy.kind == "static"

        if static_policy:
            # Static policy guarantees a single shape contract, so keep a single
            # direct jitted callable in the hot path to minimize Python overhead.
            jitted = jax.jit(self._call_ir, static_argnums=self.shape_policy.static_argnums)

            def wrapped(*args, **kwargs):
                if not kwargs and len(args) == arg_count:
                    if self._static_signature is None:
                        self._enforce_shape_policy(args)
                    elif not self._arg_ids_match(args):
                        self._enforce_shape_policy(args)
                    return jitted(*args)
                return self._call_entry(jitted, args, kwargs)

            self._jit_cache[key] = wrapped
            return wrapped

        def _shape_specialized(values: tuple[object, ...]):
            if self._last_jit_fn is not None and self._values_match_arg_ids(values, self._last_jit_arg_ids):
                return self._last_jit_fn

            signature = self._shape_signature(values)
            if self._last_jit_signature == signature and self._last_jit_fn is not None:
                self._last_jit_arg_ids = self._arg_ids(values)
                return self._last_jit_fn

            cached_shape_fn = self._jit_shape_cache.get(signature)
            if cached_shape_fn is None:
                cached_shape_fn = jax.jit(self._call_ir, static_argnums=self.shape_policy.static_argnums)
                self._jit_shape_cache[signature] = cached_shape_fn
            self._last_jit_signature = signature
            self._last_jit_arg_ids = self._arg_ids(values)
            self._last_jit_fn = cached_shape_fn
            return cached_shape_fn

        def wrapped(*args, **kwargs):
            if not kwargs and len(args) == arg_count:
                if not static_policy:
                    return _shape_specialized(args)(*args)
                if self._static_signature is None:
                    self._enforce_shape_policy(args)
                    return _shape_specialized(args)(*args)
                if self._arg_ids_match(args):
                    shape_fn = self._last_jit_fn
                    if shape_fn is None:
                        shape_fn = _shape_specialized(args)
                    return shape_fn(*args)
                self._enforce_shape_policy(args)
                return _shape_specialized(args)(*args)
            jitted = _shape_specialized(self._resolve_call_args(*args, **kwargs))
            return self._call_entry(jitted, args, kwargs)

        self._jit_cache[key] = wrapped
        return wrapped

    def grad(self, *, argnums: int = 0):
        """Return a gradient function for scalar-output expressions."""
        cached = self._grad_cache.get(argnums)
        if cached is not None:
            self._transform_stats["grad_hits"] += 1
            return cached

        self._transform_stats["grad_misses"] += 1
        kernel_key = self._grad_kernel_cache_key(argnums=argnums)
        grad_fn = _GRAD_KERNEL_CACHE.get(kernel_key)
        if grad_fn is None:
            grad_core = jax.grad(self._call_ir, argnums=argnums)
            grad_fn = jax.jit(grad_core, static_argnums=self.shape_policy.static_argnums)
            _GRAD_KERNEL_CACHE[kernel_key] = grad_fn
        arg_count = len(self.ir.arg_names)
        static_policy = self.shape_policy.kind == "static"

        def wrapped(*args, **kwargs):
            if not kwargs and len(args) == arg_count:
                if static_policy and not self._contains_tracer(args):
                    self._enforce_shape_policy(args)
                try:
                    return grad_fn(*args)
                except BQNError:
                    raise
                except Exception as err:  # pragma: no cover - defensive
                    raise classify_runtime_exception(err) from err
            return self._call_entry(grad_fn, args, kwargs)

        self._grad_cache[argnums] = wrapped
        return wrapped

    def vmap(self, *, in_axes=0, out_axes=0):
        """Return a vectorized callable over the compiled expression."""
        key = (repr(in_axes), repr(out_axes))
        cached = self._vmap_cache.get(key)
        if cached is not None:
            self._transform_stats["vmap_hits"] += 1
            return cached

        self._transform_stats["vmap_misses"] += 1
        vmapped = jax.vmap(self._call_ir, in_axes=in_axes, out_axes=out_axes)

        def wrapped(*args, **kwargs):
            return self._call_entry(vmapped, args, kwargs)

        self._vmap_cache[key] = wrapped
        return wrapped


def _freeze_for_cache(value: object) -> object:
    if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
        return value
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_for_cache(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze_for_cache(item) for item in value))
    if isinstance(value, Mapping):
        items = tuple(sorted((str(k), _freeze_for_cache(v)) for k, v in value.items()))
        return ("mapping", items)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(int(d) for d in getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", "unknown"))
        ptr: int | None = None
        interface = getattr(value, "__array_interface__", None)
        if isinstance(interface, Mapping):
            data = interface.get("data")
            if isinstance(data, tuple) and data:
                try:
                    ptr = int(data[0])
                except Exception:
                    ptr = None
        if ptr is None:
            ptr = id(value)
        return ("array", shape, dtype, ptr)
    return (type(value).__name__, repr(value))


def _constants_cache_key(constants: Mapping[str, object]) -> tuple[object, ...]:
    if not constants:
        return ()
    return tuple((name, _freeze_for_cache(constants[name])) for name in sorted(constants))


def _persistent_ir_cache_file(cache_key: tuple[object, ...]) -> Path:
    key_material = repr(
        (
            _PERSISTENT_IR_CACHE_VERSION,
            cache_key,
            getattr(jax, "__version__", "unknown"),
            platform.python_version(),
            platform.platform(),
            sys.implementation.name,
        )
    ).encode("utf-8")
    digest = hashlib.sha256(key_material).hexdigest()
    return _PERSISTENT_IR_CACHE_DIR / f"{digest}.pkl"


def _load_persistent_ir(cache_key: tuple[object, ...]) -> JaxIR | None:
    if not _USE_PERSISTENT_IR_CACHE:
        return None
    path = _persistent_ir_cache_file(cache_key)
    if not path.exists():
        _COMPILED_IR_CACHE_STATS["disk_misses"] += 1
        return None
    try:
        loaded = pickle.loads(path.read_bytes())
    except Exception:
        _COMPILED_IR_CACHE_STATS["disk_misses"] += 1
        return None
    if isinstance(loaded, JaxIR):
        _COMPILED_IR_CACHE_STATS["disk_hits"] += 1
        return loaded
    _COMPILED_IR_CACHE_STATS["disk_misses"] += 1
    return None


def _store_persistent_ir(cache_key: tuple[object, ...], ir: JaxIR) -> None:
    if not _USE_PERSISTENT_IR_CACHE:
        return
    path = _persistent_ir_cache_file(cache_key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(ir, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        # Persistent cache is opportunistic; failures should not affect correctness.
        return


def compile_cache_stats(*, reset: bool = False) -> dict[str, float | int]:
    hits = _COMPILED_IR_CACHE_STATS["hits"]
    misses = _COMPILED_IR_CACHE_STATS["misses"]
    disk_hits = _COMPILED_IR_CACHE_STATS["disk_hits"]
    disk_misses = _COMPILED_IR_CACHE_STATS["disk_misses"]
    total = hits + misses
    stats: dict[str, float | int] = {
        "hits": hits,
        "misses": misses,
        "disk_hits": disk_hits,
        "disk_misses": disk_misses,
        "size": len(_COMPILED_IR_CACHE),
        "persistent_cache_enabled": _USE_PERSISTENT_IR_CACHE,
        "persistent_cache_dir": str(_PERSISTENT_IR_CACHE_DIR),
        "hit_rate": float(hits / total) if total else 0.0,
    }
    if reset:
        _COMPILED_IR_CACHE.clear()
        _COMPILED_IR_CACHE_STATS["hits"] = 0
        _COMPILED_IR_CACHE_STATS["misses"] = 0
        _COMPILED_IR_CACHE_STATS["disk_hits"] = 0
        _COMPILED_IR_CACHE_STATS["disk_misses"] = 0
    return stats


def prelower_cache_stats(*, reset: bool = False) -> dict[str, float | int]:
    hits = _PRELOWER_CACHE_STATS["hits"]
    misses = _PRELOWER_CACHE_STATS["misses"]
    total = hits + misses
    info = _prepare_expr_cached.cache_info()
    stats: dict[str, float | int] = {
        "hits": hits,
        "misses": misses,
        "size": int(info.currsize),
        "max_size": int(info.maxsize) if info.maxsize is not None else 0,
        "hit_rate": float(hits / total) if total else 0.0,
    }
    if reset:
        _prepare_expr_cached.cache_clear()
        _PRELOWER_CACHE_STATS["hits"] = 0
        _PRELOWER_CACHE_STATS["misses"] = 0
    return stats


def transform_helper_cache_stats(*, reset: bool = False) -> dict[str, float | int]:
    hits = _TRANSFORM_HELPER_STATS["hits"]
    misses = _TRANSFORM_HELPER_STATS["misses"]
    total = hits + misses
    stats: dict[str, float | int] = {
        "hits": hits,
        "misses": misses,
        "size": len(_TRANSFORM_HELPER_CACHE),
        "hit_rate": float(hits / total) if total else 0.0,
    }
    if reset:
        _TRANSFORM_HELPER_CACHE.clear()
        _TRANSFORM_HELPER_STATS["hits"] = 0
        _TRANSFORM_HELPER_STATS["misses"] = 0
    return stats


def lower_to_ir(
    source: str,
    *,
    arg_names: tuple[str, ...] = (),
    constants: dict[str, object] | None = None,
    optimize: bool = True,
    use_cache: bool = True,
) -> JaxIR:
    """Lower a numeric-expression subset from AST to JAX-friendly IR."""
    consts = constants if constants is not None else {}
    arg_names = tuple(arg_names)
    cache_key = (
        source,
        arg_names,
        _constants_cache_key(consts),
        bool(optimize),
    )
    if use_cache:
        cached = _COMPILED_IR_CACHE.get(cache_key)
        if cached is not None:
            _COMPILED_IR_CACHE_STATS["hits"] += 1
            return cached
        _COMPILED_IR_CACHE_STATS["misses"] += 1
        cached_disk = _load_persistent_ir(cache_key)
        if cached_disk is not None:
            _COMPILED_IR_CACHE_STATS["hits"] += 1
            _COMPILED_IR_CACHE[cache_key] = cached_disk
            return cached_disk

    try:
        rewrite_aliases = _rewrite_aliases(consts)
        fingerprint = _prelower_fingerprint(source, arg_names, rewrite_aliases, bool(optimize))
        info_before = _prepare_expr_cached.cache_info()
        expr = _prepare_expr_cached(fingerprint, source, arg_names, rewrite_aliases, bool(optimize))
        info_after = _prepare_expr_cached.cache_info()
        if info_after.hits > info_before.hits:
            _PRELOWER_CACHE_STATS["hits"] += 1
        else:
            _PRELOWER_CACHE_STATS["misses"] += 1
    except ParseError as err:
        raise BQNParseError.from_parse_error(err) from err

    lowerer = _Lowerer(arg_names=arg_names, constants=consts)
    out = lowerer.lower_expr(expr)
    ir = JaxIR(nodes=tuple(lowerer.nodes), output=out, arg_names=tuple(arg_names))
    if use_cache:
        _COMPILED_IR_CACHE[cache_key] = ir
        _store_persistent_ir(cache_key, ir)
    return ir


def compile_expression(
    source: str,
    *,
    arg_names: tuple[str, ...] = (),
    constants: dict[str, object] | None = None,
    shape_policy: ShapePolicy | None = None,
    optimize: bool = True,
    use_cache: bool = True,
    trace: bool = False,
    trace_args: tuple[object, ...] | None = None,
):
    """Compile a numeric subset expression to IR-backed callable.

    When `trace=True`, returns `(compiled, jaxpr)` and requires `trace_args`.
    """
    ir = lower_to_ir(
        source,
        arg_names=arg_names,
        constants=constants,
        optimize=optimize,
        use_cache=use_cache,
    )
    compiled = CompiledExpression(ir=ir, source=source, shape_policy=shape_policy or ShapePolicy())
    if not trace:
        return compiled
    if trace_args is None:
        raise BQNTypeError("trace=True requires trace_args")
    return compiled, compiled.trace(*trace_args)


def cached_jit(
    source: str,
    *,
    arg_names: tuple[str, ...] = (),
    constants: dict[str, object] | None = None,
    shape_policy: ShapePolicy | None = None,
    optimize: bool = True,
    use_cache: bool = True,
):
    """Return a cached JIT callable keyed by expression and shape policy."""
    policy = shape_policy or ShapePolicy()
    key = (
        "jit",
        source,
        tuple(arg_names),
        _constants_cache_key(constants or {}),
        policy.kind,
        tuple(policy.static_argnums),
        bool(optimize),
    )
    cached = _TRANSFORM_HELPER_CACHE.get(key)
    if cached is not None:
        _TRANSFORM_HELPER_STATS["hits"] += 1
        return cached
    _TRANSFORM_HELPER_STATS["misses"] += 1
    fn = compile_expression(
        source,
        arg_names=arg_names,
        constants=constants,
        shape_policy=policy,
        optimize=optimize,
        use_cache=use_cache,
    ).jit()
    _TRANSFORM_HELPER_CACHE[key] = fn
    return fn


def cached_vmap(
    source: str,
    *,
    arg_names: tuple[str, ...] = (),
    constants: dict[str, object] | None = None,
    shape_policy: ShapePolicy | None = None,
    in_axes=0,
    out_axes=0,
    optimize: bool = True,
    use_cache: bool = True,
):
    """Return a cached VMAP callable keyed by expression and axes."""
    policy = shape_policy or ShapePolicy()
    key = (
        "vmap",
        source,
        tuple(arg_names),
        _constants_cache_key(constants or {}),
        policy.kind,
        tuple(policy.static_argnums),
        repr(in_axes),
        repr(out_axes),
        bool(optimize),
    )
    cached = _TRANSFORM_HELPER_CACHE.get(key)
    if cached is not None:
        _TRANSFORM_HELPER_STATS["hits"] += 1
        return cached
    _TRANSFORM_HELPER_STATS["misses"] += 1
    fn = compile_expression(
        source,
        arg_names=arg_names,
        constants=constants,
        shape_policy=policy,
        optimize=optimize,
        use_cache=use_cache,
    ).vmap(in_axes=in_axes, out_axes=out_axes)
    _TRANSFORM_HELPER_CACHE[key] = fn
    return fn


def cached_grad(
    source: str,
    *,
    arg_names: tuple[str, ...] = (),
    constants: dict[str, object] | None = None,
    shape_policy: ShapePolicy | None = None,
    argnums: int = 0,
    optimize: bool = True,
    use_cache: bool = True,
):
    """Return a cached GRAD callable keyed by expression and argnums."""
    policy = shape_policy or ShapePolicy()
    key = (
        "grad",
        source,
        tuple(arg_names),
        _constants_cache_key(constants or {}),
        policy.kind,
        tuple(policy.static_argnums),
        argnums,
        bool(optimize),
    )
    cached = _TRANSFORM_HELPER_CACHE.get(key)
    if cached is not None:
        _TRANSFORM_HELPER_STATS["hits"] += 1
        return cached
    _TRANSFORM_HELPER_STATS["misses"] += 1
    fn = compile_expression(
        source,
        arg_names=arg_names,
        constants=constants,
        shape_policy=policy,
        optimize=optimize,
        use_cache=use_cache,
    ).grad(argnums=argnums)
    _TRANSFORM_HELPER_CACHE[key] = fn
    return fn


def evaluate_with_errors(source: str, *, env: dict[str, object] | None = None):
    """Evaluate through the classic interpreter with structured error classes."""
    from .evaluator import evaluate as _evaluate

    try:
        return _evaluate(source, env=env)
    except ParseError as err:
        raise BQNParseError.from_parse_error(err) from err
    except BQNError:
        raise
    except Exception as err:
        raise classify_runtime_exception(err) from err
