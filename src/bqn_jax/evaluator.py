"""Evaluator for an expanded (still partial) BQN subset on top of JAX."""

from __future__ import annotations

import numbers
import os
import time
import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, ClassVar, Final, overload

import jax
from jax import lax
import jax.numpy as jnp
import jax.scipy.special as jsp

from .ast import Assign, Block, Call, Case, Char, Export, Expr, Infix, Member, Mod1, Mod2, Name, Nothing, Null, Number, Prefix, Program, String, Train, Vector
from .parser import parse_program
from .values import BQNChar, BoxedArray, OperationInfo, as_jax_array, depth_of as value_depth_of, is_operation, rank_of as value_rank_of, shape_of as value_shape_of, validate_value as validate_bqn_value, zero_like as value_zero_like


@dataclass(frozen=True)
class Namespace:
    values: dict[str, object]
    _bqn_namespace: ClassVar[bool] = True


_SPECIAL_NAME_ALIASES: Final[dict[str, str]] = {
    "𝕨": "𝕨",
    "𝕎": "𝕨",
    "𝕩": "𝕩",
    "𝕏": "𝕩",
    "𝕗": "𝕗",
    "𝔽": "𝕗",
    "𝕘": "𝕘",
    "𝔾": "𝕘",
    "𝕤": "𝕤",
    "𝕊": "𝕤",
    "𝕣": "𝕣",
    "_𝕣": "𝕣",
    "_𝕣_": "𝕣",
}

_NO_FAST_PATH: Final = object()
_USE_JITTED_BASE_OPS: Final[bool] = os.environ.get("BQN_JAX_DISABLE_JITTED_BASE_OPS", "0") != "1"
_USE_INT_POWER_FAST_PATH: Final[bool] = os.environ.get("BQN_JAX_DISABLE_INT_POWER_FAST_PATH", "0") != "1"
_USE_PRIMITIVE_FOLD_FAST_PATH: Final[bool] = os.environ.get("BQN_JAX_DISABLE_PRIMITIVE_FOLD_FAST_PATH", "0") != "1"
_PROGRAM_CACHE_MAX: Final[int] = max(1, int(os.environ.get("BQN_JAX_PROGRAM_CACHE_MAX", "256")))


@lru_cache(maxsize=_PROGRAM_CACHE_MAX)
def _parse_program_cached(source: str) -> Program:
    return parse_program(source)


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


_BASE_UNARY_OPS: Final[dict[str, Callable[[jnp.ndarray], jnp.ndarray]]] = {
    "+": lambda x: jnp.conjugate(x),
    "-": lambda x: -x,
    "×": _unary_sign_array,
    "÷": _unary_reciprocal_array,
    "⋆": lambda x: jnp.exp(x),
    "¬": lambda x: 1 - x,
    "⌊": _unary_floor_array,
    "⌈": _unary_ceil_array,
    "|": _unary_abs_array,
    "√": lambda x: jnp.sqrt(x),
}


def _promote_binary_pair(w: jnp.ndarray, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if w.dtype == x.dtype:
        return w, x
    dtype = jnp.result_type(w, x)
    if w.dtype == dtype:
        return w, lax.convert_element_type(x, dtype)
    if x.dtype == dtype:
        return lax.convert_element_type(w, dtype), x
    return lax.convert_element_type(w, dtype), lax.convert_element_type(x, dtype)


def _lax_add_promoted(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    if w.dtype == x.dtype and jnp.issubdtype(w.dtype, jnp.floating):
        return w + x
    ww, xx = _promote_binary_pair(w, x)
    return lax.add(ww, xx)


def _lax_sub_promoted(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    ww, xx = _promote_binary_pair(w, x)
    return lax.sub(ww, xx)


def _lax_mul_promoted(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    ww, xx = _promote_binary_pair(w, x)
    return lax.mul(ww, xx)


def _lax_cmp_promoted(cmp_op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    if w.dtype == x.dtype:
        return lax.convert_element_type(cmp_op(w, x), jnp.int32)
    ww, xx = _promote_binary_pair(w, x)
    return lax.convert_element_type(cmp_op(ww, xx), jnp.int32)


_BASE_BINARY_OPS: Final[dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]]] = {
    "+": _lax_add_promoted,
    "-": _lax_sub_promoted,
    "×": _lax_mul_promoted,
    "÷": lambda w, x: w / x,
    "⋆": lambda w, x: w**x,
    "⌊": lambda w, x: jnp.minimum(w, x),
    "⌈": lambda w, x: jnp.maximum(w, x),
    "|": lambda w, x: jnp.mod(x, w),
    "√": lambda w, x: x ** (1 / w),
    "=": lambda w, x: _lax_cmp_promoted(lax.eq, w, x),
    "≠": lambda w, x: _lax_cmp_promoted(lax.ne, w, x),
    "<": lambda w, x: _lax_cmp_promoted(lax.lt, w, x),
    "≤": lambda w, x: _lax_cmp_promoted(lax.le, w, x),
    ">": lambda w, x: _lax_cmp_promoted(lax.gt, w, x),
    "≥": lambda w, x: _lax_cmp_promoted(lax.ge, w, x),
    "∧": lambda w, x: jnp.lcm(w, x),
    "∨": lambda w, x: jnp.gcd(w, x),
}

_BASE_UNARY_FALLBACK: Final[dict[str, Callable[[object], object]]] = {
    "+": lambda v: jnp.conjugate(_as_array(v)),
    "-": lambda v: -_as_array(v),
    "×": lambda v: _unary_sign_array(_as_array(v)),
    "÷": lambda v: _unary_reciprocal_array(_as_array(v)),
    "⋆": lambda v: jnp.exp(_as_array(v)),
    "¬": lambda v: 1 - _as_array(v),
    "⌊": lambda v: _unary_floor_array(_as_array(v)),
    "⌈": lambda v: _unary_ceil_array(_as_array(v)),
    "|": lambda v: _unary_abs_array(_as_array(v)),
    "√": lambda v: jnp.sqrt(_as_array(v)),
}

_BASE_BINARY_FALLBACK: Final[dict[str, Callable[[object, object], object]]] = {
    "+": lambda l, r: _add_or_sub_charlike("+", l, r),
    "-": lambda l, r: _add_or_sub_charlike("-", l, r),
    "×": lambda l, r: _as_array(l) * _as_array(r),
    "÷": lambda l, r: _as_array(l) / _as_array(r),
    "⋆": lambda l, r: _as_array(l) ** _as_array(r),
    "⌊": lambda l, r: jnp.minimum(_as_array(l), _as_array(r)),
    "⌈": lambda l, r: jnp.maximum(_as_array(l), _as_array(r)),
    "|": lambda l, r: jnp.mod(_as_array(r), _as_array(l)),
    "√": lambda l, r: _as_array(r) ** (1 / _as_array(l)),
    "=": lambda l, r: _numeric_bool_result(_as_array(l) == _as_array(r)),
    "≠": lambda l, r: _numeric_bool_result(_as_array(l) != _as_array(r)),
    "<": lambda l, r: _cmp_lt(l, r),
    "≤": lambda l, r: _cmp_leq(l, r),
    ">": lambda l, r: _cmp_gt(l, r),
    "≥": lambda l, r: _cmp_geq(l, r),
    "∧": lambda l, r: jnp.lcm(_as_integer_array(l, where="∧"), _as_integer_array(r, where="∧")),
    "∨": lambda l, r: jnp.gcd(_as_integer_array(l, where="∨"), _as_integer_array(r, where="∨")),
}

_JITTED_UNARY_OPS: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {}
_JITTED_BINARY_OPS: dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {}
_JITTED_NONNEG_INT_POWER_OPS: dict[int, Callable[[jnp.ndarray], jnp.ndarray]] = {}


def _jitted_unary_kernel(op: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    fn = _JITTED_UNARY_OPS.get(op)
    if fn is None:
        fn = jax.jit(_BASE_UNARY_OPS[op])
        _JITTED_UNARY_OPS[op] = fn
    return fn


def _jitted_binary_kernel(op: str) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    fn = _JITTED_BINARY_OPS.get(op)
    if fn is None:
        fn = jax.jit(_BASE_BINARY_OPS[op])
        _JITTED_BINARY_OPS[op] = fn
    return fn


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


def _jitted_nonnegative_int_power_kernel(exponent: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    fn = _JITTED_NONNEG_INT_POWER_OPS.get(exponent)
    if fn is None:
        def kernel(value: jnp.ndarray) -> jnp.ndarray:
            return _nonnegative_int_power_array(value, exponent)

        fn = jax.jit(kernel)
        _JITTED_NONNEG_INT_POWER_OPS[exponent] = fn
    return fn


def _canonical_special_name(name: str) -> str | None:
    return _SPECIAL_NAME_ALIASES.get(name)


def _normalize_identifier_name(name: str) -> str:
    special = _canonical_special_name(name)
    if special is not None:
        return special
    if name.startswith("•"):
        return name
    return name.replace("_", "").casefold()


class Scope(MutableMapping[str, object]):
    def __init__(self, data: MutableMapping[str, object] | None = None, parent: "Scope | None" = None, *, allow_redefinition: bool = False) -> None:
        self.data: dict[str, object] = {}
        self.parent = parent
        self._raw_by_norm: dict[str, str] = {}
        self._display_by_norm: dict[str, str] = {}
        self.definitions: set[str] = set()
        self.exports: set[str] = set()
        self.namespace_requested: bool = False
        self.allow_redefinition: bool = allow_redefinition

        if data is not None:
            for key, value in data.items():
                norm = _normalize_identifier_name(key)
                if norm in self._raw_by_norm:
                    raise NameError(f"Duplicate initial binding for name {key!r}")
                validate_bqn_value(value, where=f"env[{key!r}]")
                self.data[key] = value
                self._raw_by_norm[norm] = key
                self._display_by_norm[norm] = key
                self.definitions.add(norm)

    def _local_raw(self, key: str) -> str | None:
        return self._raw_by_norm.get(_normalize_identifier_name(key))

    def _has_local_norm(self, norm: str) -> bool:
        return norm in self._raw_by_norm

    def _find_scope_by_norm(self, norm: str) -> "Scope | None":
        if self._has_local_norm(norm):
            return self
        if self.parent is not None:
            return self.parent._find_scope_by_norm(norm)
        return None

    def __getitem__(self, key: str) -> object:
        raw = self._local_raw(key)
        if raw is not None:
            return self.data[raw]
        if self.parent is not None:
            return self.parent[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: object) -> None:
        validate_bqn_value(value, where=f"name {key!r}")
        norm = _normalize_identifier_name(key)
        raw = self._raw_by_norm.get(norm, key)
        self.data[raw] = value
        self._raw_by_norm[norm] = raw
        self._display_by_norm.setdefault(norm, raw)

    def __delitem__(self, key: str) -> None:
        norm = _normalize_identifier_name(key)
        raw = self._raw_by_norm.get(norm)
        if raw is not None:
            del self.data[raw]
            del self._raw_by_norm[norm]
            self._display_by_norm.pop(norm, None)
            self.definitions.discard(norm)
            self.exports.discard(norm)
            return
        raise KeyError(key)

    def __iter__(self):
        seen: set[str] = set()
        current: Scope | None = self
        while current is not None:
            for norm, raw in current._raw_by_norm.items():
                if norm not in seen:
                    seen.add(norm)
                    yield current._display_by_norm.get(norm, raw)
            current = current.parent

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        norm = _normalize_identifier_name(key)
        if self._has_local_norm(norm):
            return True
        if self.parent is not None:
            return norm in self.parent
        return False

    def find_scope(self, key: str) -> "Scope | None":
        return self._find_scope_by_norm(_normalize_identifier_name(key))

    def set_existing(self, key: str, value: object) -> None:
        norm = _normalize_identifier_name(key)
        scope = self._find_scope_by_norm(norm)
        if scope is None:
            raise NameError(f"Cannot update undefined name {key!r}")
        validate_bqn_value(value, where=f"name {key!r}")
        raw = scope._raw_by_norm[norm]
        scope.data[raw] = value

    def define(self, key: str, value: object) -> None:
        norm = _normalize_identifier_name(key)
        if norm in self.definitions and not self.allow_redefinition:
            raise NameError(f"Duplicate definition for name {key!r} in the same scope")
        self[key] = value
        self.definitions.add(norm)

    def is_locally_defined(self, key: str) -> bool:
        return _normalize_identifier_name(key) in self.definitions

    def mark_export(self, key: str) -> None:
        norm = _normalize_identifier_name(key)
        if norm not in self.definitions:
            raise NameError(f"Cannot export non-local name {key!r}")
        self.exports.add(norm)
        self.namespace_requested = True

    def request_namespace_result(self) -> None:
        self.namespace_requested = True

    def exported_values(self) -> dict[str, object]:
        ordered = sorted(self.exports, key=lambda norm: self._display_by_norm.get(norm, norm))
        out: dict[str, object] = {}
        for norm in ordered:
            raw = self._raw_by_norm.get(norm)
            if raw is None:
                continue
            display = self._display_by_norm.get(norm, raw)
            out[display] = self.data[raw]
        return out


@dataclass
class EvaluationEnvironment(MutableMapping[str, object]):
    """Persistent evaluation environment for stateful evaluate() calls."""

    _scope: Scope

    def __init__(self, data: MutableMapping[str, object] | None = None) -> None:
        seed_env = {} if data is None else dict(data)
        for name, value in seed_env.items():
            validate_bqn_value(value, where=f"env[{name!r}]")
        self._scope = Scope(data=seed_env, allow_redefinition=True)

    def __getitem__(self, key: str) -> object:
        return self._scope[key]

    def __setitem__(self, key: str, value: object) -> None:
        validate_bqn_value(value, where=f"env[{key!r}]")
        self._scope[key] = value
        self._scope.definitions.add(_normalize_identifier_name(key))

    def __delitem__(self, key: str) -> None:
        del self._scope[key]

    def __iter__(self):
        return iter(self._scope)

    def __len__(self) -> int:
        return len(self._scope)

    @property
    def values(self) -> dict[str, object]:
        return {name: self._scope[name] for name in self._scope}


@dataclass
class StatefulEvaluate:
    """Callable wrapper that evaluates source in a persistent environment."""

    env: EvaluationEnvironment

    def __call__(self, source: str):
        result, _ = evaluate(source, self.env)
        return result


@dataclass(frozen=True)
class UserFunction:
    cases: tuple[Case, ...]
    closure: Scope
    _bqn_operation_kind: ClassVar[str] = "block"

    @property
    def info(self) -> OperationInfo:
        return OperationInfo(kind="block", symbol=None, valence="ambivalent")


@dataclass(frozen=True)
class DerivedMod1:
    op: str
    operand: object
    _bqn_operation_kind: ClassVar[str] = "modifier1"

    @property
    def info(self) -> OperationInfo:
        return OperationInfo(kind="modifier1", symbol=self.op, valence="ambivalent")


@dataclass(frozen=True)
class DerivedMod2:
    op: str
    left: object
    right: object
    _bqn_operation_kind: ClassVar[str] = "modifier2"

    @property
    def info(self) -> OperationInfo:
        return OperationInfo(kind="modifier2", symbol=self.op, valence="ambivalent")


@dataclass(frozen=True)
class DerivedTrain:
    parts: tuple[object, ...]
    _bqn_operation_kind: ClassVar[str] = "train"

    @property
    def info(self) -> OperationInfo:
        return OperationInfo(kind="train", symbol=None, valence="ambivalent")


@dataclass(frozen=True)
class PrimitiveFunction:
    op: str
    _bqn_operation_kind: ClassVar[str] = "primitive"

    @property
    def info(self) -> OperationInfo:
        return OperationInfo(kind="primitive", symbol=self.op, valence="ambivalent")


_MISSING: Final = object()

_PRIMITIVE_FUNCTION_GLYPHS: Final[set[str]] = {
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
    "«",
    "»",
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
    "⋈",
    "/",
    "¬",
    "∧",
    "∨",
    "⍋",
    "⍒",
    "∊",
    "⍷",
}

_SYSTEM_POLICY: Final = (
    "System values are read-only and side-effect free in this runtime scope; "
    "no file/network/process mutation is exposed through • names."
)


def _string_to_codepoints(text: str):
    return jnp.asarray([ord(ch) for ch in text], dtype=jnp.int32)


def _format_scalar_for_system_text(value) -> str:
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        real = float(value)
        if math.isnan(real):
            return "NaN"
        if math.isinf(real):
            return "∞" if real > 0 else "¯∞"
        if real.is_integer():
            return str(int(real))
        return repr(real)
    if isinstance(value, numbers.Complex):
        comp = complex(value)
        real = _format_scalar_for_system_text(comp.real)
        imag_abs = _format_scalar_for_system_text(abs(comp.imag))
        if comp.imag >= 0:
            return f"{real}+{imag_abs}i"
        return f"{real}+¯{imag_abs}i"
    if isinstance(value, str):
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
    return str(value)


def _render_system_value_text(value, *, fmt: bool) -> str:
    if isinstance(value, BQNChar):
        return _format_scalar_for_system_text(value.value)

    if isinstance(value, Namespace):
        keys = sorted(value.values.keys())
        body = " ".join(keys)
        return f"•{{{body}}}"

    if is_operation(value):
        info = getattr(value, "info", None)
        symbol = getattr(info, "symbol", None) if info is not None else None
        if isinstance(symbol, str) and symbol:
            return f"•{symbol}"
        name = getattr(value, "__name__", "callable")
        return f"•{name}"

    if isinstance(value, list):
        items = [_render_system_value_text(item, fmt=fmt) for item in value]
        if fmt:
            return f"⟨ {' '.join(items)} ⟩"
        return f"⟨{','.join(items)}⟩"

    arr = _as_array(value)
    if arr.ndim == 0:
        return _format_scalar_for_system_text(arr.item())
    if arr.ndim == 1:
        items = [_render_system_value_text(arr[i], fmt=fmt) for i in range(arr.shape[0])]
        if fmt:
            return f"⟨ {' '.join(items)} ⟩"
        return "‿".join(items)

    shape = "‿".join(str(int(d)) for d in arr.shape)
    flat = arr.reshape((-1,))
    items = [_render_system_value_text(flat[i], fmt=fmt) for i in range(flat.shape[0])]
    return f"({shape}⥊{'‿'.join(items)})"


def _system_repr(value):
    return _string_to_codepoints(_render_system_value_text(value, fmt=False))


def _system_fmt(value):
    return _string_to_codepoints(_render_system_value_text(value, fmt=True))


def _coerce_system_text(value, *, where: str) -> str:
    if isinstance(value, BQNChar):
        return value.value

    if isinstance(value, list):
        codepoints = [_as_python_int_scalar(item, where=where) for item in value]
        return "".join(chr(cp) for cp in codepoints)

    arr = _as_array(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.ndim != 1:
        raise ValueError(f"{where} requires a scalar or rank-1 character-code array argument")

    codepoints = [_as_python_int_scalar(arr[i], where=where) for i in range(arr.shape[0])]
    return "".join(chr(cp) for cp in codepoints)


def _system_parse_float(value):
    raw = _coerce_system_text(value, where="•ParseFloat").strip()
    if not raw:
        raise ValueError("•ParseFloat malformed input")
    cleaned = raw.replace("¯", "-")
    if cleaned == "π":
        return jnp.asarray(jnp.pi)
    if cleaned in {"∞", "inf", "+inf"}:
        return jnp.asarray(jnp.inf)
    if cleaned in {"-∞", "-inf"}:
        return jnp.asarray(-jnp.inf)
    try:
        return jnp.asarray(float(cleaned))
    except Exception as err:  # pragma: no cover - defensive branch
        raise ValueError("•ParseFloat malformed input") from err


def _system_unix_time(_):
    return jnp.asarray(time.time())


def _system_mono_time(_):
    return jnp.asarray(time.monotonic())


def _system_delay(value):
    arr = _as_array(value)
    if arr.ndim != 0:
        raise ValueError("•Delay requires a scalar delay argument")
    seconds = float(arr.item())
    if seconds < 0:
        raise ValueError("•Delay requires a non-negative delay")
    # perf_counter is monotonic and typically provides higher resolution than
    # monotonic on Windows runners for short delays.
    start = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - start
    if elapsed == 0.0 and seconds > 0.0:
        # Defend against coarse timer quantization for very short sleeps.
        elapsed = seconds
    return jnp.asarray(elapsed)


def _system_type(value):
    if is_operation(value):
        return jnp.asarray(3, dtype=jnp.int32)
    if isinstance(value, Namespace):
        return jnp.asarray(4, dtype=jnp.int32)
    if isinstance(value, BQNChar):
        return jnp.asarray(1, dtype=jnp.int32)
    if isinstance(value, list):
        return jnp.asarray(0, dtype=jnp.int32)
    arr = _as_array(value)
    return jnp.asarray(1 if arr.ndim == 0 else 0, dtype=jnp.int32)


def _system_values() -> dict[str, object]:
    return {
        "pi": jnp.asarray(jnp.pi),
        "e": jnp.asarray(jnp.e),
        "inf": jnp.asarray(jnp.inf),
        "nan": jnp.asarray(jnp.nan),
        "i": jnp.asarray(1j),
        "true": jnp.asarray(1, dtype=jnp.int32),
        "false": jnp.asarray(0, dtype=jnp.int32),
        "cwd": _string_to_codepoints(os.getcwd()),
        "time": jnp.asarray(time.time()),
        "BQN": _string_to_codepoints("bgn-jax"),
        "version": _string_to_codepoints("bgn-jax"),
        "policy": _string_to_codepoints(_SYSTEM_POLICY),
        "Type": _system_type,
        "Repr": _system_repr,
        "Fmt": _system_fmt,
        "ParseFloat": _system_parse_float,
        "UnixTime": _system_unix_time,
        "MonoTime": _system_mono_time,
        "Delay": _system_delay,
        "type": _system_type,
        "repr": _system_repr,
        "fmt": _system_fmt,
        "parse_float": _system_parse_float,
        "unix_time": _system_unix_time,
        "mono_time": _system_mono_time,
        "delay": _system_delay,
    }


def _resolve_system_name(name: str):
    if name == "•":
        return Namespace(values=_system_values())

    key = name[1:]
    values = _system_values()
    if key in values:
        return values[key]
    raise NameError(f"Undefined system value {name!r}")


def _is_scalar_array(value: object) -> bool:
    return isinstance(value, jnp.ndarray) and value.ndim == 0


def _as_array(value):
    return as_jax_array(value)


def _as_python_int_scalar(value, *, where: str) -> int:
    arr = _as_array(value)
    if arr.ndim != 0:
        raise ValueError(f"{where} requires a scalar integer argument")
    scalar = arr.item()
    if isinstance(scalar, numbers.Integral):
        return int(scalar)
    if isinstance(scalar, numbers.Real):
        real = float(scalar)
        if real.is_integer():
            return int(real)
    raise ValueError(f"{where} requires an integer argument")


def _as_shape(value) -> tuple[int, ...]:
    if isinstance(value, list):
        dims = value
    else:
        arr = _as_array(value)
        if arr.ndim == 0:
            dims = [arr.item()]
        elif arr.ndim == 1:
            dims = [x.item() for x in arr]
        else:
            raise ValueError("Reshape left argument must be a scalar or rank-1 shape vector")

    shape: list[int] = []
    for dim in dims:
        if isinstance(dim, numbers.Integral):
            dim = int(dim)
        elif isinstance(dim, numbers.Real):
            real = float(dim)
            if not real.is_integer():
                raise ValueError("Reshape dimensions must be integers")
            dim = int(real)
        else:
            raise ValueError("Reshape dimensions must be integers")
        if dim < 0:
            raise ValueError("Reshape dimensions must be non-negative")
        shape.append(dim)
    return tuple(shape)


def _shape_of(value) -> tuple[int, ...]:
    return value_shape_of(value)


def _rank_of(value) -> int:
    return value_rank_of(value)


def _depth_of(value) -> int:
    return value_depth_of(value)


def _length_of(value) -> int:
    shape = _shape_of(value)
    if not shape:
        return 1
    return int(shape[0])


def _numeric_bool_result(value):
    return _as_array(value).astype(jnp.int32)


def _pack_vector(items) -> object:
    if not items:
        return jnp.asarray([])

    if all(isinstance(x, jnp.ndarray) for x in items):
        arrays = [x for x in items]
        first_shape = arrays[0].shape
        if all(arr.shape == first_shape for arr in arrays):
            return jnp.stack(arrays, axis=0)
    boxed = BoxedArray(items)
    validate_bqn_value(boxed, where="boxed vector")
    return boxed


def _pack_vector_literal(items) -> object:
    """Pack items from a BQN vector literal (⟨⟩ or ‿).

    Only stacks 0-d scalar arrays; higher-dimensional items stay as a
    BoxedArray so that pervasive binary ops get list-level agreement
    instead of broadcasting along a stacked matrix axis.
    """
    if not items:
        return jnp.asarray([])

    if all(isinstance(x, jnp.ndarray) for x in items):
        arrays = [x for x in items]
        first_shape = arrays[0].shape
        if first_shape == () and all(arr.shape == () for arr in arrays):
            return jnp.stack(arrays, axis=0)
    boxed = BoxedArray(items)
    validate_bqn_value(boxed, where="boxed vector")
    return boxed


def _map_unary(fn, value):
    if isinstance(value, list):
        # Preserve list (BoxedArray) structure -- BQN lists stay lists
        # through pervasive operations; they do not collapse into
        # higher-rank JAX arrays.
        return BoxedArray([_map_unary(fn, item) for item in value])
    return fn(value)


def _map_binary(fn, left, right):
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Mismatched boxed list lengths for dyadic operation")
        return BoxedArray([_map_binary(fn, l_item, r_item) for l_item, r_item in zip(left, right, strict=True)])
    if isinstance(left, list):
        return BoxedArray([_map_binary(fn, l_item, right) for l_item in left])
    if isinstance(right, list):
        return BoxedArray([_map_binary(fn, left, r_item) for r_item in right])
    return fn(left, right)


def _fast_unary_base(op: str, right):
    if not _USE_JITTED_BASE_OPS:
        return _NO_FAST_PATH
    if isinstance(right, list):
        return _NO_FAST_PATH
    kernel = _BASE_UNARY_OPS.get(op)
    if kernel is None:
        return _NO_FAST_PATH
    return _jitted_unary_kernel(op)(_as_array(right))


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


def _fast_integer_power(base: jnp.ndarray, exponent_value):
    if not _USE_INT_POWER_FAST_PATH:
        return _NO_FAST_PATH
    exponent = _static_integer_scalar_or_none(exponent_value)
    if exponent is None or exponent not in {0, 1, 2}:
        return _NO_FAST_PATH
    return _jitted_nonnegative_int_power_kernel(exponent)(base)


def _fast_binary_base(op: str, left, right):
    if not _USE_JITTED_BASE_OPS:
        return _NO_FAST_PATH
    if isinstance(left, list) or isinstance(right, list):
        return _NO_FAST_PATH
    kernel = _BASE_BINARY_OPS.get(op)
    if kernel is None:
        return _NO_FAST_PATH
    w = _as_array(left)
    x = _as_array(right)
    if op == "⋆":
        fast_pow = _fast_integer_power(w, x)
        if fast_pow is not _NO_FAST_PATH:
            return fast_pow
    if op in {"∧", "∨"}:
        w = _as_integer_array(w, where=op)
        x = _as_integer_array(x, where=op)
    return _jitted_binary_kernel(op)(w, x)


def _zero_like(value):
    return value_zero_like(value)


def _fill_cell_for_take(value):
    if isinstance(value, list):
        if not value:
            return jnp.asarray(0)
        return _zero_like(value[0])

    arr = _as_array(value)
    if arr.ndim == 0:
        return jnp.asarray(0, dtype=arr.dtype)
    return jnp.zeros(arr.shape[1:], dtype=arr.dtype)


def _array_fill_block(fill_cell, missing: int):
    fill_arr = _as_array(fill_cell)
    if fill_arr.ndim == 0:
        return jnp.full((missing,), fill_arr.item(), dtype=fill_arr.dtype)
    return jnp.broadcast_to(fill_arr, (missing, *fill_arr.shape))


def _is_char_code_scalar(value) -> bool:
    if isinstance(value, BQNChar):
        return True
    return _is_scalar_array(value) and jnp.issubdtype(value.dtype, jnp.integer)


def _char_code_to_scalar(value) -> int:
    if isinstance(value, BQNChar):
        return value.codepoint
    return int(_as_array(value).item())


def _add_or_sub_charlike(op: str, left, right):
    w_arr = _as_array(left)
    x_arr = _as_array(right)

    if op == "+":
        return w_arr + x_arr

    if op == "-":
        return w_arr - x_arr

    raise ValueError(f"Unsupported character-like arithmetic op {op!r}")


def _format_assert_message(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BQNChar):
        return value.value
    if isinstance(value, list):
        return str(value)

    arr = _as_array(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.ndim == 1 and jnp.issubdtype(arr.dtype, jnp.integer):
        codepoints = [int(x) for x in arr.tolist()]
        if all(0 <= cp <= 0x10FFFF for cp in codepoints):
            return "".join(chr(cp) for cp in codepoints)
    return str(arr.tolist())


def _assert_condition(value, *, message) -> object:
    if isinstance(value, list):
        raise ValueError(message)
    arr = _as_array(value)
    if arr.ndim != 0:
        raise ValueError(message)

    scalar = arr.item()
    if isinstance(scalar, numbers.Integral):
        ok = int(scalar) == 1
    elif isinstance(scalar, numbers.Real):
        ok = float(scalar) == 1.0
    else:
        ok = False

    if not ok:
        raise ValueError(message)
    return value


def _eval_prefix(op: str, right):
    if op.startswith("´"):
        return _fold(op[1:], right)

    if op == "≡":
        return jnp.asarray(_depth_of(right), dtype=jnp.int32)
    if op == "=":
        return jnp.asarray(_rank_of(right), dtype=jnp.int32)
    if op == "≠":
        return jnp.asarray(_length_of(right), dtype=jnp.int32)
    if op == "≢":
        return jnp.asarray(_shape_of(right), dtype=jnp.int32)
    if op == ">":
        if isinstance(right, list):
            merged: list[object] = []
            for item in right:
                if isinstance(item, list):
                    merged.extend(item)
                else:
                    merged.append(item)
            if merged:
                first_shape = _shape_of(merged[0])
                for item in merged[1:]:
                    if _shape_of(item) != first_shape:
                        raise ValueError("Monadic > requires compatible element shapes")
            return _pack_vector(merged)
        return right
    if op == "<":
        return _pack_vector([right])
    if op in {"⊣", "⊢"}:
        return right

    if op in _BASE_UNARY_FALLBACK:
        fast = _fast_unary_base(op, right)
        if fast is not _NO_FAST_PATH:
            return fast
        return _map_unary(_BASE_UNARY_FALLBACK[op], right)
    if op == "!":
        if isinstance(right, list):
            message = _format_assert_message(right)
        else:
            arr = _as_array(right)
            if arr.ndim == 0 and isinstance(arr.item(), numbers.Number):
                message = "Assertion error"
            else:
                message = _format_assert_message(right)
        return _assert_condition(right, message=message)
    if op == "↕":
        x = _as_array(right)
        n = _as_python_int_scalar(x, where="↕")
        if n < 0:
            raise ValueError("↕ requires a non-negative integer")
        return jnp.arange(n)
    if op == "⥊":
        x = _as_array(right)
        return jnp.ravel(x)
    if op == "⌽":
        return _reverse(right)
    if op == "»":
        return _nudge_after(right)
    if op == "«":
        return _nudge_before(right)
    if op == "⍉":
        return _transpose(right)
    if op == "≍":
        return _couple_monadic(right)
    if op == "⋈":
        return _couple_monadic(right)
    if op == "⊔":
        return _group_monadic(right)
    if op == "∧":
        return _sort_up(right)
    if op == "∨":
        return _sort_down(right)
    if op == "⍋":
        return _grade_up(right)
    if op == "⍒":
        return _grade_down(right)
    if op == "∊":
        return _mark_firsts(right)
    if op == "⊐":
        return _classify(right)
    if op == "⊒":
        return _occurrence_count(right)
    if op == "⍷":
        return _deduplicate(right)
    if op == "∾":
        return _join_monadic(right)
    if op == "⊑":
        # Monadic ⊑ — First.  Returns the first element.
        if isinstance(right, list):
            if not right:
                raise ValueError("⊑ on empty list")
            return right[0]
        arr = _as_array(right)
        if arr.ndim == 0:
            return arr
        if arr.shape[0] == 0:
            raise ValueError("⊑ on empty array")
        return arr[0]

    raise ValueError(f"Unsupported monadic primitive {op!r}")


def _match_values(left, right) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_match_values(l, r) for l, r in zip(left, right, strict=True))
    if isinstance(left, list) or isinstance(right, list):
        return False
    l_arr = _as_array(left)
    r_arr = _as_array(right)
    if l_arr.shape != r_arr.shape:
        return False
    return bool(jnp.all(l_arr == r_arr))


def _cmp_leq(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return jnp.asarray(1 if _sort_key(left) <= _sort_key(right) else 0, dtype=jnp.int32)

    # Character-like ordering approximation: integer scalars can compare.
    if _is_scalar_array(_as_array(left)) and _is_scalar_array(_as_array(right)):
        return _numeric_bool_result(_as_array(left) <= _as_array(right))
    return _numeric_bool_result(_as_array(left) <= _as_array(right))


def _cmp_lt(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return jnp.asarray(1 if _sort_key(left) < _sort_key(right) else 0, dtype=jnp.int32)
    return _numeric_bool_result(_as_array(left) < _as_array(right))


def _cmp_gt(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return jnp.asarray(1 if _sort_key(left) > _sort_key(right) else 0, dtype=jnp.int32)
    return _numeric_bool_result(_as_array(left) > _as_array(right))


def _cmp_geq(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return jnp.asarray(1 if _sort_key(left) >= _sort_key(right) else 0, dtype=jnp.int32)
    return _numeric_bool_result(_as_array(left) >= _as_array(right))


def _reverse(value):
    if isinstance(value, list):
        return _pack_vector(list(reversed(value)))
    arr = _as_array(value)
    if arr.ndim == 0:
        return arr
    return jnp.flip(arr, axis=0)


def _rotate(count: int, value):
    if isinstance(value, list):
        if not value:
            return value
        n = count % len(value)
        return _pack_vector([*value[n:], *value[:n]])

    arr = _as_array(value)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 0:
        return arr
    n = count % int(arr.shape[0])
    return jnp.concatenate((arr[n:], arr[:n]), axis=0)


def _transpose(value):
    if isinstance(value, list):
        return value
    arr = _as_array(value)
    if arr.ndim <= 1:
        return arr
    return jnp.transpose(arr)


def _transpose_axes(left, right):
    if isinstance(right, list):
        return right
    arr = _as_array(right)
    if arr.ndim <= 1:
        return arr

    axes_arr = _as_array(left)
    if axes_arr.ndim == 0:
        axis = _as_python_int_scalar(axes_arr, where="⍉")
        if arr.ndim != 2:
            raise ValueError("Scalar left argument for ⍉ currently requires rank-2 right argument")
        if axis == 0:
            return arr
        if axis in {1, -1}:
            return jnp.transpose(arr)
        raise ValueError("Scalar left argument for ⍉ must be 0 or 1")

    if axes_arr.ndim != 1:
        raise ValueError("⍉ left argument must be a scalar or rank-1 axis vector")
    axes = [_as_python_int_scalar(axes_arr[i], where="⍉") for i in range(axes_arr.shape[0])]
    if len(axes) != arr.ndim:
        raise ValueError("⍉ axis vector length must match right rank")

    norm_axes: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += arr.ndim
        if axis < 0 or axis >= arr.ndim:
            raise ValueError("⍉ axis out of bounds")
        norm_axes.append(axis)
    if len(set(norm_axes)) != len(norm_axes):
        raise ValueError("⍉ axis vector must be a permutation")
    return jnp.transpose(arr, axes=tuple(norm_axes))


def _binomial(left, right):
    w = _as_array(left)
    x = _as_array(right)
    raw = jnp.exp(jsp.gammaln(x + 1) - jsp.gammaln(w + 1) - jsp.gammaln(x - w + 1))
    valid = (w >= 0) & (w <= x)
    return jnp.where(valid, jnp.round(raw), 0.0)


def _sort_key(value):
    if isinstance(value, list):
        return ("list", len(value), tuple(_sort_key(item) for item in value))

    if isinstance(value, Namespace):
        return ("namespace", tuple(sorted(value.values.keys())))

    arr = _as_array(value)
    if arr.ndim == 0:
        return ("atom", arr.item())
    return ("array", tuple(int(d) for d in arr.shape), tuple(arr.ravel().tolist()))


def _array_row_lex_order(arr: jnp.ndarray, *, reverse: bool) -> jnp.ndarray:
    if arr.ndim == 1:
        order = jnp.argsort(arr, stable=True)
    else:
        flat = jnp.reshape(arr, (int(arr.shape[0]), -1))
        if int(flat.shape[1]) == 0:
            order = jnp.arange(int(flat.shape[0]), dtype=jnp.int32)
        else:
            order = jnp.lexsort(flat[:, ::-1].T)
    if reverse:
        order = jnp.flip(order, axis=0)
    return lax.convert_element_type(order, jnp.int32)


def _sort_up(value):
    if isinstance(value, list):
        return _pack_vector(sorted(value, key=_sort_key))

    arr = _as_array(value)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return jnp.sort(arr)
    try:
        return arr[_array_row_lex_order(arr, reverse=False)]
    except Exception:
        cells = [arr[i] for i in range(arr.shape[0])]
        return _pack_vector(sorted(cells, key=_sort_key))


def _sort_down(value):
    if isinstance(value, list):
        return _pack_vector(sorted(value, key=_sort_key, reverse=True))

    arr = _as_array(value)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1:
        return jnp.flip(jnp.sort(arr))
    try:
        return arr[_array_row_lex_order(arr, reverse=True)]
    except Exception:
        cells = [arr[i] for i in range(arr.shape[0])]
        return _pack_vector(sorted(cells, key=_sort_key, reverse=True))


def _grade(value, *, reverse: bool):
    if not isinstance(value, list):
        arr = _as_array(value)
        if arr.ndim == 0:
            return jnp.asarray([0], dtype=jnp.int32)
        try:
            return _array_row_lex_order(arr, reverse=reverse)
        except Exception:
            pass
    cells = _cells(value)
    indices = list(range(len(cells)))
    indices.sort(key=lambda i: _sort_key(cells[i]), reverse=reverse)
    return jnp.asarray(indices, dtype=jnp.int32)


def _grade_up(value):
    return _grade(value, reverse=False)


def _grade_down(value):
    return _grade(value, reverse=True)


def _bins(left, right, *, reverse: bool):
    if not isinstance(left, list) and not isinstance(right, list):
        left_arr = _as_array(left)
        right_arr = _as_array(right)
        if left_arr.ndim == 1:
            keys = -left_arr if reverse else left_arr
            queries = -right_arr if reverse else right_arr
            idx = jnp.searchsorted(keys, queries, side="right")
            return lax.convert_element_type(idx, jnp.int32)

    left_cells = _cells(left)
    right_cells = _cells(right)
    right_is_scalar = not isinstance(right, list) and _as_array(right).ndim == 0

    left_keys = [_sort_key(cell) for cell in left_cells]
    out: list[int] = []
    for query in right_cells:
        query_key = _sort_key(query)
        idx = 0
        if reverse:
            while idx < len(left_keys) and left_keys[idx] >= query_key:
                idx += 1
        else:
            while idx < len(left_keys) and left_keys[idx] <= query_key:
                idx += 1
        out.append(idx)

    if right_is_scalar:
        return jnp.asarray(out[0], dtype=jnp.int32)
    return jnp.asarray(out, dtype=jnp.int32)


def _join_monadic(right):
    """Monadic ∾ (Join): concatenate all elements of a list."""
    if isinstance(right, list):
        if len(right) == 0:
            return jnp.asarray([], dtype=jnp.int32)
        result = right[0]
        for item in right[1:]:
            result = _concat(result, item)
        return result
    # On a plain array, monadic Join is identity (already flat).
    return _as_array(right)


def _concat(left, right):
    if isinstance(left, list) and isinstance(right, list):
        return _pack_vector([*left, *right])
    if isinstance(left, list):
        return _pack_vector([*left, right])
    if isinstance(right, list):
        return _pack_vector([left, *right])

    left_arr = _as_array(left)
    right_arr = _as_array(right)
    if left_arr.ndim == 0:
        left_arr = jnp.reshape(left_arr, (1,))
    if right_arr.ndim == 0:
        right_arr = jnp.reshape(right_arr, (1,))
    if left_arr.ndim != right_arr.ndim:
        raise ValueError("∾ requires matching ranks after scalar promotion")
    return jnp.concatenate((left_arr, right_arr), axis=0)


def _take(count, value):
    n = abs(count)

    if isinstance(value, list):
        fill_cell = _fill_cell_for_take(value)
        if count >= 0:
            taken = list(value[:n])
            missing = n - len(taken)
            if missing > 0:
                taken.extend(fill_cell for _ in range(missing))
            return _pack_vector(taken)

        taken = list(value[-n:]) if n > 0 else []
        missing = n - len(taken)
        if missing > 0:
            taken = [*(fill_cell for _ in range(missing)), *taken]
        return _pack_vector(taken)

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


def _drop(count, value):
    if isinstance(value, list):
        if count >= 0:
            return _pack_vector(value[count:])
        return _pack_vector(value[:count])

    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    if count >= 0:
        return arr[count:]
    return arr[:count]


def _coerce_shift_fill(fill, template, *, where: str):
    if isinstance(template, list):
        return fill

    arr = _as_array(template)
    fill_arr = _as_array(fill)
    if arr.ndim == 0:
        if fill_arr.ndim != 0:
            raise ValueError(f"{where} dyadic fill must be scalar for scalar right argument")
        return fill_arr

    cell_shape = tuple(int(d) for d in arr.shape[1:])
    if fill_arr.shape == cell_shape:
        return fill_arr
    if fill_arr.ndim == 0 and not cell_shape:
        return fill_arr
    try:
        return jnp.broadcast_to(fill_arr, cell_shape)
    except Exception as err:  # pragma: no cover - defensive branch
        raise ValueError(f"{where} dyadic fill shape is incompatible with right argument cell shape") from err


def _nudge_after(value, *, fill=_MISSING):
    if isinstance(value, list):
        if not value:
            return _pack_vector([])
        fill_cell = _fill_cell_for_take(value) if fill is _MISSING else fill
        return _pack_vector([fill_cell, *value[:-1]])

    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    if arr.shape[0] == 0:
        return arr

    fill_cell = _fill_cell_for_take(arr) if fill is _MISSING else _coerce_shift_fill(fill, arr, where="»")
    fill_block = _array_fill_block(fill_cell, 1)
    return jnp.concatenate((fill_block, arr[:-1]), axis=0)


def _nudge_before(value, *, fill=_MISSING):
    if isinstance(value, list):
        if not value:
            return _pack_vector([])
        fill_cell = _fill_cell_for_take(value) if fill is _MISSING else fill
        return _pack_vector([*value[1:], fill_cell])

    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    if arr.shape[0] == 0:
        return arr

    fill_cell = _fill_cell_for_take(arr) if fill is _MISSING else _coerce_shift_fill(fill, arr, where="«")
    fill_block = _array_fill_block(fill_cell, 1)
    return jnp.concatenate((arr[1:], fill_block), axis=0)


def _pick(index, value):
    if isinstance(value, list):
        return value[index]

    arr = _as_array(value)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    return arr[index]


def _couple_monadic(value):
    if isinstance(value, list):
        return BoxedArray([value])

    arr = _as_array(value)
    if arr.ndim == 0:
        return jnp.reshape(arr, (1,))
    return jnp.expand_dims(arr, axis=0)


def _couple_dyadic(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return _pack_vector([left, right])

    left_arr = _as_array(left)
    right_arr = _as_array(right)
    if left_arr.shape != right_arr.shape:
        # Mismatched shapes are boxed as a pair in this subset.
        return _pack_vector([left_arr, right_arr])

    if left_arr.ndim == 0:
        return jnp.asarray([left_arr.item(), right_arr.item()])
    return jnp.stack((left_arr, right_arr), axis=0)


def _to_int_index(value, *, where: str) -> int:
    arr = _as_array(value)
    if arr.ndim != 0:
        raise ValueError(f"{where} indices must be scalar integers")
    scalar = arr.item()
    if isinstance(scalar, numbers.Integral):
        return int(scalar)
    if isinstance(scalar, numbers.Real):
        real = float(scalar)
        if real.is_integer():
            return int(real)
    raise ValueError(f"{where} indices must be integers")


def _as_index_spec(value, *, where: str) -> tuple[list[int], tuple[int, ...], bool]:
    if isinstance(value, list):
        return ([_to_int_index(v, where=where) for v in value], (len(value),), False)

    arr = _as_array(value)
    if arr.ndim == 0:
        return ([_to_int_index(arr, where=where)], (), True)

    flat = jnp.ravel(arr)
    indices = [_to_int_index(flat[i], where=where) for i in range(flat.shape[0])]
    return (indices, tuple(int(d) for d in arr.shape), False)


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _normalize_axis_indices(values: list[int], *, axis_length: int, where: str) -> list[int]:
    out: list[int] = []
    for value in values:
        if value < -axis_length or value >= axis_length:
            raise ValueError(f"{where} indexing out-of-bounds")
        out.append(value + axis_length if value < 0 else value)
    return out


def _as_axis_index_array(value, *, axis_length: int, where: str) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
    if isinstance(value, list):
        raw = [_to_int_index(v, where=where) for v in value]
        shape = (len(value),)
    else:
        arr = _as_array(value)
        if arr.ndim == 0:
            raw = [_to_int_index(arr, where=where)]
            shape = ()
        else:
            flat = jnp.ravel(arr)
            raw = [_to_int_index(flat[i], where=where) for i in range(flat.shape[0])]
            shape = tuple(int(d) for d in arr.shape)

    normalized = _normalize_axis_indices(raw, axis_length=axis_length, where=where)
    if shape == ():
        return (jnp.asarray(normalized[0], dtype=jnp.int32), (), True)

    idx = jnp.asarray(normalized, dtype=jnp.int32)
    if len(shape) > 1:
        idx = jnp.reshape(idx, shape)
    return (idx, shape, False)


def _reshape_boxed(value, shape: tuple[int, ...], *, where: str):
    if len(shape) <= 1:
        return value

    if isinstance(value, jnp.ndarray):
        if value.ndim == 0:
            raise ValueError(f"{where} cannot reshape scalar selection to rank-{len(shape)} index shape")
        expected = _shape_size(shape)
        if value.shape[0] != expected:
            raise ValueError(f"{where} selection shape mismatch")
        return jnp.reshape(value, (*shape, *value.shape[1:]))

    if not isinstance(value, list):
        raise ValueError(f"{where} cannot reshape non-array selection")

    expected = _shape_size(shape)
    if len(value) != expected:
        raise ValueError(f"{where} selection shape mismatch")

    def build(offset: int, dims: tuple[int, ...]):
        if len(dims) == 1:
            end = offset + dims[0]
            return _pack_vector(value[offset:end]), end

        rows: list[object] = []
        cur = offset
        for _ in range(dims[0]):
            row, cur = build(cur, dims[1:])
            rows.append(row)
        return _pack_vector(rows), cur

    out, end = build(0, shape)
    if end != expected:
        raise ValueError(f"{where} selection shape mismatch")
    return out


def _cells(value) -> list[object]:
    if isinstance(value, list):
        return value
    arr = _as_array(value)
    if arr.ndim == 0:
        return [arr]
    return [arr[i] for i in range(arr.shape[0])]


def _as_search_queries(value, *, cell_rank: int, where: str) -> tuple[list[object], tuple[int, ...], bool]:
    if isinstance(value, list):
        query_rank = 1
        if query_rank < cell_rank:
            raise ValueError(f"{where} searched-for argument rank must be at least {cell_rank}")
        if cell_rank == 0:
            return (value, (len(value),), False)
        if cell_rank == 1:
            return ([value], (), True)
        raise ValueError(f"{where} searched-for boxed-list rank is too low for cell rank {cell_rank}")

    arr = _as_array(value)
    query_rank = arr.ndim
    if query_rank < cell_rank:
        raise ValueError(f"{where} searched-for argument rank must be at least {cell_rank}")

    if cell_rank == 0:
        if query_rank == 0:
            return ([arr], (), True)
        flat = jnp.ravel(arr)
        queries: list[object] = [flat[i] for i in range(flat.shape[0])]
        return (queries, tuple(int(d) for d in arr.shape), False)

    prefix_rank = query_rank - cell_rank
    if prefix_rank == 0:
        return ([arr], (), True)

    query_shape = tuple(int(d) for d in arr.shape[:prefix_rank])
    queries = [arr[idx] for idx in product(*[range(dim) for dim in query_shape])]
    return (queries, query_shape, False)


def _pack_query_result(values: list[int], query_shape: tuple[int, ...], *, is_scalar: bool):
    if is_scalar:
        return jnp.asarray(values[0], dtype=jnp.int32)

    out = jnp.asarray(values, dtype=jnp.int32)
    if len(query_shape) > 1:
        out = jnp.reshape(out, query_shape)
    return out


def _principal_search_cells(value, *, where: str) -> tuple[list[object], int]:
    if isinstance(value, list):
        return (value, 0)

    arr = _as_array(value)
    if arr.ndim == 0:
        raise ValueError(f"{where} searched-in argument must have rank at least 1")
    return ([arr[i] for i in range(arr.shape[0])], arr.ndim - 1)


def _prepare_search(
    principal,
    query,
    *,
    where: str,
) -> tuple[list[object], list[object], tuple[int, ...], bool]:
    principal_cells, cell_rank = _principal_search_cells(principal, where=where)
    queries, query_shape, is_scalar = _as_search_queries(query, cell_rank=cell_rank, where=where)
    return principal_cells, queries, query_shape, is_scalar


def _fast_array_search_views(
    principal,
    query,
    *,
    where: str,
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[int, ...], bool] | None:
    if isinstance(principal, list) or isinstance(query, list):
        return None

    principal_arr = _as_array(principal)
    if principal_arr.ndim == 0:
        raise ValueError(f"{where} searched-in argument must have rank at least 1")

    cell_rank = principal_arr.ndim - 1
    cell_size = 1 if cell_rank == 0 else int(math.prod(int(d) for d in principal_arr.shape[1:]))
    principal_flat = jnp.reshape(principal_arr, (int(principal_arr.shape[0]), cell_size))

    query_arr = _as_array(query)
    if query_arr.ndim < cell_rank:
        raise ValueError(f"{where} searched-for argument rank must be at least {cell_rank}")

    if cell_rank == 0:
        query_shape = tuple(int(d) for d in query_arr.shape)
        query_flat = jnp.reshape(query_arr, (-1, 1))
        is_scalar = query_arr.ndim == 0
        return principal_flat, query_flat, query_shape, is_scalar

    prefix_rank = query_arr.ndim - cell_rank
    if prefix_rank == 0:
        query_flat = jnp.reshape(query_arr, (1, cell_size))
        return principal_flat, query_flat, (), True

    query_shape = tuple(int(d) for d in query_arr.shape[:prefix_rank])
    query_flat = jnp.reshape(query_arr, (-1, cell_size))
    return principal_flat, query_flat, query_shape, False


def _pack_array_search_result(values: jnp.ndarray, query_shape: tuple[int, ...], *, is_scalar: bool):
    out = lax.convert_element_type(values, jnp.int32)
    if is_scalar:
        return out[0]
    if len(query_shape) > 1:
        return jnp.reshape(out, query_shape)
    return out


def _first_match_indices(match_matrix: jnp.ndarray, *, missing_index: int) -> jnp.ndarray:
    has = jnp.any(match_matrix, axis=1)
    first = jnp.argmax(match_matrix, axis=1)
    missing = jnp.asarray(missing_index, dtype=first.dtype)
    return jnp.where(has, first, missing)


def _cell_equality_matrix(cells: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(cells[:, None, :] == cells[None, :, :], axis=2)


def _select_first_axis(left, right):
    indices, index_shape, is_scalar = _as_index_spec(left, where="⊏")

    if isinstance(right, list):
        indices = _normalize_axis_indices(indices, axis_length=len(right), where="⊏")
        if is_scalar:
            return right[indices[0]]
        out = _pack_vector([right[i] for i in indices])
        if len(index_shape) > 1:
            out = _reshape_boxed(out, index_shape, where="⊏")
        return out

    arr = _as_array(right)
    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,))
    indices = _normalize_axis_indices(indices, axis_length=int(arr.shape[0]), where="⊏")
    if is_scalar:
        return arr[indices[0]]

    out = arr[jnp.asarray(indices, dtype=jnp.int32)]
    if len(index_shape) > 1:
        out = jnp.reshape(out, (*index_shape, *arr.shape[1:]))
    return out


def _select_multi_axis(left: list[object], right):
    if not left:
        raise ValueError("⊏ multi-axis selection requires a non-empty index list")

    if isinstance(right, list):
        if len(left) != 1:
            raise ValueError("⊏ multi-axis selection rank exceeds right argument rank")
        return _select_first_axis(left[0], right)

    arr = _as_array(right)
    if arr.ndim < len(left):
        raise ValueError("⊏ multi-axis selection rank exceeds right argument rank")

    result = arr
    axis = 0
    for index_spec in left:
        axis_length = int(result.shape[axis])
        idx_arr, idx_shape, is_scalar = _as_axis_index_array(index_spec, axis_length=axis_length, where="⊏")
        result = jnp.take(result, idx_arr, axis=axis)
        if not is_scalar:
            axis += len(idx_shape)
    return result


def _select(left, right):
    if isinstance(left, list) and left:
        return _select_multi_axis(left, right)
    return _select_first_axis(left, right)


def _index_of(left, right):
    fast = _fast_array_search_views(left, right, where="⊐")
    if fast is not None:
        principal_flat, query_flat, query_shape, is_scalar = fast
        matches = jnp.all(query_flat[:, None, :] == principal_flat[None, :, :], axis=2)
        out = _first_match_indices(matches, missing_index=int(principal_flat.shape[0]))
        return _pack_array_search_result(out, query_shape, is_scalar=is_scalar)

    left_cells, queries, query_shape, is_scalar = _prepare_search(left, right, where="⊐")

    out: list[int] = []
    for query in queries:
        found = len(left_cells)
        for i, candidate in enumerate(left_cells):
            if _match_values(candidate, query):
                found = i
                break
        out.append(found)
    return _pack_query_result(out, query_shape, is_scalar=is_scalar)


def _require_rank_at_least_one(value, *, where: str) -> None:
    if isinstance(value, list):
        return
    arr = _as_array(value)
    if arr.ndim == 0:
        raise ValueError(f"{where} requires an array of rank at least 1")


def _mark_firsts(value):
    _require_rank_at_least_one(value, where="∊")
    if not isinstance(value, list):
        arr = _as_array(value)
        if arr.ndim >= 1:
            cells = jnp.reshape(arr, (int(arr.shape[0]), -1))
            eq = _cell_equality_matrix(cells)
            lower = jnp.tril(jnp.ones((int(cells.shape[0]), int(cells.shape[0])), dtype=jnp.bool_))
            first_idx = jnp.argmax(eq & lower, axis=1)
            marks = first_idx == jnp.arange(int(cells.shape[0]), dtype=first_idx.dtype)
            return lax.convert_element_type(marks, jnp.int32)

    cells = _cells(value)
    out: list[int] = []
    seen: list[object] = []
    for cell in cells:
        first = 1
        for prev in seen:
            if _match_values(prev, cell):
                first = 0
                break
        out.append(first)
        seen.append(cell)
    return jnp.asarray(out, dtype=jnp.int32)


def _classify(value):
    _require_rank_at_least_one(value, where="⊐")
    if not isinstance(value, list):
        arr = _as_array(value)
        if arr.ndim >= 1:
            cells = jnp.reshape(arr, (int(arr.shape[0]), -1))
            eq = _cell_equality_matrix(cells)
            lower = jnp.tril(jnp.ones((int(cells.shape[0]), int(cells.shape[0])), dtype=jnp.bool_))
            first_idx = jnp.argmax(eq & lower, axis=1)
            _, inverse = jnp.unique(first_idx, return_inverse=True)
            return lax.convert_element_type(inverse, jnp.int32)

    cells = _cells(value)
    out: list[int] = []
    classes: list[object] = []
    for cell in cells:
        cls = None
        for i, existing in enumerate(classes):
            if _match_values(existing, cell):
                cls = i
                break
        if cls is None:
            classes.append(cell)
            cls = len(classes) - 1
        out.append(cls)
    return jnp.asarray(out, dtype=jnp.int32)


def _as_integer_array(value, *, where: str):
    arr = _as_array(value)
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return arr
    if jnp.issubdtype(arr.dtype, jnp.floating):
        rounded = jnp.round(arr)
        if not bool(jnp.all(arr == rounded)):
            raise ValueError(f"{where} requires integer arguments")
        return rounded.astype(jnp.int32)
    raise ValueError(f"{where} requires integer arguments")


def _occurrence_count(value):
    _require_rank_at_least_one(value, where="⊒")
    if not isinstance(value, list):
        arr = _as_array(value)
        if arr.ndim >= 1:
            cells = jnp.reshape(arr, (int(arr.shape[0]), -1))
            eq = _cell_equality_matrix(cells)
            strict_lower = jnp.tril(jnp.ones((int(cells.shape[0]), int(cells.shape[0])), dtype=jnp.int32), k=-1)
            counts = jnp.sum(lax.convert_element_type(eq, jnp.int32) * strict_lower, axis=1)
            return lax.convert_element_type(counts, jnp.int32)

    cells = _cells(value)
    out: list[int] = []
    seen: list[object] = []
    for cell in cells:
        count = 0
        for prev in seen:
            if _match_values(prev, cell):
                count += 1
        out.append(count)
        seen.append(cell)
    return jnp.asarray(out, dtype=jnp.int32)


def _deduplicate(value):
    _require_rank_at_least_one(value, where="⍷")
    if not isinstance(value, list):
        arr = _as_array(value)
        if arr.ndim >= 1:
            cells = jnp.reshape(arr, (int(arr.shape[0]), -1))
            eq = _cell_equality_matrix(cells)
            lower = jnp.tril(jnp.ones((int(cells.shape[0]), int(cells.shape[0])), dtype=jnp.bool_))
            first_idx = jnp.argmax(eq & lower, axis=1)
            marks = first_idx == jnp.arange(int(cells.shape[0]), dtype=first_idx.dtype)
            return arr[marks]

    cells = _cells(value)
    out: list[object] = []
    for cell in cells:
        if not any(_match_values(cell, existing) for existing in out):
            out.append(cell)
    return _pack_vector(out)


def _member_of(left, right):
    fast = _fast_array_search_views(right, left, where="∊")
    if fast is not None:
        right_flat, query_flat, query_shape, is_scalar = fast
        matches = jnp.all(query_flat[:, None, :] == right_flat[None, :, :], axis=2)
        member = jnp.any(matches, axis=1)
        return _pack_array_search_result(lax.convert_element_type(member, jnp.int32), query_shape, is_scalar=is_scalar)

    right_cells, queries, query_shape, is_scalar = _prepare_search(right, left, where="∊")
    out: list[int] = []
    for query in queries:
        out.append(1 if any(_match_values(candidate, query) for candidate in right_cells) else 0)
    return _pack_query_result(out, query_shape, is_scalar=is_scalar)


def _progressive_index_of(left, right):
    fast = _fast_array_search_views(left, right, where="⊒")
    if fast is not None:
        left_flat, query_flat, query_shape, is_scalar = fast
        missing = int(left_flat.shape[0])

        def _step(used: jnp.ndarray, query_vec: jnp.ndarray):
            matches = jnp.all(left_flat == query_vec[None, :], axis=1)
            available = matches & (~used)
            has = jnp.any(available)
            first = jnp.argmax(available)
            out = jnp.where(has, first, jnp.asarray(missing, dtype=first.dtype))

            def _mark(u: jnp.ndarray) -> jnp.ndarray:
                return u.at[out].set(True)

            next_used = lax.cond(has, _mark, lambda u: u, used)
            return next_used, out

        init_used = jnp.zeros((missing,), dtype=jnp.bool_)
        _, out = lax.scan(_step, init_used, query_flat)
        return _pack_array_search_result(out, query_shape, is_scalar=is_scalar)

    left_cells, queries, query_shape, is_scalar = _prepare_search(left, right, where="⊒")

    used = [False] * len(left_cells)
    out: list[int] = []
    for query in queries:
        found = len(left_cells)
        for i, candidate in enumerate(left_cells):
            if used[i]:
                continue
            if _match_values(candidate, query):
                found = i
                used[i] = True
                break
        out.append(found)
    return _pack_query_result(out, query_shape, is_scalar=is_scalar)


def _find_array_windows(left_arr: jnp.ndarray, right_arr: jnp.ndarray) -> jnp.ndarray:
    left_rank = left_arr.ndim
    right_rank = right_arr.ndim
    if left_rank > right_rank:
        raise ValueError("⍷ left argument rank cannot exceed right argument rank")

    frame_rank = right_rank - left_rank
    frame_shape = tuple(int(d) for d in right_arr.shape[:frame_rank])
    right_cell_shape = tuple(int(d) for d in right_arr.shape[frame_rank:])
    left_shape = tuple(int(d) for d in left_arr.shape)

    out_cell_shape = tuple(max(0, right_cell_shape[i] - left_shape[i] + 1) for i in range(left_rank))
    if any(dim == 0 for dim in out_cell_shape):
        return jnp.zeros((*frame_shape, *out_cell_shape), dtype=jnp.int32)

    # Scalar-cell search degenerates to elementwise equality over frames.
    if left_rank == 0:
        return lax.convert_element_type(right_arr == left_arr, jnp.int32)

    start_axes = [jnp.arange(dim, dtype=jnp.int32) for dim in out_cell_shape]
    start_mesh = jnp.meshgrid(*start_axes, indexing="ij")
    starts = jnp.stack(start_mesh, axis=-1).reshape((-1, left_rank))

    offset_axes = [jnp.arange(dim, dtype=jnp.int32) for dim in left_shape]
    offset_mesh = jnp.meshgrid(*offset_axes, indexing="ij")
    offsets = jnp.stack(offset_mesh, axis=-1).reshape((-1, left_rank))
    left_flat = jnp.reshape(left_arr, (1, -1))

    def _find_one_cell(cell: jnp.ndarray) -> jnp.ndarray:
        idx = starts[:, None, :] + offsets[None, :, :]
        idx_tuple = tuple(idx[..., axis] for axis in range(left_rank))
        windows = cell[idx_tuple]
        windows_flat = jnp.reshape(windows, (starts.shape[0], -1))
        matches = jnp.all(windows_flat == left_flat, axis=1)
        return jnp.reshape(lax.convert_element_type(matches, jnp.int32), out_cell_shape)

    if frame_rank == 0:
        return _find_one_cell(right_arr)

    frame_count = int(math.prod(frame_shape))
    cells = jnp.reshape(right_arr, (frame_count, *right_cell_shape))
    out = jax.vmap(_find_one_cell)(cells)
    return jnp.reshape(out, (*frame_shape, *out_cell_shape))


def _find(left, right):
    def promote(value):
        if isinstance(value, list):
            return value
        arr = _as_array(value)
        if arr.ndim == 0:
            return jnp.reshape(arr, (1,))
        return arr

    def iter_indices(shape: tuple[int, ...]):
        if not shape:
            yield ()
            return
        for idx in product(*[range(int(dim)) for dim in shape]):
            yield tuple(int(i) for i in idx)

    def subarray(container, start: tuple[int, ...], shape: tuple[int, ...]):
        if isinstance(container, list):
            if len(start) != 1 or len(shape) != 1:
                raise ValueError("⍷ boxed-list search currently supports rank-1 cells only")
            begin = start[0]
            end = begin + shape[0]
            return container[begin:end]

        arr = _as_array(container)
        slices = tuple(slice(start[i], start[i] + shape[i]) for i in range(len(shape)))
        return arr[slices]

    def cell_at(value, index: tuple[int, ...]):
        if isinstance(value, list):
            if index:
                raise ValueError("⍷ boxed-list frame rank must be 0")
            return value
        if not index:
            return value
        return value[index]

    left = promote(left)
    right = promote(right)

    if not isinstance(left, list) and not isinstance(right, list):
        return _find_array_windows(_as_array(left), _as_array(right))

    left_rank = _rank_of(left)
    right_rank = _rank_of(right)
    if left_rank > right_rank:
        raise ValueError("⍷ left argument rank cannot exceed right argument rank")

    left_shape = _shape_of(left)
    right_shape = _shape_of(right)

    frame_rank = right_rank - left_rank
    frame_shape = right_shape[:frame_rank]
    right_cell_shape = right_shape[frame_rank:]
    out_cell_shape = tuple(max(0, right_cell_shape[i] - left_shape[i] + 1) for i in range(left_rank))
    result_shape = (*frame_shape, *out_cell_shape)

    frame_indices = list(iter_indices(frame_shape))
    window_indices = list(iter_indices(out_cell_shape))
    out: list[int] = []
    for frame_index in frame_indices:
        right_cell = cell_at(right, frame_index)
        for start in window_indices:
            candidate = subarray(right_cell, start, left_shape)
            out.append(1 if _match_values(left, candidate) else 0)

    result = jnp.asarray(out, dtype=jnp.int32)
    if result_shape:
        result = jnp.reshape(result, result_shape)
    return result


def _as_group_keys(value, *, length: int | None = None) -> list[int]:
    if isinstance(value, list):
        keys = [_to_int_index(v, where="⊔") for v in value]
    else:
        arr = _as_array(value)
        if arr.ndim == 0:
            keys = [_to_int_index(arr, where="⊔")]
        else:
            flat = jnp.ravel(arr)
            keys = [_to_int_index(flat[i], where="⊔") for i in range(flat.shape[0])]

    for k in keys:
        if k < -1:
            raise ValueError("⊔ keys must be at least ¯1")

    if length is not None and len(keys) not in {1, length}:
        raise ValueError("⊔ key vector length must match right argument length or be scalar")
    return keys


def _group(left, right):
    right_cells = _cells(right)
    keys = _as_group_keys(left, length=len(right_cells))
    if len(keys) == 1 and len(right_cells) > 1:
        keys = keys * len(right_cells)

    group_count = 0 if not keys else max(keys) + 1
    buckets: list[list[object]] = [[] for _ in range(group_count)]
    for key, cell in zip(keys, right_cells, strict=True):
        if key == -1:
            continue
        buckets[key].append(cell)

    # Group returns boxed groups, so keep the outer container boxed.
    return BoxedArray([_pack_vector(bucket) for bucket in buckets])


def _group_monadic(value):
    keys = _as_group_keys(value)
    indices = jnp.arange(len(keys), dtype=jnp.int32)
    return _group(keys, indices)


def _as_replicate_counts(left, length: int | None = None) -> list[int]:
    if isinstance(left, list):
        counts = [_to_int_index(v, where="/") for v in left]
    else:
        arr = _as_array(left)
        if arr.ndim == 0:
            counts = [_to_int_index(arr, where="/")]
        else:
            flat = jnp.ravel(arr)
            counts = [_to_int_index(flat[i], where="/") for i in range(flat.shape[0])]

    for c in counts:
        if c < 0:
            raise ValueError("/ replication counts must be non-negative")

    if length is not None and len(counts) not in {1, length}:
        raise ValueError("/ count vector length must match right argument length or be scalar")
    return counts


def _replicate(left, right):
    right_cells = _cells(right)
    counts = _as_replicate_counts(left, length=len(right_cells))
    if len(counts) == 1 and len(right_cells) > 1:
        counts = counts * len(right_cells)

    out: list[object] = []
    for count, cell in zip(counts, right_cells, strict=True):
        out.extend(cell for _ in range(count))
    return _pack_vector(out)


def _eval_infix(op: str, left, right):
    if op.startswith("´"):
        return _fold(op[1:], right, init=left)

    if op == "≡":
        return jnp.asarray(1 if _match_values(left, right) else 0, dtype=jnp.int32)
    if op in _BASE_BINARY_FALLBACK:
        fast = _fast_binary_base(op, left, right)
        if fast is not _NO_FAST_PATH:
            return fast
        return _map_binary(_BASE_BINARY_FALLBACK[op], left, right)

    if op == "⥊":
        shape = _as_shape(left)
        x = _as_array(right)
        return jnp.reshape(x, shape)
    if op == "⊣":
        return left
    if op == "⊢":
        return right
    if op == "∾":
        return _concat(left, right)
    if op == "↑":
        count = _as_python_int_scalar(left, where="↑")
        return _take(count, right)
    if op == "↓":
        count = _as_python_int_scalar(left, where="↓")
        return _drop(count, right)
    if op == "⌽":
        count = _as_python_int_scalar(left, where="⌽")
        return _rotate(count, right)
    if op == "»":
        return _nudge_after(right, fill=left)
    if op == "«":
        return _nudge_before(right, fill=left)
    if op == "⍉":
        return _transpose_axes(left, right)
    if op == "!":
        message = _format_assert_message(left)
        return _assert_condition(right, message=message)
    if op == "⊑":
        index = _as_python_int_scalar(left, where="⊑")
        return _pick(index, right)
    if op == "⊏":
        return _select(left, right)
    if op == "⊐":
        return _index_of(left, right)
    if op == "⊒":
        return _progressive_index_of(left, right)
    if op == "∊":
        return _member_of(left, right)
    if op == "⍷":
        return _find(left, right)
    if op == "≍":
        return _couple_dyadic(left, right)
    if op == "⋈":
        return _couple_dyadic(left, right)
    if op == "/":
        return _replicate(left, right)
    if op == "⊔":
        return _group(left, right)
    if op == "⍋":
        return _bins(left, right, reverse=False)
    if op == "⍒":
        return _bins(left, right, reverse=True)

    raise ValueError(f"Unsupported dyadic primitive {op!r}")


def _extract_sequence(value) -> list[object]:
    if isinstance(value, list):
        return value
    arr = _as_array(value)
    if arr.ndim != 1:
        raise ValueError("Destructuring assignment expects a rank-1 array or list")
    return [arr[i] for i in range(arr.shape[0])]


def _extract_namespace(value: object, *, where: str) -> Namespace:
    if not isinstance(value, Namespace):
        raise ValueError(f"{where} requires a namespace value")
    return value


def _lookup_namespace_value(namespace: Namespace, name: str) -> object:
    if name in namespace.values:
        return namespace.values[name]

    wanted = _normalize_identifier_name(name)
    for key, value in namespace.values.items():
        if _normalize_identifier_name(key) == wanted:
            return value
    raise NameError(f"Undefined namespace member {name!r}")


def _reshape_modifier_output(values: list[object], frame_shape: tuple[int, ...], *, where: str):
    packed = _pack_vector(values)
    if not frame_shape:
        return packed

    if isinstance(packed, jnp.ndarray):
        return jnp.reshape(packed, (*frame_shape, *packed.shape[1:]))
    return _reshape_boxed(packed, frame_shape, where=where)


def _is_callable_value(value: object) -> bool:
    return is_operation(value)


def _as_rank_spec_values(value) -> list[int]:
    if isinstance(value, list):
        items = value
    else:
        arr = _as_array(value)
        if arr.ndim == 0:
            items = [arr]
        elif arr.ndim == 1:
            items = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError("⎉ rank spec must be a scalar or rank-1 vector")

    ranks = [_as_python_int_scalar(item, where="⎉") for item in items]
    if not ranks:
        raise ValueError("⎉ rank spec cannot be empty")
    if len(ranks) > 2:
        raise ValueError("⎉ dyadic rank spec must contain at most two ranks")
    return ranks


def _normalize_rank(rank: int, argument_rank: int) -> int:
    if rank >= 0:
        return min(rank, argument_rank)
    return max(0, argument_rank + rank)


def _rank_cells(value, *, rank: int) -> tuple[list[object], tuple[int, ...], bool]:
    if isinstance(value, list):
        cell_rank = _normalize_rank(rank, 1)
        if cell_rank == 1:
            return ([value], (), True)
        return (list(value), (len(value),), False)

    arr = _as_array(value)
    argument_rank = arr.ndim
    cell_rank = _normalize_rank(rank, argument_rank)
    if cell_rank == argument_rank:
        return ([arr], (), True)

    frame_shape = tuple(int(d) for d in arr.shape[: argument_rank - cell_rank])
    if cell_rank == 0:
        flat = jnp.reshape(arr, (-1,))
        return ([flat[i] for i in range(flat.shape[0])], frame_shape, False)

    cell_shape = tuple(int(d) for d in arr.shape[argument_rank - cell_rank :])
    rows = jnp.reshape(arr, (-1, *cell_shape))
    return ([rows[i] for i in range(rows.shape[0])], frame_shape, False)


def _apply_rank_monadic(func_value, right_value, *, rank: int):
    cells, frame_shape, is_scalar = _rank_cells(right_value, rank=rank)
    if is_scalar:
        return _apply_callable(func_value, cells[0])

    values = [_apply_callable(func_value, cell) for cell in cells]
    return _reshape_modifier_output(values, frame_shape, where="⎉")


def _apply_rank_dyadic(func_value, left_value, right_value, *, left_rank: int, right_rank: int):
    left_cells, left_frame_shape, _ = _rank_cells(left_value, rank=left_rank)
    right_cells, right_frame_shape, _ = _rank_cells(right_value, rank=right_rank)

    if left_frame_shape and right_frame_shape and left_frame_shape != right_frame_shape:
        raise ValueError("⎉ dyadic frame shapes must match")
    frame_shape = right_frame_shape if right_frame_shape else left_frame_shape

    if len(left_cells) == 1 and len(right_cells) > 1:
        left_cells = left_cells * len(right_cells)
    if len(right_cells) == 1 and len(left_cells) > 1:
        right_cells = right_cells * len(left_cells)
    if len(left_cells) != len(right_cells):
        raise ValueError("⎉ dyadic frame lengths are incompatible")

    values = [_apply_callable(func_value, r_item, left_value=l_item) for l_item, r_item in zip(left_cells, right_cells, strict=True)]
    if not frame_shape:
        return values[0]
    return _reshape_modifier_output(values, frame_shape, where="⎉")


def _apply_at_depth(func_value, value, *, depth: int, left_value=_MISSING):
    if depth <= 0:
        if left_value is _MISSING:
            return _apply_callable(func_value, value)
        return _apply_callable(func_value, value, left_value=left_value)

    if isinstance(value, list):
        return _pack_vector([_apply_at_depth(func_value, item, depth=depth - 1, left_value=left_value) for item in value])
    return value


def _export_from_target(target: Expr, env: Scope) -> None:
    if isinstance(target, Name):
        env.mark_export(target.value)
        return

    if isinstance(target, Nothing):
        return

    if isinstance(target, Vector):
        for item in target.items:
            _export_from_target(item, env)
        return

    raise SyntaxError("Export target must be a name or destructuring vector")


def _identity_for_fold(op: str):
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


def _apply_dyad(op: str, left, right):
    return _eval_infix(op, left, right)


def _fast_fold_primitive_array(op: str, value, *, init=_MISSING):
    if not _USE_PRIMITIVE_FOLD_FAST_PATH:
        return _NO_FAST_PATH
    if isinstance(value, list):
        return _NO_FAST_PATH
    arr = _as_array(value)
    if arr.ndim < 1:
        return _NO_FAST_PATH

    axis_len = int(arr.shape[0])
    if axis_len == 0:
        if init is not _MISSING:
            return init
        identity = _identity_for_fold(op)
        if identity is _MISSING:
            raise ValueError(f"Fold of empty array has no known identity for op {op!r}")
        return identity

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

    if reduced is _MISSING:
        return _NO_FAST_PATH
    if init is _MISSING:
        return reduced
    return _apply_dyad(op, init, reduced)


def _fold(op: str, value, *, init=_MISSING):
    fast = _fast_fold_primitive_array(op, value, init=init)
    if fast is not _NO_FAST_PATH:
        return fast

    sequence = _extract_sequence(value)

    if not sequence:
        if init is not _MISSING:
            return init
        identity = _identity_for_fold(op)
        if identity is _MISSING:
            raise ValueError(f"Fold of empty array has no known identity for op {op!r}")
        return identity

    if init is _MISSING:
        acc = sequence[-1]
    else:
        acc = _apply_dyad(op, init, sequence[-1])

    for index in range(len(sequence) - 2, -1, -1):
        acc = _apply_dyad(op, sequence[index], acc)
    return acc


def _fold_callable(func_value, value, *, init=_MISSING):
    if isinstance(func_value, PrimitiveFunction):
        return _fold(func_value.op, value, init=init)

    sequence = _extract_sequence(value)

    if not sequence:
        if init is not _MISSING:
            return init
        if isinstance(func_value, PrimitiveFunction):
            identity = _identity_for_fold(func_value.op)
            if identity is not _MISSING:
                return identity
        raise ValueError("Fold of empty array has no known identity")

    if init is _MISSING:
        acc = sequence[-1]
    else:
        acc = _apply_callable(func_value, sequence[-1], left_value=init)

    for index in range(len(sequence) - 2, -1, -1):
        acc = _apply_callable(func_value, acc, left_value=sequence[index])
    return acc


def _scan_callable(func_value, value, *, init=_MISSING):
    sequence = _extract_sequence(value)
    if not sequence:
        return _pack_vector([])

    out: list[object] = []
    if init is _MISSING:
        acc = sequence[0]
        out.append(acc)
        start = 1
    else:
        acc = _apply_callable(func_value, sequence[0], left_value=init)
        out.append(acc)
        start = 1

    for index in range(start, len(sequence)):
        acc = _apply_callable(func_value, sequence[index], left_value=acc)
        out.append(acc)
    return _pack_vector(out)


def _inverse_axes_spec(left) -> jnp.ndarray | int:
    axes_arr = _as_array(left)
    if axes_arr.ndim == 0:
        return _as_python_int_scalar(axes_arr, where="⍉⁼")
    if axes_arr.ndim != 1:
        raise ValueError("⍉⁼ left argument must be a scalar or rank-1 axis vector")

    axes = [_as_python_int_scalar(axes_arr[i], where="⍉⁼") for i in range(axes_arr.shape[0])]
    rank = len(axes)
    normalized: list[int] = []
    for axis in axes:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise ValueError("⍉⁼ axis out of bounds")
        normalized.append(axis)
    if len(set(normalized)) != rank:
        raise ValueError("⍉⁼ axis vector must be a permutation")

    inverse = [0] * rank
    for i, axis in enumerate(normalized):
        inverse[axis] = i
    return jnp.asarray(inverse, dtype=jnp.int32)


def _undo_primitive(op: str, result, *, left_value=_MISSING):
    if left_value is _MISSING:
        if op == "+":
            return _map_unary(lambda v: jnp.conjugate(_as_array(v)), result)
        if op == "-":
            return _map_unary(lambda v: -_as_array(v), result)
        if op == "÷":
            return _map_unary(lambda v: 1 / _as_array(v), result)
        if op == "⋆":
            return _map_unary(lambda v: jnp.log(_as_array(v)), result)
        if op == "¬":
            return _map_unary(lambda v: 1 - _as_array(v), result)
        if op == "⌽":
            return _reverse(result)
        if op == "»":
            return _nudge_before(result)
        if op == "«":
            return _nudge_after(result)
        if op == "⍉":
            return _transpose(result)
        if op == "√":
            return _map_unary(lambda v: _as_array(v) ** 2, result)
        raise ValueError(f"Undo is not available for monadic primitive {op!r}")

    if op == "+":
        return _map_binary(lambda w, y: _as_array(y) - _as_array(w), left_value, result)
    if op == "-":
        return _map_binary(lambda w, y: _as_array(w) - _as_array(y), left_value, result)
    if op == "×":
        return _map_binary(lambda w, y: _as_array(y) / _as_array(w), left_value, result)
    if op == "÷":
        return _map_binary(lambda w, y: _as_array(w) / _as_array(y), left_value, result)
    if op == "⋆":
        return _map_binary(lambda w, y: jnp.log(_as_array(y)) / jnp.log(_as_array(w)), left_value, result)
    if op == "√":
        return _map_binary(lambda w, y: _as_array(y) ** _as_array(w), left_value, result)
    if op == "⌽":
        count = _as_python_int_scalar(left_value, where="⌽⁼")
        return _rotate(-count, result)
    if op == "»":
        return _nudge_before(result, fill=left_value)
    if op == "«":
        return _nudge_after(result, fill=left_value)
    if op == "⍉":
        inverse_spec = _inverse_axes_spec(left_value)
        return _transpose_axes(inverse_spec, result)

    raise ValueError(f"Undo is not available for dyadic primitive {op!r}")


def _undo_callable(func_value, result, *, left_value=_MISSING):
    if isinstance(func_value, PrimitiveFunction):
        return _undo_primitive(func_value.op, result, left_value=left_value)
    if callable(func_value):
        inverse = getattr(func_value, "_bqn_inverse", None)
        if inverse is None:
            raise ValueError("Undo is not available for callable without _bqn_inverse")
        if left_value is _MISSING:
            return inverse(result)
        try:
            return inverse(left_value, result)
        except TypeError:
            return inverse(result)
    raise ValueError(f"Undo is not available for callable {type(func_value).__name__}")


def _assign_target(target: Expr, value, env: Scope, *, op: str) -> None:
    if isinstance(target, Name):
        if target.value.startswith("•"):
            raise ValueError("System values are read-only")
        if _canonical_special_name(target.value) is not None and op != "↩":
            raise ValueError("Special names cannot be assigned with ← or ⇐")
        if op == "↩":
            env.set_existing(target.value, value)
        else:
            env.define(target.value, value)
        return

    if isinstance(target, Nothing):
        if op == "↩":
            raise SyntaxError("Modified assignment target cannot be ·")
        return

    if isinstance(target, Vector):
        if isinstance(value, Namespace):
            for sub_target in target.items:
                if not isinstance(sub_target, Name):
                    raise ValueError("Namespace destructuring assignment targets must be names")
                sub_value = _lookup_namespace_value(value, sub_target.value)
                _assign_target(sub_target, sub_value, env, op=op)
            return

        seq = _extract_sequence(value)
        if len(seq) != len(target.items):
            raise ValueError("Destructuring target length mismatch")
        for sub_target, sub_value in zip(target.items, seq, strict=True):
            _assign_target(sub_target, sub_value, env, op=op)
        return

    raise SyntaxError("Left-hand side of assignment must be a name or destructuring vector")


def _is_general_case(case: Case) -> bool:
    return case.header is None and not case.predicates


def _header_target_is_ambivalent_w(target: Expr) -> bool:
    return isinstance(target, Name) and _canonical_special_name(target.value) == "𝕨"


def _case_is_compatible(
    case: Case,
    case_index: int,
    all_cases: tuple[Case, ...],
    left_value,
) -> bool:
    has_left = left_value is not _MISSING

    if case.header is None:
        # Preserve existing two-general-case dispatch ({monadic;dyadic}) behavior.
        if (
            has_left
            and len(all_cases) == 2
            and case_index == 0
            and _is_general_case(all_cases[0])
            and _is_general_case(all_cases[1])
        ):
            return False
        return True

    if len(case.header) == 1:
        if _header_target_is_ambivalent_w(case.header[0]):
            return True
        return not has_left
    if len(case.header) == 2:
        return has_left

    raise ValueError("Case headers currently support one or two names")


def _bind_header_name(call_scope: Scope, name: str, value) -> None:
    if _canonical_special_name(name) is not None:
        # Header use of special names has no effect.
        return
    call_scope.define(name, value)


def _bind_header_target(call_scope: Scope, target: Expr, value) -> None:
    if isinstance(target, Name):
        _bind_header_name(call_scope, target.value, value)
        return

    if isinstance(target, Vector):
        seq = _extract_sequence(value)
        if len(seq) != len(target.items):
            raise ValueError("Header destructuring target length mismatch")
        for sub_target, sub_value in zip(target.items, seq, strict=True):
            _bind_header_target(call_scope, sub_target, sub_value)
        return

    raise ValueError("Case header targets must be names or destructuring vectors")


def _bind_case_header(case: Case, call_scope: Scope, left_value, right_value) -> None:
    if case.header is None:
        return

    if len(case.header) == 1:
        _bind_header_target(call_scope, case.header[0], right_value)
        return

    if len(case.header) == 2:
        if left_value is _MISSING:
            raise ValueError("Dyadic case header requires a left argument")
        _bind_header_target(call_scope, case.header[0], left_value)
        _bind_header_target(call_scope, case.header[1], right_value)
        return

    raise ValueError("Case headers currently support one or two names")


def _predicate_to_bool(value) -> bool:
    arr = _as_array(value)
    if arr.ndim != 0:
        raise ValueError("Predicate expression must produce a scalar 0 or 1")

    scalar = arr.item()
    if isinstance(scalar, numbers.Integral):
        intval = int(scalar)
        if intval in {0, 1}:
            return bool(intval)
        raise ValueError("Predicate scalar must be exactly 0 or 1")

    if isinstance(scalar, numbers.Real):
        real = float(scalar)
        if real in {0.0, 1.0}:
            return bool(int(real))
        raise ValueError("Predicate scalar must be exactly 0 or 1")

    raise ValueError("Predicate expression must produce a numeric scalar 0 or 1")


def _case_predicates_pass(case: Case, call_scope: Scope) -> bool:
    for predicate in case.predicates:
        if not _predicate_to_bool(_eval_expr(predicate, call_scope)):
            return False
    return True


def _wrap_insert_result(result, template):
    if isinstance(template, list) or isinstance(result, list):
        return _pack_vector([result])

    template_arr = _as_array(template)
    if template_arr.ndim == 0:
        raise ValueError("˝ requires right argument rank at least 1")

    result_arr = _as_array(result)
    if result_arr.ndim == 0:
        return jnp.reshape(result_arr, (1,))
    return jnp.expand_dims(result_arr, axis=0)


def _apply_callable(func_value, right_value, *, left_value=_MISSING):
    if isinstance(func_value, DerivedTrain):
        if len(func_value.parts) == 2:
            left_func, right_func = func_value.parts
            if left_value is _MISSING:
                right_result = _apply_callable(right_func, right_value)
            else:
                right_result = _apply_callable(right_func, right_value, left_value=left_value)
            return _apply_callable(left_func, right_result)

        if len(func_value.parts) == 3:
            left_func, center_func, right_func = func_value.parts
            # Nothing (·) as left element: behaves as an atop (2-train).
            if left_func is _MISSING:
                if left_value is _MISSING:
                    right_result = _apply_callable(right_func, right_value)
                else:
                    right_result = _apply_callable(right_func, right_value, left_value=left_value)
                return _apply_callable(center_func, right_result)
            # Non-callable left element (subject-left fork): constant left.
            if not _is_callable_value(left_func):
                if left_value is _MISSING:
                    right_result = _apply_callable(right_func, right_value)
                else:
                    right_result = _apply_callable(right_func, right_value, left_value=left_value)
                return _apply_callable(center_func, right_result, left_value=left_func)
            if left_value is _MISSING:
                right_result = _apply_callable(right_func, right_value)
                left_result = _apply_callable(left_func, right_value)
            else:
                right_result = _apply_callable(right_func, right_value, left_value=left_value)
                left_result = _apply_callable(left_func, right_value, left_value=left_value)
            return _apply_callable(center_func, right_result, left_value=left_result)

        raise ValueError("Train currently supports only 2- and 3-train forms")

    if isinstance(func_value, DerivedMod1):
        if func_value.op == "˙":
            return func_value.operand
        if func_value.op == "´":
            if left_value is _MISSING:
                return _fold_callable(func_value.operand, right_value)
            return _fold_callable(func_value.operand, right_value, init=left_value)
        if func_value.op == "`":
            if left_value is _MISSING:
                return _scan_callable(func_value.operand, right_value)
            return _scan_callable(func_value.operand, right_value, init=left_value)
        if func_value.op == "˜":
            if not _is_callable_value(func_value.operand):
                # Non-callable operand: value˜ acts as a constant function.
                return func_value.operand
            if left_value is _MISSING:
                # Self: monadic call x -> x F x.
                return _apply_callable(func_value.operand, right_value, left_value=right_value)
            # Swap: dyadic call w x -> x F w.
            return _apply_callable(func_value.operand, left_value, left_value=right_value)
        if func_value.op == "¨":
            if left_value is _MISSING:
                if isinstance(right_value, list):
                    return _pack_vector([_apply_callable(func_value.operand, item) for item in right_value])
                arr = _as_array(right_value)
                if arr.ndim == 0:
                    return _apply_callable(func_value.operand, right_value)
                return _pack_vector([_apply_callable(func_value.operand, arr[i]) for i in range(arr.shape[0])])

            l_cells = _cells(left_value)
            r_cells = _cells(right_value)
            if len(l_cells) == 1 and len(r_cells) > 1:
                l_cells = l_cells * len(r_cells)
            if len(r_cells) == 1 and len(l_cells) > 1:
                r_cells = r_cells * len(l_cells)
            if len(l_cells) != len(r_cells):
                raise ValueError("¨ dyadic cell lengths are incompatible")
            return _pack_vector(
                [_apply_callable(func_value.operand, r_item, left_value=l_item) for l_item, r_item in zip(l_cells, r_cells, strict=True)]
            )

        if func_value.op == "⌜":
            if left_value is _MISSING:
                # Monadic table behaves like each in this subset.
                return _apply_callable(DerivedMod1(op="¨", operand=func_value.operand), right_value)

            l_cells = _cells(left_value)
            r_cells = _cells(right_value)
            rows: list[object] = []
            for l_item in l_cells:
                rows.append(_pack_vector([_apply_callable(func_value.operand, r_item, left_value=l_item) for r_item in r_cells]))
            return _pack_vector(rows)

        if func_value.op == "˘":
            if left_value is _MISSING:
                if isinstance(right_value, list):
                    return _pack_vector([_apply_callable(func_value.operand, item) for item in right_value])
                arr = _as_array(right_value)
                if arr.ndim <= 1:
                    return _apply_callable(func_value.operand, right_value)
                frame_shape = tuple(int(d) for d in arr.shape[:-1])
                rows = jnp.reshape(arr, (-1, arr.shape[-1]))
                values = [_apply_callable(func_value.operand, rows[i]) for i in range(rows.shape[0])]
                return _reshape_modifier_output(values, frame_shape, where="˘")

            if isinstance(left_value, list) or isinstance(right_value, list):
                l_cells = _cells(left_value)
                r_cells = _cells(right_value)
                if len(l_cells) != len(r_cells):
                    raise ValueError("˘ dyadic list cell lengths are incompatible")
                return _pack_vector(
                    [_apply_callable(func_value.operand, r_item, left_value=l_item) for l_item, r_item in zip(l_cells, r_cells, strict=True)]
                )

            l_arr = _as_array(left_value)
            r_arr = _as_array(right_value)
            if l_arr.ndim == 0 or r_arr.ndim == 0:
                return _apply_callable(func_value.operand, right_value, left_value=left_value)
            if l_arr.shape[:-1] != r_arr.shape[:-1]:
                raise ValueError("˘ dyadic frame shapes must match")
            frame_shape = tuple(int(d) for d in l_arr.shape[:-1])
            l_rows = jnp.reshape(l_arr, (-1, l_arr.shape[-1]))
            r_rows = jnp.reshape(r_arr, (-1, r_arr.shape[-1]))
            values = [_apply_callable(func_value.operand, r_rows[i], left_value=l_rows[i]) for i in range(l_rows.shape[0])]
            return _reshape_modifier_output(values, frame_shape, where="˘")

        if func_value.op == "˝":
            seq = _cells(right_value)
            if not seq:
                return jnp.asarray([], dtype=jnp.int32)
            if left_value is _MISSING:
                acc = seq[0]
                for item in seq[1:]:
                    acc = _apply_callable(func_value.operand, item, left_value=acc)
                return _wrap_insert_result(acc, right_value)
            acc = left_value
            for item in seq:
                acc = _apply_callable(func_value.operand, item, left_value=acc)
            return _wrap_insert_result(acc, right_value)

        if func_value.op == "⁼":
            try:
                if left_value is _MISSING:
                    return _undo_callable(func_value.operand, right_value)
                return _undo_callable(func_value.operand, right_value, left_value=left_value)
            except ValueError:
                if left_value is _MISSING:
                    return _apply_callable(func_value.operand, right_value)
                return _apply_callable(func_value.operand, right_value, left_value=left_value)
        raise ValueError(f"Unsupported 1-modifier derivation {func_value.op!r}")

    if isinstance(func_value, DerivedMod2):
        if func_value.op == "∘":
            if left_value is _MISSING:
                g = _apply_callable(func_value.right, right_value)
            else:
                g = _apply_callable(func_value.right, right_value, left_value=left_value)
            return _apply_callable(func_value.left, g)

        if func_value.op == "○":
            if left_value is _MISSING:
                g = _apply_callable(func_value.right, right_value)
                return _apply_callable(func_value.left, g)
            gx = _apply_callable(func_value.right, right_value)
            gw = _apply_callable(func_value.right, left_value)
            return _apply_callable(func_value.left, gx, left_value=gw)

        if func_value.op == "⊸":
            if _is_callable_value(func_value.left):
                if left_value is _MISSING:
                    fw = _apply_callable(func_value.left, right_value)
                else:
                    fw = _apply_callable(func_value.left, left_value)
            else:
                fw = func_value.left
            return _apply_callable(func_value.right, right_value, left_value=fw)

        if func_value.op == "⟜":
            if _is_callable_value(func_value.right):
                gx = _apply_callable(func_value.right, right_value)
            else:
                gx = func_value.right
            if left_value is _MISSING:
                return _apply_callable(func_value.left, gx, left_value=right_value)
            return _apply_callable(func_value.left, gx, left_value=left_value)

        if func_value.op == "⊘":
            if left_value is _MISSING:
                return _apply_callable(func_value.left, right_value)
            return _apply_callable(func_value.right, right_value, left_value=left_value)

        if func_value.op == "◶":
            if left_value is _MISSING:
                selector = _apply_callable(func_value.left, right_value)
            else:
                selector = _apply_callable(func_value.left, right_value, left_value=left_value)
            chosen = _select(selector, func_value.right)
            if left_value is _MISSING:
                return _apply_callable(chosen, right_value)
            return _apply_callable(chosen, right_value, left_value=left_value)

        if func_value.op == "⌾":
            transformed = _apply_callable(func_value.right, right_value)
            if left_value is _MISSING:
                inner = _apply_callable(func_value.left, transformed)
            else:
                inner = _apply_callable(func_value.left, transformed, left_value=left_value)
            try:
                return _undo_callable(func_value.right, inner)
            except ValueError:
                # Preserve permissive fallback for non-invertible right functions.
                return _apply_callable(func_value.right, inner)

        if func_value.op == "⎉":
            rank_spec = func_value.right
            if _is_callable_value(rank_spec):
                if left_value is _MISSING:
                    rank_spec = _apply_callable(rank_spec, right_value)
                else:
                    rank_spec = _apply_callable(rank_spec, right_value, left_value=left_value)

            ranks = _as_rank_spec_values(rank_spec)
            if left_value is _MISSING:
                if len(ranks) != 1:
                    raise ValueError("⎉ monadic rank spec must be a scalar")
                return _apply_rank_monadic(func_value.left, right_value, rank=ranks[0])

            if len(ranks) == 1:
                left_rank = right_rank = ranks[0]
            else:
                left_rank, right_rank = ranks
            return _apply_rank_dyadic(func_value.left, left_value, right_value, left_rank=left_rank, right_rank=right_rank)

        if func_value.op == "⚇":
            if _is_callable_value(func_value.right):
                depth_value = _apply_callable(func_value.right, right_value)
            else:
                depth_value = func_value.right
            depth = _as_python_int_scalar(depth_value, where="⚇")
            if depth < 0:
                raise ValueError("⚇ depth must be non-negative")
            return _apply_at_depth(func_value.left, right_value, depth=depth, left_value=left_value)

        if func_value.op == "⍟":
            if _is_callable_value(func_value.right):
                if left_value is _MISSING:
                    power_value = _apply_callable(func_value.right, right_value)
                else:
                    power_value = _apply_callable(func_value.right, right_value, left_value=left_value)
            else:
                power_value = func_value.right
            power = _as_python_int_scalar(power_value, where="⍟")

            current = right_value
            steps = abs(power)
            if power >= 0:
                for _ in range(steps):
                    if left_value is _MISSING:
                        current = _apply_callable(func_value.left, current)
                    else:
                        current = _apply_callable(func_value.left, current, left_value=left_value)
                return current

            for _ in range(steps):
                if left_value is _MISSING:
                    current = _undo_callable(func_value.left, current)
                else:
                    current = _undo_callable(func_value.left, current, left_value=left_value)
            return current

        if func_value.op == "⎊":
            try:
                if left_value is _MISSING:
                    return _apply_callable(func_value.left, right_value)
                return _apply_callable(func_value.left, right_value, left_value=left_value)
            except Exception:
                if left_value is _MISSING:
                    return _apply_callable(func_value.right, right_value)
                return _apply_callable(func_value.right, right_value, left_value=left_value)

        raise ValueError(f"Unsupported 2-modifier derivation {func_value.op!r}")

    if isinstance(func_value, PrimitiveFunction):
        if left_value is _MISSING:
            return _eval_prefix(func_value.op, right_value)
        return _eval_infix(func_value.op, left_value, right_value)

    if isinstance(func_value, UserFunction):
        if not func_value.cases:
            return jnp.asarray([], dtype=jnp.int32)

        for index, case in enumerate(func_value.cases):
            call_scope = Scope(parent=func_value.closure)
            call_scope["𝕩"] = right_value
            if left_value is not _MISSING:
                call_scope["𝕨"] = left_value

            if not _case_is_compatible(case, index, func_value.cases, left_value):
                continue
            _bind_case_header(case, call_scope, left_value, right_value)
            if not _case_predicates_pass(case, call_scope):
                continue
            return _evaluate_program(case.body, call_scope)

        raise ValueError("No matching block case for call")

    if callable(func_value):
        if left_value is _MISSING:
            return func_value(right_value)
        return func_value(left_value, right_value)

    # Data as a function is a constant function.
    return func_value


def _eval_expr(expr: Expr, env: Scope) -> object:
    if isinstance(expr, Number):
        return jnp.asarray(expr.value)

    if isinstance(expr, Char):
        return BQNChar(expr.value)

    if isinstance(expr, String):
        return jnp.asarray([ord(ch) for ch in expr.value], dtype=jnp.int32)

    if isinstance(expr, Null):
        return jnp.asarray(0, dtype=jnp.int32)

    if isinstance(expr, Name):
        if expr.value in _PRIMITIVE_FUNCTION_GLYPHS:
            return PrimitiveFunction(op=expr.value)
        # Modifier glyphs (˜, ¨, ∘, ⊸, etc.) used as first-class values.
        if expr.value in {"˙", "˜", "¨", "˘", "⌜", "˝", "⁼", "´", "`",
                          "∘", "○", "⊸", "⟜", "⊘", "◶", "⌾", "⎉", "⚇", "⍟", "⎊"}:
            return PrimitiveFunction(op=expr.value)
        if expr.value in env:
            return env[expr.value]
        if expr.value == "i":
            return jnp.asarray(1j)
        if expr.value.startswith("•"):
            return _resolve_system_name(expr.value)
        raise NameError(f"Undefined name {expr.value!r}")

    if isinstance(expr, Member):
        value = _eval_expr(expr.value, env)
        namespace = _extract_namespace(value, where=".")
        return _lookup_namespace_value(namespace, expr.attr)

    if isinstance(expr, Vector):
        values = [_eval_expr(item, env) for item in expr.items]
        return _pack_vector_literal(values)

    if isinstance(expr, Nothing):
        return _MISSING

    if isinstance(expr, Prefix):
        right = _eval_expr(expr.right, env)
        return _eval_prefix(expr.op, right)

    if isinstance(expr, Infix):
        right = _eval_expr(expr.right, env)
        left = _eval_expr(expr.left, env)
        return _eval_infix(expr.op, left, right)

    if isinstance(expr, Mod1):
        operand = _eval_expr(expr.operand, env)
        return DerivedMod1(op=expr.op, operand=operand)

    if isinstance(expr, Mod2):
        right = _eval_expr(expr.right, env)
        left = _eval_expr(expr.left, env)
        return DerivedMod2(op=expr.op, left=left, right=right)

    if isinstance(expr, Train):
        if len(expr.parts) == 2:
            right = _eval_expr(expr.parts[1], env)
            left = _eval_expr(expr.parts[0], env)
            return DerivedTrain(parts=(left, right))

        if len(expr.parts) == 3:
            right = _eval_expr(expr.parts[2], env)
            center = _eval_expr(expr.parts[1], env)
            left = _eval_expr(expr.parts[0], env)
            return DerivedTrain(parts=(left, center, right))

        raise ValueError("Train currently supports only 2- and 3-train forms")

    if isinstance(expr, Assign):
        right_value = _eval_expr(expr.right, env)
        _assign_target(expr.left, right_value, env, op=expr.op)
        if expr.op == "⇐":
            _export_from_target(expr.left, env)
            env.request_namespace_result()
        return right_value

    if isinstance(expr, Export):
        env.request_namespace_result()
        if expr.target is not None:
            _export_from_target(expr.target, env)
        return Namespace(values=env.exported_values())

    if isinstance(expr, Block):
        return UserFunction(cases=expr.cases, closure=env)

    if isinstance(expr, Call):
        right_value = _eval_expr(expr.right, env)
        func_value = _eval_expr(expr.func, env)
        if expr.left is None:
            return _apply_callable(func_value, right_value)
        left_value = _eval_expr(expr.left, env)
        return _apply_callable(func_value, right_value, left_value=left_value)

    raise TypeError(f"Unsupported expression node: {type(expr)!r}")


def _evaluate_program(program: Program, env: Scope):
    if not program.statements:
        return jnp.asarray([], dtype=jnp.int32)
    result = None
    for stmt in program.statements:
        result = _eval_expr(stmt, env)
        validate_bqn_value(result, where="statement result")
    if env.namespace_requested or env.exports:
        exported = Namespace(values=env.exported_values())
        validate_bqn_value(exported, where="export namespace")
        return exported
    assert result is not None
    validate_bqn_value(result, where="program result")
    return result


def _evaluate_with_scope(source: str, runtime_env: Scope):
    runtime_env.exports.clear()
    runtime_env.namespace_requested = False
    try:
        parsed = _parse_program_cached(source)
        return _evaluate_program(parsed, runtime_env)
    finally:
        runtime_env.exports.clear()
        runtime_env.namespace_requested = False


@overload
def evaluate(source: str) -> object:
    ...


@overload
def evaluate(source: str, env: MutableMapping[str, object]) -> object:
    ...


@overload
def evaluate(source: str, env: EvaluationEnvironment) -> tuple[object, EvaluationEnvironment]:
    ...


@overload
def evaluate(env: EvaluationEnvironment) -> StatefulEvaluate:
    ...


@overload
def evaluate(env: MutableMapping[str, object]) -> StatefulEvaluate:
    ...


def evaluate(
    source_or_env: str | EvaluationEnvironment | MutableMapping[str, object],
    env: MutableMapping[str, object] | EvaluationEnvironment | None = None,
):
    """Parse and evaluate BQN, with optional persistent environment support."""
    if isinstance(source_or_env, str):
        if isinstance(env, EvaluationEnvironment):
            result = _evaluate_with_scope(source_or_env, env._scope)
            return result, env

        seed_env = {} if env is None else dict(env)
        for name, value in seed_env.items():
            validate_bqn_value(value, where=f"env[{name!r}]")
        runtime_env = Scope(data=seed_env)
        return _evaluate_with_scope(source_or_env, runtime_env)

    if env is not None:
        raise TypeError("evaluate(env) form takes exactly one argument")

    if isinstance(source_or_env, EvaluationEnvironment):
        return StatefulEvaluate(source_or_env)

    if isinstance(source_or_env, MutableMapping):
        return StatefulEvaluate(EvaluationEnvironment(source_or_env))

    raise TypeError(
        "evaluate() expects either source text, an EvaluationEnvironment, or a mapping"
    )
