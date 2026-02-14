"""Runtime value model and validators for the BQN evaluator subset."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


class BoxedArray(list):
    """Explicit container type for boxed arrays."""


@dataclass(frozen=True)
class BQNChar:
    """First-class character value."""

    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 1:
            raise ValueError("BQNChar must contain exactly one codepoint")

    @property
    def codepoint(self) -> int:
        return ord(self.value)


class ValueKind(str, Enum):
    ATOM = "atom"
    ARRAY = "array"
    BOXED_ARRAY = "boxed_array"
    CHARACTER = "character"
    NAMESPACE = "namespace"
    OPERATION = "operation"


@dataclass(frozen=True)
class ValueInfo:
    kind: ValueKind
    shape: tuple[int, ...]
    rank: int
    depth: int


@dataclass(frozen=True)
class OperationInfo:
    kind: str
    symbol: str | None
    valence: str


def is_namespace(value: object) -> bool:
    return bool(getattr(value, "_bqn_namespace", False))


def is_operation(value: object) -> bool:
    return bool(getattr(value, "_bqn_operation_kind", None)) or callable(value)


def operation_info(value: object) -> OperationInfo | None:
    if hasattr(value, "info"):
        info = getattr(value, "info")
        if isinstance(info, OperationInfo):
            return info

    marker = getattr(value, "_bqn_operation_kind", None)
    if marker is not None:
        symbol = getattr(value, "op", None)
        if not isinstance(symbol, str):
            symbol = None
        return OperationInfo(kind=str(marker), symbol=symbol, valence="ambivalent")

    if callable(value):
        symbol = getattr(value, "__name__", None)
        if not isinstance(symbol, str):
            symbol = None
        return OperationInfo(kind="callable", symbol=symbol, valence="ambivalent")
    return None


def is_boxed_array(value: object) -> bool:
    return isinstance(value, (BoxedArray, list))


def as_jax_array(value: object):
    if isinstance(value, jnp.ndarray):
        return value
    if isinstance(value, BQNChar):
        return jnp.asarray(value.codepoint, dtype=jnp.int32)
    return jnp.asarray(value)


def shape_of(value: object) -> tuple[int, ...]:
    if isinstance(value, BQNChar):
        return ()
    if is_boxed_array(value):
        return (len(value),)
    if is_namespace(value) or is_operation(value):
        return ()
    arr = as_jax_array(value)
    return tuple(int(d) for d in arr.shape)


def rank_of(value: object) -> int:
    return len(shape_of(value))


def depth_of(value: object) -> int:
    if isinstance(value, BQNChar):
        return 0
    if is_boxed_array(value):
        if not value:
            return 1
        return 1 + max(depth_of(item) for item in value)
    if is_namespace(value) or is_operation(value):
        return 0
    arr = as_jax_array(value)
    return 0 if arr.ndim == 0 else 1


def kind_of(value: object) -> ValueKind:
    if isinstance(value, BQNChar):
        return ValueKind.CHARACTER
    if is_boxed_array(value):
        return ValueKind.BOXED_ARRAY
    if is_namespace(value):
        return ValueKind.NAMESPACE
    if is_operation(value):
        return ValueKind.OPERATION
    arr = as_jax_array(value)
    if arr.ndim == 0:
        return ValueKind.ATOM
    return ValueKind.ARRAY


def value_info(value: object) -> ValueInfo:
    kind = kind_of(value)
    shape = shape_of(value)
    return ValueInfo(kind=kind, shape=shape, rank=len(shape), depth=depth_of(value))


def zero_like(value):
    if isinstance(value, BQNChar):
        return BQNChar("\x00")
    if is_boxed_array(value):
        if not value:
            return jnp.asarray(0)
        return BoxedArray([zero_like(item) for item in value])

    arr = as_jax_array(value)
    return jnp.zeros(arr.shape, dtype=arr.dtype)


def validate_value(value: object, *, where: str = "value") -> None:
    if isinstance(value, BQNChar):
        return
    if is_boxed_array(value):
        for idx, item in enumerate(value):
            validate_value(item, where=f"{where}[{idx}]")
        return
    if is_namespace(value) or is_operation(value):
        return
    if isinstance(value, jnp.ndarray):
        return
    if isinstance(value, numbers.Number):
        return
    if isinstance(value, bool):
        return
    raise TypeError(f"{where} has unsupported runtime type {type(value).__name__}")
