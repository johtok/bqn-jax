"""Structured error types for parser/runtime separation."""

from __future__ import annotations

from dataclasses import dataclass

from .parser import ParseError


class BQNError(Exception):
    """Base class for structured bqn-jax errors."""


@dataclass(frozen=True)
class BQNParseError(BQNError):
    """Wraps parser failures with explicit parse-stage typing."""

    message: str
    start: int
    end: int
    expected: tuple[str, ...] = ()
    found: str | None = None

    @classmethod
    def from_parse_error(cls, err: ParseError) -> "BQNParseError":
        return cls(
            message=err.message,
            start=err.start,
            end=err.end,
            expected=err.expected,
            found=err.found,
        )

    def __str__(self) -> str:
        expected = ""
        if self.expected:
            expected = f"; expected {', '.join(self.expected)}"
        found = ""
        if self.found is not None:
            found = f"; found {self.found}"
        return f"{self.message} at span [{self.start}, {self.end}){expected}{found}"


class BQNRuntimeError(BQNError):
    """Generic runtime failure after successful parse."""


class BQNShapeError(BQNRuntimeError):
    """Runtime shape/rank/axis compatibility failure."""


class BQNTypeError(BQNRuntimeError):
    """Runtime type/value-kind compatibility failure."""


class BQNUnsupportedError(BQNRuntimeError):
    """Feature exists in the language but is not supported in this execution path."""


def classify_runtime_exception(err: Exception) -> BQNRuntimeError:
    """Best-effort runtime error classification for structured APIs."""
    message = str(err)
    lowered = message.lower()

    shape_markers = (
        "shape",
        "rank",
        "frame",
        "axis",
        "length",
        "reshape",
        "out-of-bounds",
        "bounds",
        "indexing",
        "indices",
    )
    if any(marker in lowered for marker in shape_markers):
        return BQNShapeError(message)

    type_markers = (
        "type",
        "integer",
        "number",
        "callable",
        "undefined",
        "non-",
        "cannot be called",
        "must be",
        "unexpected argument types",
    )
    if any(marker in lowered for marker in type_markers):
        return BQNTypeError(message)

    return BQNRuntimeError(message)
