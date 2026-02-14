"""bqn-jax public API."""

from .parser import ParseError, parse, parse_program
from .errors import (
    BQNError,
    BQNParseError,
    BQNRuntimeError,
    BQNShapeError,
    BQNTypeError,
    BQNUnsupportedError,
)

try:
    from .evaluator import EvaluationEnvironment, StatefulEvaluate, evaluate
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("jax"):
        _jax_import_error = exc

        def evaluate(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for evaluate(). Install runtime deps first."
            ) from _jax_import_error

        class EvaluationEnvironment:  # pragma: no cover - import-time fallback
            def __init__(self, *_args, **_kwargs) -> None:
                raise ModuleNotFoundError(
                    "jax is required for EvaluationEnvironment(). Install runtime deps first."
                ) from _jax_import_error

        class StatefulEvaluate:  # pragma: no cover - import-time fallback
            def __init__(self, *_args, **_kwargs) -> None:
                raise ModuleNotFoundError(
                    "jax is required for StatefulEvaluate(). Install runtime deps first."
                ) from _jax_import_error

    else:
        raise

try:
    from .ir import (
        CompiledExpression,
        ShapePolicy,
        cached_grad,
        cached_jit,
        cached_vmap,
        compile_cache_stats,
        compile_expression,
        evaluate_with_errors,
        lower_to_ir,
        prelower_cache_stats,
        transform_helper_cache_stats,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("jax"):
        _jax_ir_import_error = exc

        def lower_to_ir(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for lower_to_ir(). Install runtime deps first."
            ) from _jax_ir_import_error

        def compile_expression(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for compile_expression(). Install runtime deps first."
            ) from _jax_ir_import_error

        def compile_cache_stats(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for compile_cache_stats(). Install runtime deps first."
            ) from _jax_ir_import_error

        def transform_helper_cache_stats(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for transform_helper_cache_stats(). Install runtime deps first."
            ) from _jax_ir_import_error

        def prelower_cache_stats(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for prelower_cache_stats(). Install runtime deps first."
            ) from _jax_ir_import_error

        def cached_jit(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for cached_jit(). Install runtime deps first."
            ) from _jax_ir_import_error

        def cached_vmap(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for cached_vmap(). Install runtime deps first."
            ) from _jax_ir_import_error

        def cached_grad(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for cached_grad(). Install runtime deps first."
            ) from _jax_ir_import_error

        def evaluate_with_errors(*_args, **_kwargs):
            raise ModuleNotFoundError(
                "jax is required for evaluate_with_errors(). Install runtime deps first."
            ) from _jax_ir_import_error

        class ShapePolicy:  # pragma: no cover - import-time fallback
            def __init__(self, *_args, **_kwargs) -> None:
                raise ModuleNotFoundError(
                    "jax is required for ShapePolicy(). Install runtime deps first."
                ) from _jax_ir_import_error

        class CompiledExpression:  # pragma: no cover - import-time fallback
            def __init__(self, *_args, **_kwargs) -> None:
                raise ModuleNotFoundError(
                    "jax is required for CompiledExpression(). Install runtime deps first."
                ) from _jax_ir_import_error

    else:
        raise

__all__ = [
    "parse",
    "parse_program",
    "ParseError",
    "evaluate",
    "EvaluationEnvironment",
    "StatefulEvaluate",
    "lower_to_ir",
    "compile_expression",
    "compile_cache_stats",
    "prelower_cache_stats",
    "transform_helper_cache_stats",
    "cached_jit",
    "cached_vmap",
    "cached_grad",
    "evaluate_with_errors",
    "ShapePolicy",
    "CompiledExpression",
    "BQNError",
    "BQNParseError",
    "BQNRuntimeError",
    "BQNShapeError",
    "BQNTypeError",
    "BQNUnsupportedError",
]
