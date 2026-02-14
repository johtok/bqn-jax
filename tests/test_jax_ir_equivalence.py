from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@dataclass(frozen=True)
class SharedSupportedCase:
    id: str
    source: str
    arg_names: tuple[str, ...] = ()
    args: tuple[object, ...] = ()
    constants: dict[str, object] | None = None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for jax-ir equivalence tests")
class JaxIrEquivalenceTests(unittest.TestCase):
    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def _assert_equivalent(self, source: str, *, arg_names=(), args=(), constants=None) -> None:
        from bqn_jax import compile_expression, evaluate

        compiled = compile_expression(source, arg_names=arg_names, constants=constants)
        compiled_out = compiled(*args)

        env = {name: value for name, value in zip(arg_names, args, strict=True)}
        if constants:
            env.update(constants)
        interpreter_out = evaluate(source, env=env if env else None)

        self.assertEqual(self._to_python(compiled_out), self._to_python(interpreter_out))

    def _shared_supported_cases(self) -> tuple[SharedSupportedCase, ...]:
        import jax.numpy as jnp

        return (
            SharedSupportedCase(
                id="vector_quadratic",
                source="(x × x) + y",
                arg_names=("x", "y"),
                args=(jnp.asarray([1.0, 2.0]), jnp.asarray([10.0, 20.0])),
            ),
            SharedSupportedCase(
                id="reshape_vector",
                source="2‿3 ⥊ x",
                arg_names=("x",),
                args=(jnp.asarray([0, 1, 2, 3, 4, 5]),),
            ),
            SharedSupportedCase(
                id="take_with_fill",
                source="5 ↑ x",
                arg_names=("x",),
                args=(jnp.asarray([1, 2, 3]),),
            ),
            SharedSupportedCase(
                id="drop_negative",
                source="¯2 ↓ x",
                arg_names=("x",),
                args=(jnp.asarray([10, 20, 30, 40]),),
            ),
            SharedSupportedCase(
                id="rotate_vector",
                source="1 ⌽ x",
                arg_names=("x",),
                args=(jnp.asarray([10, 20, 30, 40]),),
            ),
            SharedSupportedCase(
                id="join_vectors",
                source="x ∾ y",
                arg_names=("x", "y"),
                args=(jnp.asarray([1, 2]), jnp.asarray([3, 4])),
            ),
            SharedSupportedCase(
                id="rank_monad",
                source="= x",
                arg_names=("x",),
                args=(jnp.asarray([[1, 2], [3, 4]]),),
            ),
            SharedSupportedCase(
                id="shape_monad",
                source="≢ x",
                arg_names=("x",),
                args=(jnp.asarray([[1, 2], [3, 4]]),),
            ),
            SharedSupportedCase(
                id="conjugate_monad",
                source="+ x",
                arg_names=("x",),
                args=(jnp.asarray([1 + 2j, -3 + 4j]),),
            ),
            SharedSupportedCase(
                id="dyadic_and_lcm",
                source="x ∧ y",
                arg_names=("x", "y"),
                args=(jnp.asarray([4, 6]), jnp.asarray([6, 15])),
            ),
            SharedSupportedCase(
                id="dyadic_or_gcd",
                source="x ∨ y",
                arg_names=("x", "y"),
                args=(jnp.asarray([4, 6]), jnp.asarray([6, 15])),
            ),
            SharedSupportedCase(
                id="fold_reduce",
                source="0 +´ x",
                arg_names=("x",),
                args=(jnp.asarray([1, 2, 3]),),
            ),
            SharedSupportedCase(
                id="train_compose",
                source="+ - x",
                arg_names=("x",),
                args=(jnp.asarray([1 + 2j, -3 + 4j]),),
            ),
            # Shared-supported CBQN imported cases that are lowerable in the JAX backend.
            SharedSupportedCase(id="cbqn_null_subtract", source="@-@"),
            SharedSupportedCase(id="cbqn_depth_vector", source="≡↕10"),
            SharedSupportedCase(id="cbqn_depth_atom", source="≡0"),
        )

    def test_shared_supported_program_equivalence(self) -> None:
        cases = self._shared_supported_cases()
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case.id):
                self._assert_equivalent(
                    case.source,
                    arg_names=case.arg_names,
                    args=case.args,
                    constants=case.constants,
                )


if __name__ == "__main__":
    unittest.main()
