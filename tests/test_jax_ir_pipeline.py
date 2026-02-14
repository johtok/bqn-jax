from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for jax-ir pipeline tests")
class JaxIrPipelineTests(unittest.TestCase):
    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def test_lower_to_ir_basic(self) -> None:
        from bqn_jax import lower_to_ir

        ir = lower_to_ir("(x × x) + y", arg_names=("x", "y"))
        self.assertEqual(ir.arg_names, ("x", "y"))
        self.assertGreaterEqual(len(ir.nodes), 4)
        self.assertEqual(ir.nodes[ir.output].op, "infix:+")

    def test_lower_to_ir_cse_reuses_repeated_subexpressions(self) -> None:
        from bqn_jax import lower_to_ir

        ir = lower_to_ir("(x × x) + (x × x)", arg_names=("x",))
        mul_nodes = [node for node in ir.nodes if node.op == "infix:×"]
        self.assertEqual(len(mul_nodes), 1)
        out = ir.nodes[ir.output]
        self.assertEqual(out.op, "infix:+")
        self.assertEqual(out.inputs[0], mul_nodes[0].id)
        self.assertEqual(out.inputs[1], mul_nodes[0].id)

    def test_rewrite_implicit_train_and_modifier_calls(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        self.assertAlmostEqual(
            float(compile_expression("F x", arg_names=("x",), constants={"F": "-"})(3.0)),
            -3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("F G x", arg_names=("x",), constants={"F": "+", "G": "-"})(3.0)),
            -3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("w F x", arg_names=("w", "x"), constants={"F": "+"})(2.0, 5.0)),
            7.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("F∘G x", arg_names=("x",), constants={"F": "+", "G": "-"})(3.0)),
            -3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("x F○G y", arg_names=("x", "y"), constants={"F": "+", "G": "-"})(2.0, 5.0)),
            -7.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("F⊸G x", arg_names=("x",), constants={"F": "-", "G": "+"})(5.0)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("F⟜G x", arg_names=("x",), constants={"F": "+", "G": "-"})(5.0)),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(compile_expression("F⊘G x", arg_names=("x",), constants={"F": "+", "G": "-"})(5.0)),
            5.0,
            places=6,
        )
        folded = compile_expression("+´ x", arg_names=("x",))
        self.assertAlmostEqual(float(folded(jnp.asarray([1.0, 2.0, 3.0]))), 6.0, places=6)

    def test_compile_expression_and_trace(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        compiled, jaxpr = compile_expression(
            "(x × x) + y",
            arg_names=("x", "y"),
            trace=True,
            trace_args=(jnp.asarray([1.0, 2.0]), jnp.asarray([10.0, 20.0])),
        )
        out = compiled(jnp.asarray([1.0, 2.0]), jnp.asarray([10.0, 20.0]))
        self.assertEqual(self._to_python(out), [11.0, 24.0])
        self.assertIn("mul", str(jaxpr))
        self.assertIn("add", str(jaxpr))

    def test_jit_grad_vmap_with_shape_policy(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import ShapePolicy, compile_expression
        from bqn_jax.errors import BQNShapeError

        scalar = compile_expression("(x × x) + 1", arg_names=("x",))
        grad_fn = scalar.grad()
        self.assertAlmostEqual(float(grad_fn(3.0)), 6.0, places=6)

        vmap_fn = scalar.vmap()
        vmapped = vmap_fn(jnp.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(self._to_python(vmapped), [2.0, 5.0, 10.0])

        static = compile_expression("x + 1", arg_names=("x",), shape_policy=ShapePolicy(kind="static"))
        jit_fn = static.jit()
        first = jit_fn(jnp.asarray([1.0, 2.0]))
        self.assertEqual(self._to_python(first), [2.0, 3.0])
        with self.assertRaises(BQNShapeError):
            jit_fn(jnp.asarray([1.0, 2.0, 3.0]))

    def test_autodiff_multi_argument_argnums(self) -> None:
        from bqn_jax import compile_expression

        compiled = compile_expression("(x × x) + (2 × (y × y))", arg_names=("x", "y"))
        grad_x = compiled.grad(argnums=0)
        grad_y = compiled.grad(argnums=1)

        self.assertAlmostEqual(float(grad_x(3.0, 4.0)), 6.0, places=6)
        self.assertAlmostEqual(float(grad_y(3.0, 4.0)), 16.0, places=6)

    def test_autodiff_higher_order_derivative(self) -> None:
        import jax

        from bqn_jax import compile_expression

        compiled = compile_expression("x ⋆ 3", arg_names=("x",))
        first = compiled.grad()
        second = jax.grad(first)

        self.assertAlmostEqual(float(first(5.0)), 75.0, places=5)
        self.assertAlmostEqual(float(second(5.0)), 30.0, places=5)

    def test_autodiff_reduction_gradient(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        compiled = compile_expression("0 +´ x", arg_names=("x",))
        grad_fn = compiled.grad()
        out = grad_fn(jnp.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(self._to_python(out), [1.0, 1.0, 1.0])

    def test_autodiff_shape_policy_enforced(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import ShapePolicy, compile_expression
        from bqn_jax.errors import BQNShapeError

        compiled = compile_expression(
            "(x × x) + 1",
            arg_names=("x",),
            shape_policy=ShapePolicy(kind="static"),
        )
        grad_fn = compiled.grad()
        self.assertAlmostEqual(float(grad_fn(jnp.asarray(3.0))), 6.0, places=6)
        with self.assertRaises(BQNShapeError):
            grad_fn(jnp.asarray([1.0, 2.0]))

    def test_autodiff_requires_scalar_output(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression
        from bqn_jax.errors import BQNShapeError

        compiled = compile_expression("x + 1", arg_names=("x",))
        grad_fn = compiled.grad()
        with self.assertRaises(BQNShapeError):
            grad_fn(jnp.asarray([1.0, 2.0, 3.0]))

    def test_parse_vs_runtime_error_separation(self) -> None:
        from bqn_jax import evaluate_with_errors
        from bqn_jax.errors import BQNParseError, BQNShapeError, BQNTypeError

        with self.assertRaises(BQNParseError):
            evaluate_with_errors("1 + )")

        with self.assertRaises(BQNShapeError):
            evaluate_with_errors("1‿2 + 1‿2‿3")

        with self.assertRaises(BQNTypeError):
            evaluate_with_errors("↕ 1‿2")

    def test_fold_lowering_and_execution(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        folded_literal = compile_expression("+´ 1‿2‿3‿4")
        self.assertEqual(float(folded_literal()), 10.0)

        folded_arg = compile_expression("0 +´ x", arg_names=("x",))
        mat = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
        out = folded_arg(mat)
        self.assertEqual(self._to_python(out), [4.0, 6.0])

    def test_fold_init_identity_canonicalizes_to_plain_fold(self) -> None:
        from bqn_jax import lower_to_ir

        ir = lower_to_ir("0 +´ x", arg_names=("x",), use_cache=False)
        self.assertEqual(ir.nodes[ir.output].op, "fold:+")

    def test_expanded_array_lowering_semantics(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        shaped = compile_expression("2‿3 ⥊ ↕6")
        self.assertEqual(self._to_python(shaped()), [[0, 1, 2], [3, 4, 5]])

        take = compile_expression("5 ↑ x", arg_names=("x",))
        drop = compile_expression("¯2 ↓ x", arg_names=("x",))
        rotate = compile_expression("1 ⌽ x", arg_names=("x",))
        self.assertEqual(self._to_python(take(jnp.asarray([10, 20, 30]))), [10, 20, 30, 0, 0])
        self.assertEqual(self._to_python(drop(jnp.asarray([10, 20, 30, 40]))), [10, 20])
        self.assertEqual(self._to_python(rotate(jnp.asarray([10, 20, 30, 40]))), [20, 30, 40, 10])

        transpose = compile_expression("1 ⍉ (2‿3 ⥊ ↕6)")
        self.assertEqual(self._to_python(transpose()), [[0, 3], [1, 4], [2, 5]])

        concat = compile_expression("x ∾ y", arg_names=("x", "y"))
        self.assertEqual(self._to_python(concat(jnp.asarray([1, 2]), jnp.asarray([3, 4]))), [1, 2, 3, 4])

        rank = compile_expression("= x", arg_names=("x",))
        shape = compile_expression("≢ x", arg_names=("x",))
        self.assertEqual(int(rank(jnp.asarray([[1, 2], [3, 4]]))), 2)
        self.assertEqual(self._to_python(shape(jnp.asarray([[1, 2], [3, 4]]))), [2, 2])

        conjugate = compile_expression("+ x", arg_names=("x",))
        self.assertEqual(self._to_python(conjugate(jnp.asarray([1 + 2j]))), [(1 - 2j)])

        lcm = compile_expression("x ∧ y", arg_names=("x", "y"))
        gcd = compile_expression("x ∨ y", arg_names=("x", "y"))
        self.assertEqual(self._to_python(lcm(jnp.asarray([4, 6]), jnp.asarray([6, 15]))), [12, 30])
        self.assertEqual(self._to_python(gcd(jnp.asarray([4, 6]), jnp.asarray([6, 15]))), [2, 3])

    def test_lowering_constants_and_system_values(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        with_const = compile_expression("x + c", arg_names=("x",), constants={"c": 2.5})
        self.assertAlmostEqual(float(with_const(3.5)), 6.0, places=6)

        with_system = compile_expression("•pi + x", arg_names=("x",))
        self.assertAlmostEqual(float(with_system(jnp.asarray(1.0))), 4.1415926, places=5)

    def test_keyword_argument_contracts(self) -> None:
        from bqn_jax import compile_expression
        from bqn_jax.errors import BQNTypeError

        compiled = compile_expression("(x × x) + y", arg_names=("x", "y"))
        self.assertEqual(float(compiled(x=3.0, y=2.0)), 11.0)

        with self.assertRaises(BQNTypeError):
            compiled(x=3.0)
        with self.assertRaises(BQNTypeError):
            compiled(x=3.0, y=2.0, z=1.0)
        with self.assertRaises(BQNTypeError):
            compiled(3.0, y=2.0)

    def test_dynamic_and_static_shape_policy_on_plain_calls(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import ShapePolicy, compile_expression
        from bqn_jax.errors import BQNShapeError

        dynamic = compile_expression("x + 1", arg_names=("x",), shape_policy=ShapePolicy(kind="dynamic"))
        self.assertEqual(self._to_python(dynamic(jnp.asarray([1.0, 2.0]))), [2.0, 3.0])
        self.assertEqual(self._to_python(dynamic(jnp.asarray([1.0, 2.0, 3.0]))), [2.0, 3.0, 4.0])

        static = compile_expression("x + 1", arg_names=("x",), shape_policy=ShapePolicy(kind="static"))
        self.assertEqual(self._to_python(static(jnp.asarray([1.0, 2.0]))), [2.0, 3.0])
        with self.assertRaises(BQNShapeError):
            static(jnp.asarray([1.0, 2.0, 3.0]))

    def test_vmap_custom_in_axes(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        compiled = compile_expression("x + y", arg_names=("x", "y"))
        vmapped = compiled.vmap(in_axes=(0, None))
        out = vmapped(jnp.asarray([1.0, 2.0, 3.0]), 10.0)
        self.assertEqual(self._to_python(out), [11.0, 12.0, 13.0])

    def test_vmap_honors_static_shape_policy(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import ShapePolicy, compile_expression
        from bqn_jax.errors import BQNShapeError

        compiled = compile_expression("x + 1", arg_names=("x",), shape_policy=ShapePolicy(kind="static"))
        vmapped = compiled.vmap()
        self.assertEqual(self._to_python(vmapped(jnp.asarray([1.0, 2.0]))), [2.0, 3.0])
        with self.assertRaises(BQNShapeError):
            vmapped(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    def test_ir_optimization_rewrites_identity_arithmetic(self) -> None:
        from bqn_jax import lower_to_ir

        plus = lower_to_ir("x + 0", arg_names=("x",))
        mul = lower_to_ir("1 × x", arg_names=("x",))
        square = lower_to_ir("x ⋆ 2", arg_names=("x",))
        self.assertEqual(plus.nodes[plus.output].op, "arg")
        self.assertEqual(mul.nodes[mul.output].op, "arg")
        self.assertEqual(square.nodes[square.output].op, "infix:×")

    def test_dynamic_power_exponent_keeps_runtime_semantics(self) -> None:
        import jax.numpy as jnp

        from bqn_jax import compile_expression

        compiled = compile_expression("x ⋆ e", arg_names=("x", "e"))
        x = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)
        self.assertEqual(self._to_python(compiled(x, 2)), [4.0, 9.0, 16.0])
        self.assertEqual(self._to_python(compiled(x, 3)), [8.0, 27.0, 64.0])
        reciprocal = self._to_python(compiled(x, -1.0))
        self.assertAlmostEqual(float(reciprocal[0]), 0.5, places=6)
        self.assertAlmostEqual(float(reciprocal[1]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(reciprocal[2]), 0.25, places=6)

    def test_compile_cache_statistics(self) -> None:
        from bqn_jax import compile_cache_stats, compile_expression

        compile_cache_stats(reset=True)
        compile_expression("x + 1", arg_names=("x",))
        compile_expression("x + 1", arg_names=("x",))
        stats = compile_cache_stats()
        self.assertGreaterEqual(int(stats["misses"]), 1)
        self.assertGreaterEqual(int(stats["hits"]), 1)

    def test_prelower_cache_statistics(self) -> None:
        from bqn_jax import lower_to_ir, prelower_cache_stats

        prelower_cache_stats(reset=True)
        lower_to_ir("0 +´ x", arg_names=("x",), use_cache=False)
        lower_to_ir("0 +´ x", arg_names=("x",), use_cache=False)
        stats = prelower_cache_stats()
        self.assertGreaterEqual(int(stats["misses"]), 1)
        self.assertGreaterEqual(int(stats["hits"]), 1)

    def test_transforms_are_cached_on_compiled_expression(self) -> None:
        from bqn_jax import compile_expression

        compiled = compile_expression("x + 1", arg_names=("x",))
        j1 = compiled.jit()
        j2 = compiled.jit()
        self.assertIs(j1, j2)
        v1 = compiled.vmap()
        v2 = compiled.vmap()
        self.assertIs(v1, v2)
        g1 = compiled.grad()
        g2 = compiled.grad()
        self.assertIs(g1, g2)
        stats = compiled.transform_cache_stats()
        self.assertGreaterEqual(int(stats["total_hits"]), 3)

    def test_global_cached_helpers(self) -> None:
        from bqn_jax import cached_grad, cached_jit, cached_vmap, transform_helper_cache_stats

        transform_helper_cache_stats(reset=True)
        j1 = cached_jit("x + 1", arg_names=("x",))
        j2 = cached_jit("x + 1", arg_names=("x",))
        self.assertIs(j1, j2)
        v1 = cached_vmap("x + 1", arg_names=("x",))
        v2 = cached_vmap("x + 1", arg_names=("x",))
        self.assertIs(v1, v2)
        g1 = cached_grad("x + 1", arg_names=("x",))
        g2 = cached_grad("x + 1", arg_names=("x",))
        self.assertIs(g1, g2)
        stats = transform_helper_cache_stats()
        self.assertGreaterEqual(int(stats["hits"]), 3)

    def test_runtime_only_feature_errors_are_explicit(self) -> None:
        from bqn_jax import lower_to_ir
        from bqn_jax.errors import BQNUnsupportedError

        with self.assertRaises(BQNUnsupportedError) as block_err:
            lower_to_ir("{𝕩 + 1}")
        self.assertIn("Runtime-only feature", str(block_err.exception))

        with self.assertRaises(BQNUnsupportedError) as assign_err:
            lower_to_ir("x ← 1")
        self.assertIn("Runtime-only feature", str(assign_err.exception))

        with self.assertRaises(BQNUnsupportedError) as member_err:
            lower_to_ir("ns.a", arg_names=("ns",))
        self.assertIn("Runtime-only feature", str(member_err.exception))

    def test_ir_error_paths(self) -> None:
        from bqn_jax import compile_expression, lower_to_ir
        from bqn_jax.errors import BQNParseError, BQNTypeError, BQNUnsupportedError

        with self.assertRaises(BQNParseError):
            lower_to_ir("1 + )")

        with self.assertRaises(BQNUnsupportedError):
            lower_to_ir("{𝕩 + 1}")

        with self.assertRaises(BQNTypeError):
            compile_expression("x + 1", arg_names=("x",), trace=True)


if __name__ == "__main__":
    unittest.main()
