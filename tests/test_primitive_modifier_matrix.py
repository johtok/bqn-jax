from __future__ import annotations

import importlib.util
import unittest

from primitive_matrix_cases import (
    ALL_MATRIX_CASES,
    DYADIC_PRIMITIVES,
    MOD1_GLYPHS,
    MOD2_GLYPHS,
    MONADIC_PRIMITIVES,
    MODIFIER_MATRIX_CASES,
    PRIMITIVE_MATRIX_CASES,
)


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for primitive matrix tests")
class PrimitiveModifierMatrixTests(unittest.TestCase):
    def test_matrix_covers_declared_surface(self) -> None:
        from bqn_jax import parser as parser_mod

        matrix_monadic = {case.symbol for case in PRIMITIVE_MATRIX_CASES if case.valence == "monadic"}
        matrix_dyadic = {case.symbol for case in PRIMITIVE_MATRIX_CASES if case.valence == "dyadic"}
        matrix_mod1 = {case.symbol for case in MODIFIER_MATRIX_CASES if case.valence == "derived1"}
        matrix_mod2 = {case.symbol for case in MODIFIER_MATRIX_CASES if case.valence == "derived2"}

        self.assertEqual(matrix_monadic, MONADIC_PRIMITIVES)
        self.assertEqual(matrix_dyadic, DYADIC_PRIMITIVES)
        self.assertEqual(matrix_mod1, MOD1_GLYPHS)
        self.assertEqual(matrix_mod2, MOD2_GLYPHS)
        self.assertEqual(matrix_monadic, parser_mod._MONADIC_OPS)
        self.assertEqual(matrix_dyadic, parser_mod._DYADIC_OPS)
        self.assertEqual(matrix_mod1, parser_mod._MOD1_OPS)
        self.assertEqual(matrix_mod2, parser_mod._MOD2_OPS)

    def test_matrix_cases_execute(self) -> None:
        from bqn_jax import evaluate

        for case in ALL_MATRIX_CASES:
            with self.subTest(symbol=case.symbol, valence=case.valence, arg_class=case.arg_class):
                evaluate(case.expr)

    def test_assert_primitive_messages(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaisesRegex(ValueError, "Assertion error"):
            evaluate("! 0")
        with self.assertRaisesRegex(ValueError, "boom"):
            evaluate('"boom" ! 0')

    def test_dyadic_equality_is_elementwise(self) -> None:
        from bqn_jax import evaluate

        eq = evaluate("1‿2 = 1‿3")
        ne = evaluate("1‿2 ≠ 1‿3")
        self.assertEqual(list(eq.tolist()), [1, 0])
        self.assertEqual(list(ne.tolist()), [0, 1])

    def test_dyadic_and_or_numeric_behavior(self) -> None:
        from bqn_jax import evaluate

        conj = evaluate("1‿0‿2 ∧ 0‿4‿5")
        disj = evaluate("1‿0‿0 ∨ 0‿0‿2")
        self.assertEqual(list(conj.tolist()), [0, 0, 10])
        self.assertEqual(list(disj.tolist()), [1, 0, 2])

    def test_bins_include_equal_keys(self) -> None:
        from bqn_jax import evaluate

        up = evaluate("2‿4‿6 ⍋ 1‿2‿3‿7")
        down = evaluate("6‿4‿2 ⍒ 7‿6‿5‿1")
        self.assertEqual(list(up.tolist()), [0, 1, 1, 3])
        self.assertEqual(list(down.tolist()), [0, 1, 1, 3])

    def test_classify_uses_dense_class_numbers(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("⊐ 3‿1‿3‿2‿1")
        self.assertEqual(list(out.tolist()), [0, 1, 0, 2, 1])

    def test_insert_preserves_leading_axis(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("+˝ 1‿2‿3")
        self.assertEqual(list(out.tolist()), [6.0])

    def test_monadic_merge_rejects_incompatible_shapes(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("> ⟨1⋄2‿3⟩")


if __name__ == "__main__":
    unittest.main()
