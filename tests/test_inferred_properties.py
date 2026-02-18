from __future__ import annotations

import importlib.util
import math
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for inferred-property tests")
class InferredPropertiesTests(unittest.TestCase):
    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def _assert_close(self, got, want, *, places: int = 6) -> None:
        got_py = self._to_python(got)
        want_py = self._to_python(want)

        def check(g, w):
            if isinstance(w, list):
                self.assertIsInstance(g, list)
                self.assertEqual(len(g), len(w))
                for g_item, w_item in zip(g, w, strict=True):
                    check(g_item, w_item)
                return

            if isinstance(w, (int, float)) and isinstance(g, (int, float)):
                if math.isinf(float(w)) or math.isinf(float(g)):
                    self.assertEqual(math.isinf(float(w)), math.isinf(float(g)))
                    self.assertEqual(math.copysign(1.0, float(w)), math.copysign(1.0, float(g)))
                    return
                self.assertAlmostEqual(float(g), float(w), places=places)
                return

            if isinstance(w, complex) or isinstance(g, complex):
                gw = complex(g)
                ww = complex(w)
                self.assertAlmostEqual(gw.real, ww.real, places=places)
                self.assertAlmostEqual(gw.imag, ww.imag, places=places)
                return

            self.assertEqual(g, w)

        check(got_py, want_py)

    def test_fold_identity_inferred_laws(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("+´ ⟨⟩", "0"),
            ("×´ ⟨⟩", "1"),
            ("∧´ ⟨⟩", "1"),
            ("∨´ ⟨⟩", "0"),
        ]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), evaluate(expected))

    def test_fill_idempotence_for_take(self) -> None:
        from bqn_jax import evaluate

        counts = [0, 1, 2, 5, -1, -2, -5]
        for count in counts:
            with self.subTest(count=count):
                # Use BQN ¯ for negative literals; Python's - would be
                # monadic negate that consumes the entire right expression.
                bqn_count = str(count).replace("-", "¯")
                lhs = evaluate(f"{bqn_count} ↑ ({bqn_count} ↑ 1‿2)")
                rhs = evaluate(f"{bqn_count} ↑ 1‿2")
                self._assert_close(lhs, rhs)

    def test_fill_defaults_for_nudges(self) -> None:
        from bqn_jax import evaluate

        eq_cases = [
            ("» 1‿2‿3", "0 » 1‿2‿3"),
            ("« 1‿2‿3", "0 « 1‿2‿3"),
            ("» (2‿2 ⥊ ↕4)", "0 » (2‿2 ⥊ ↕4)"),
            ("« (2‿2 ⥊ ↕4)", "0 « (2‿2 ⥊ ↕4)"),
        ]
        for lhs_expr, rhs_expr in eq_cases:
            with self.subTest(lhs=lhs_expr, rhs=rhs_expr):
                self._assert_close(evaluate(lhs_expr), evaluate(rhs_expr))

    def test_undo_round_trip_inferred_laws(self) -> None:
        from bqn_jax import evaluate

        monadic_cases = [
            ("(-⁼) (- 2‿¯3‿0)", "2‿¯3‿0"),
            ("(÷⁼) (÷ 2‿4‿0.5)", "2‿4‿0.5"),
            ("(⋆⁼) (⋆ 1‿2‿3)", "1‿2‿3"),
            ("(√⁼) (√ 1‿4‿9)", "1‿4‿9"),
            ("(¬⁼) (¬ 0‿2‿¯1)", "0‿2‿¯1"),
            ("(⌽⁼) (⌽ 1‿2‿3)", "1‿2‿3"),
            ("(⍉⁼) (⍉ (2‿3 ⥊ ↕6))", "2‿3 ⥊ ↕6"),
        ]
        for lhs_expr, rhs_expr in monadic_cases:
            with self.subTest(lhs=lhs_expr):
                self._assert_close(evaluate(lhs_expr), evaluate(rhs_expr))

        dyadic_cases = [
            ("3 (+⁼) (3 + 1‿2‿3)", "1‿2‿3"),
            ("10 (-⁼) (10 - 1‿2‿3)", "1‿2‿3"),
            ("3 (×⁼) (3 × 1‿2‿3)", "1‿2‿3"),
            ("12 (÷⁼) (12 ÷ 2‿3‿4)", "2‿3‿4"),
            ("2 (⋆⁼) (2 ⋆ 1‿2‿3)", "1‿2‿3"),
            ("2 (√⁼) (2 √ 1‿4‿9)", "1‿4‿9"),
            ("2 (⌽⁼) (2 ⌽ 1‿2‿3‿4)", "1‿2‿3‿4"),
            ("1‿0 (⍉⁼) (1‿0 ⍉ (2‿3 ⥊ ↕6))", "2‿3 ⥊ ↕6"),
        ]
        for lhs_expr, rhs_expr in dyadic_cases:
            with self.subTest(lhs=lhs_expr):
                self._assert_close(evaluate(lhs_expr), evaluate(rhs_expr))

    def test_under_matches_explicit_conjugation(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("(+⌾-) 3", "(-⁼) (+ (- 3))"),
            ("2 (+⌾-) 3", "(-⁼) (2 + (- 3))"),
            ("(+⌾÷) 4", "(÷⁼) (+ (÷ 4))"),
            ("2 (+⌾÷) 4", "(÷⁼) (2 + (÷ 4))"),
        ]
        for lhs_expr, rhs_expr in cases:
            with self.subTest(lhs=lhs_expr):
                self._assert_close(evaluate(lhs_expr), evaluate(rhs_expr))

    def test_metamorphic_equivalences(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("⌽⌽ 1‿2‿3", "1‿2‿3"),
            ("⍉⍉ (2‿3 ⥊ ↕6)", "2‿3 ⥊ ↕6"),
            ("+´ (1‿2 ∾ 3‿4)", "(+´ 1‿2) + (+´ 3‿4)"),
            ("×´ (1‿2 ∾ 3‿4)", "(×´ 1‿2) × (×´ 3‿4)"),
            ("∧´ (2‿3 ∾ 4‿6)", "(∧´ 2‿3) ∧ (∧´ 4‿6)"),
            ("∨´ (12‿18 ∾ 6‿9)", "(∨´ 12‿18) ∨ (∨´ 6‿9)"),
            ("(-∘×˜) 5", "- (×˜ 5)"),
        ]
        for lhs_expr, rhs_expr in cases:
            with self.subTest(lhs=lhs_expr, rhs=rhs_expr):
                self._assert_close(evaluate(lhs_expr), evaluate(rhs_expr))


if __name__ == "__main__":
    unittest.main()
