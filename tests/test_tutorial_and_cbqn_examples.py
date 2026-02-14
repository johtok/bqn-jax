from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for interpreter tests")
class TutorialAndCbqnExamplesTests(unittest.TestCase):
    """Examples sourced from the BQN tutorial and a compatible CBQN test subset."""

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

            if isinstance(w, complex):
                self.assertAlmostEqual(complex(g).real, w.real, places=places)
                self.assertAlmostEqual(complex(g).imag, w.imag, places=places)
                return

            if isinstance(w, (int, float)):
                self.assertAlmostEqual(float(g), float(w), places=places)
                return

            self.assertEqual(g, w)

        check(got_py, want_py)

    def test_tutorial_expression_and_combinator_examples(self) -> None:
        from bqn_jax import evaluate

        # Sources:
        # - https://mlochbaum.github.io/BQN/tutorial/expression.html
        # - https://mlochbaum.github.io/BQN/tutorial/combinator.html
        cases = [
            ("2 + 3", 5.0),
            ("6-   5", 1.0),
            ("- 1.5", -1.5),
            ("2 × π", 6.283185307179586),
            ("9 ÷ 2", 4.5),
            ("÷ ∞", 0.0),
            ("2 ⋆ 3", 8.0),
            ("3 ⋆ 2", 9.0),
            ("√ 2", 1.4142135623730951),
            ("3 √ 27", 3.0),
            ("2×3 - 5", -4.0),
            ("(2×3) - 5", 1.0),
            ("(4÷3) × π × 2⋆3", 33.51032257080078),
            ("'*' - @", 42.0),
            ("@ + 97", 97.0),
            ("2 -˜ 'd'", 98.0),
            ("+˜ 3", 6.0),
            ("×˜ 5", 25.0),
            ("2 ⋆˜ 5", 25.0),
            ("√⁼ 5", 25.0),
            ("2 ⋆⁼ 32", 5.0),
            ("2 ⋆ 2 ⋆⁼ 32", 32.0),
            ("10 ⋆⁼ 1e4", 4.0),
            ("2 3˙ 4", 3.0),
            ("3 ×˜∘+ 4", 49.0),
            ("-∘(×˜) 5", -25.0),
            ("3 < 4", 1),
            ("4 > ∞", 0),
            ('"abcd" ≡ "abdd"', 0),
            ('"abc"‿"de" ≡ "abc"‿"de"', 1),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)

    def test_tutorial_list_and_variable_examples(self) -> None:
        from bqn_jax import evaluate

        # Sources:
        # - https://mlochbaum.github.io/BQN/tutorial/list.html
        # - https://mlochbaum.github.io/BQN/tutorial/variable.html
        cases = [
            ("0‿1‿2", [0.0, 1.0, 2.0]),
            ("(0‿1)‿2", [[0.0, 1.0], 2.0]),
            ("0‿(1‿2)", [0.0, [1.0, 2.0]]),
            ("2 + 1‿2‿3", [3.0, 4.0, 5.0]),
            ("2 × ⟨0‿2 ⋄ 1‿3‿5⟩", [[0.0, 4.0], [2.0, 6.0, 10.0]]),
            ("⟨ 10, 20‿30 ⟩ + ⟨ 1‿2, 3 ⟩", [[11.0, 12.0], [23.0, 33.0]]),
            ("2 ⌽ ⟨0,1,2,3,4⟩", [2.0, 3.0, 4.0, 0.0, 1.0]),
            ("+´ 2‿3‿4", 9.0),
            ("×´ 2‿3‿4", 24.0),
            ("-´ 1‿2‿3‿4‿5", 3.0),
            ("↕7", [0, 1, 2, 3, 4, 5, 6]),
            ("4 ↑ ↕7", [0, 1, 2, 3]),
            ('hey ← "Hi there" ⋄ hey ∾ ", World!"', [72, 105, 32, 116, 104, 101, 114, 101, 44, 32, 87, 111, 114, 108, 100, 33]),
            ("three ← 3 ⋄ three ↩ 4 ⋄ three", 4.0),
            ('1 ⊑ "BQN"', 81.0),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)

    def test_cbqn_prims_subset_examples(self) -> None:
        from bqn_jax import evaluate

        # Source: https://github.com/dzaima/CBQN/tree/develop/test/cases
        # Cases from prims.bqn that are directly compatible with this runtime.
        cases = [
            ("(<1)+ 1", "<2"),
            ("1 +<1", "<2"),
            ("(<1)+<1", "<2"),
            ("@-@", "0"),
            ("2‿3‿0‿1/↕4", "6⥊0‿0‿1‿1‿1‿3"),
            (
                "⟨1‿¯1, 2‿2⥊3‿2‿¯3‿¯4⟩ ⊏ 2‿4‿3⥊↕24",
                "2‿2‿2‿3⥊21‿22‿23‿18‿19‿20‿15‿16‿17‿12‿13‿14‿21‿22‿23‿18‿19‿20‿15‿16‿17‿12‿13‿14",
            ),
            ("≡¨ ⟨⟨⟩, ↕0, \"\"⟩", "1‿1‿1"),
            ("≡⟨1,1‿2,1⟩", "2"),
            ("≡↕10", "1"),
            ("≡0", "0"),
            ("⟨⟩⍷1‿0⥊\"\"", "1‿1⥊1"),
            (
                "1‿2‿2‿3‿3‿2‿2‿1⊒1‿8‿5‿5‿6‿2‿5‿3‿1‿5‿1‿2‿4‿3‿4‿3‿1‿5‿2‿3‿4‿1‿5‿2",
                "0‿8‿8‿8‿8‿1‿8‿3‿7‿8‿8‿2‿8‿4‿8‿8‿8‿8‿5‿8‿8‿8‿8‿6",
            ),
        ]

        for expr, expected_expr in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), evaluate(expected_expr))


if __name__ == "__main__":
    unittest.main()
