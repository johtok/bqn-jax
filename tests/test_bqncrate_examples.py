from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for interpreter tests")
class BqnCrateExamplesTests(unittest.TestCase):
    """Subset of expressions sourced from https://mlochbaum.github.io/bqncrate/table.tsv."""

    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def _assert_close(self, got, want, *, places: int = 6) -> None:
        got_py = self._to_python(got)

        def check(G, w):
            if isinstance(w, list):
                self.assertIsInstance(G, list)
                self.assertEqual(len(G), len(w))
                for g_item, w_item in zip(G, w, strict=True):
                    check(g_item, w_item)
                return

            if isinstance(w, complex):
                self.assertAlmostEqual(complex(G).real, w.real, places=places)
                self.assertAlmostEqual(complex(G).imag, w.imag, places=places)
                return

            if isinstance(w, (int, float)):
                self.assertAlmostEqual(float(G), float(w), places=places)
                return

            self.assertEqual(G, w)

        check(got_py, want)

    def test_bqncrate_arithmetic_primitives(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("+ 3", 3.0),
            ("2 + 3", 5.0),
            ("- 3", -3.0),
            ("7 - 2", 5.0),
            ("× ¯2‿0‿3", [-1.0, 0.0, 1.0]),
            ("6 × 7", 42.0),
            ("⌈ 1.2", 2.0),
            ("2 ⌈ 5", 5.0),
            ("⌊ 1.8", 1.0),
            ("2 ⌊ 5", 2.0),
            ("÷ 4", 0.25),
            ("8 ÷ 2", 4.0),
            ("2 ⋆ 3", 8.0),
            ("√ 9", 3.0),
            ("2 √ 9", 3.0),
            ("| ¯5", 5.0),
            ("3 | 10", 1.0),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)

    def test_bqncrate_structural_and_array_primitives(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("= (2‿3 ⥊ ↕6)", 2),
            ("≠ 10‿20‿30", 3),
            ("≢ (2‿3 ⥊ ↕6)", [2, 3]),
            ("↕ 5", [0, 1, 2, 3, 4]),
            ("1‿2 ∾ 3‿4", [1.0, 2.0, 3.0, 4.0]),
            ("2 ↑ 10‿20‿30‿40", [10.0, 20.0]),
            ("5 ↑ 10‿20‿30", [10.0, 20.0, 30.0, 0.0, 0.0]),
            ("2 ↓ 10‿20‿30‿40", [30.0, 40.0]),
            ("⌽ 1‿2‿3", [3.0, 2.0, 1.0]),
            ("1 ⌽ 1‿2‿3", [2.0, 3.0, 1.0]),
            ("2 ⊑ 10‿20‿30", 30.0),
            ("1‿3 ⊏ 10‿20‿30‿40", [20.0, 40.0]),
            ("⥊ (2‿3 ⥊ ↕6)", [0, 1, 2, 3, 4, 5]),
            ("2‿3 ⥊ ↕6", [[0, 1, 2], [3, 4, 5]]),
            ("⊔ 1‿0‿1‿2", [[1], [0, 2], [3]]),
            ("1‿0‿2 / 10‿20‿30", [10.0, 30.0, 30.0]),
            ("≍ 10‿20", [[10.0, 20.0]]),
            ("1‿2 ≍ 3‿4", [[1.0, 2.0], [3.0, 4.0]]),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)

    def test_bqncrate_search_and_ordering_primitives(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("∧ 3‿1‿2", [1.0, 2.0, 3.0]),
            ("∨ 3‿1‿2", [3.0, 2.0, 1.0]),
            ("⍋ 3‿1‿2", [1, 2, 0]),
            ("⍒ 3‿1‿2", [0, 2, 1]),
            ("2‿4‿6 ⍋ 1‿2‿3‿7", [0, 1, 1, 3]),
            ("6‿4‿2 ⍒ 7‿6‿5‿1", [0, 1, 1, 3]),
            ("∊ 3‿1‿3‿2‿1", [1, 1, 0, 1, 0]),
            ("1‿5‿4 ∊ 3‿1‿4", [1, 0, 1]),
            ("⊐ 3‿1‿3‿2‿1", [0, 1, 0, 2, 1]),
            ("3‿1‿4 ⊐ 1‿5‿4", [1, 3, 2]),
            ("⊒ 3‿1‿3‿2‿1‿3", [0, 0, 1, 0, 1, 2]),
            ("3‿1‿3 ⊒ 3‿3‿3", [0, 2, 3]),
            ("⍷ 3‿1‿3‿2‿1", [3.0, 1.0, 2.0]),
            ("1‿2 ⍷ 0‿1‿2‿1‿2‿3", [0, 1, 0, 1, 0]),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)

    def test_bqncrate_modifiers_and_combinators(self) -> None:
        from bqn_jax import evaluate

        G = lambda x: 2 * x
        setattr(G, "_bqn_inverse", lambda y: y / 2)

        cases = [
            ("2 F˜ 5", 52.0, {"F": lambda w, x: 10 * w + x}),
            ("F¨ 1‿2‿3", [2.0, 3.0, 4.0], {"F": lambda x: x + 1}),
            ("1‿2 Add⌜ 10‿20", [[11.0, 21.0], [12.0, 22.0]], {"Add": lambda w, x: w + x}),
            ("+´ 1‿2‿3‿4", 10.0, {}),
            ("10 +´ 1‿2‿3", 16.0, {}),
            ("2 (+⁼) 5", 3.0, {}),
            ("F⌾G 3", 3.5, {"F": lambda x: x + 1, "G": G}),
            ("F∘G 3", 40.0, {"F": lambda x: 10 * x, "G": lambda x: x + 1}),
            ("2 F○G 3", -10.0, {"F": lambda w, x: w - x, "G": lambda x: 10 * x}),
            ("2 F⊸G 3", 303.0, {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}),
            ("2 F⟜G 3", 204.0, {"F": lambda w, x: 100 * w + x, "G": lambda x: x + 1}),
            ("2 F⊘G 3", 203.0, {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}),
            (
                "Sel◶A‿B 3",
                103.0,
                {"Sel": lambda x: x % 2, "A": lambda x: x + 10, "B": lambda x: x + 100},
            ),
            ("F⚇1 ⟨1⋄2‿3⟩", [2.0, [3.0, 4.0]], {"F": lambda x: x + 1}),
        ]

        for expr, expected, env in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr, env=env), expected)

    def test_bqncrate_rank_complex_and_system_entries(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("+´⎉1 (2‿3 ⥊ ↕6)", [3, 12]),
            ("1 + 2i", complex(1.0, 2.0)),
            ("i × i", complex(-1.0, 0.0)),
            ("•pi", 3.1415926535),
            ("•.e", 2.7182818284),
        ]

        for expr, expected in cases:
            with self.subTest(expr=expr):
                self._assert_close(evaluate(expr), expected)


if __name__ == "__main__":
    unittest.main()
