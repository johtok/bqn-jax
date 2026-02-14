from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for interpreter tests")
class ReadmeExamplesTests(unittest.TestCase):
    """Coverage for the README evaluate(...) example block."""

    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def _assert_close(self, got, want, *, places: int = 6) -> None:
        got_py = self._to_python(got)

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

        check(got_py, want)

    def _train_env(self):
        return {
            "F": lambda *a: a[0] + 1 if len(a) == 1 else a[0] + a[1] + 1,
            "G": lambda *a: 10 * a[0] if len(a) == 1 else 10 * a[0] + a[1],
            "H": lambda *a: 2 * a[0] if len(a) == 1 else a[0] - a[1],
            "I": lambda *a: a[0] - 3 if len(a) == 1 else a[0] * a[1],
        }

    def test_readme_examples_block(self) -> None:
        from bqn_jax import evaluate

        cases = [
            ("1 + 2 × 3", 7.0, None),
            ("3 ⥊ 1‿2‿3", [1.0, 2.0, 3.0], None),
            ("≢ (2‿3 ⥊ ↕6)", [2, 3], None),
            ("a ← 10 ⋄ b ← 3 ⋄ a | b", 3.0, None),
            ("+´ 1‿2‿3‿4", 10.0, None),
            ("F ← {𝕨 + 𝕩} ⋄ 2 F 3", 5.0, None),
            ("F ← {𝕩+1;𝕨+𝕩} ⋄ F 4", 5.0, None),
            ("F ← {F x: x+1} ⋄ F 4", 5.0, None),
            ("F ← {𝕩=0? ⋄ 42; 𝕩+1} ⋄ F 0", 42.0, None),
            ("2 ↑ 10‿20‿30‿40", [10.0, 20.0], None),
            ("5 ↑ 10‿20‿30", [10.0, 20.0, 30.0, 0.0, 0.0], None),
            ("2 ↓ 10‿20‿30‿40", [30.0, 40.0], None),
            ("1‿2 ∾ 3‿4", [1.0, 2.0, 3.0, 4.0], None),
            ("2 ⊑ 10‿20‿30", 30.0, None),
            ("1‿3 ⊏ 10‿20‿30‿40", [20.0, 40.0], None),
            ("(2‿2 ⥊ 0‿2‿1‿3) ⊏ 10‿20‿30‿40", [[10.0, 30.0], [20.0, 40.0]], None),
            ("(2‿2 ⥊ 0‿2‿1‿3) ⊏ ⟨10⋄20⋄30⋄40⋄50‿51⟩", [[10.0, 30.0], [20.0, 40.0]], None),
            ("3‿1‿4 ⊐ 1‿5‿4", [1, 3, 2], None),
            ("⊔ 1‿0‿1‿2", [[1], [0, 2], [3]], None),
            ("(2‿2 ⥊ 1‿0‿1‿2) ⊔ 10‿20‿30‿40", [[20.0], [10.0, 30.0], [40.0]], None),
            ("1‿0‿2 / 10‿20‿30", [10.0, 30.0, 30.0], None),
            ("(2‿2 ⥊ 1‿0‿2‿1) / 10‿20‿30‿40", [10.0, 30.0, 30.0, 40.0], None),
            ("1 ≍ 2‿3", [1.0, [2.0, 3.0]], None),
            ('"oops" ! 1', 1.0, None),
            ("∧ 3‿1‿2", [1.0, 2.0, 3.0], None),
            ("1‿0‿2 ∧ 0‿4‿5", [0, 0, 10], None),
            ("⍋ 3‿1‿2", [1, 2, 0], None),
            ("⍒ 3‿1‿2", [0, 2, 1], None),
            ("∧ ⟨⟨1⋄2‿3⟩⋄⟨1⋄2‿2⟩⋄0⟩", [0.0, [1.0, [2.0, 2.0]], [1.0, [2.0, 3.0]]], None),
            ("∊ 3‿1‿3‿2‿1", [1, 1, 0, 1, 0], None),
            ("⊒ 3‿1‿3‿2‿1‿3", [0, 0, 1, 0, 1, 2], None),
            ("3‿1‿4 ∊ 1‿5‿4", [0, 1, 1], None),
            ("3‿1‿3 ⊒ 3‿3‿3", [0, 2, 3], None),
            ("1‿2 ⍷ 0‿1‿2‿1‿2‿3", [0, 1, 0, 1, 0], None),
            ("2‿4‿6 ⍋ 1‿2‿3‿7", [0, 1, 1, 3], None),
            ("6‿4‿2 ⍒ 7‿6‿5‿1", [0, 1, 1, 3], None),
            ("3˙ 99", 3.0, None),
            ("2 F˜ 5", 52.0, {"F": lambda w, x: 10 * w + x}),
            ("F∘G 3", 40.0, {"F": lambda x: 10 * x, "G": lambda x: x + 1}),
            ("2 F○G 3", -10.0, {"F": lambda w, x: w - x, "G": lambda x: 10 * x}),
            ("2 F⊸G 3", 303.0, {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}),
            ("2 F⟜G 3", 204.0, {"F": lambda w, x: 100 * w + x, "G": lambda x: x + 1}),
            ("2 F⊘G 3", 203.0, {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}),
            ("Sel◶A‿B 3", 103.0, {"Sel": lambda x: x % 2, "A": lambda x: x + 10, "B": lambda x: x + 100}),
            ("2 (+⁼) 5", 3.0, None),
            ("1 + 2i", complex(1.0, 2.0), None),
            ("•pi", 3.1415926535, None),
            ("F G 3", 31.0, self._train_env()),
            ("F G H 3", 46.0, self._train_env()),
            ("F G H I 4", 52.0, self._train_env()),
            ("F (G H) 4", 81.0, self._train_env()),
        ]

        for expr, expected, env in cases:
            with self.subTest(expr=expr):
                got = evaluate(expr, env=env) if env is not None else evaluate(expr)
                self._assert_close(got, expected)

        policy = self._to_python(evaluate("•policy"))
        self.assertIsInstance(policy, list)
        self.assertGreater(len(policy), 0)
        self.assertTrue(all(isinstance(c, int) for c in policy))
        policy_text = "".join(chr(c) for c in policy)
        self.assertIn("read-only", policy_text)


if __name__ == "__main__":
    unittest.main()
