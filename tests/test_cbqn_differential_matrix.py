from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import unittest

from primitive_matrix_cases import ALL_MATRIX_CASES


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


def _find_cbqn_binary() -> str | None:
    configured = os.environ.get("CBQN_BIN") or os.environ.get("BQN_BIN")
    if configured:
        return configured
    for candidate in ("bqn", "cbqn"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


CBQN_BIN = _find_cbqn_binary()


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for CBQN differential tests")
@unittest.skipUnless(CBQN_BIN is not None, "CBQN binary is required for differential tests")
class CbqnDifferentialMatrixTests(unittest.TestCase):
    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(item) for item in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def _assert_close_value(self, got, want) -> None:
        g = self._to_python(got)
        w = self._to_python(want)

        if isinstance(g, list) or isinstance(w, list):
            self.assertIsInstance(g, list)
            self.assertIsInstance(w, list)
            self.assertEqual(len(g), len(w))
            for g_item, w_item in zip(g, w, strict=True):
                self._assert_close_value(g_item, w_item)
            return

        if isinstance(g, (int, float)) and isinstance(w, (int, float)):
            if math.isinf(float(g)) or math.isinf(float(w)):
                self.assertEqual(math.isinf(float(g)), math.isinf(float(w)))
                self.assertEqual(math.copysign(1.0, float(g)), math.copysign(1.0, float(w)))
                return
            self.assertAlmostEqual(float(g), float(w), places=6)
            return

        if isinstance(g, complex) or isinstance(w, complex):
            gc = complex(g)
            wc = complex(w)
            self.assertAlmostEqual(gc.real, wc.real, places=6)
            self.assertAlmostEqual(gc.imag, wc.imag, places=6)
            return

        self.assertEqual(g, w)

    def _decode_bqn_string_literal(self, raw: str) -> str:
        text = raw.strip()
        if len(text) < 2 or not text.startswith('"') or not text.endswith('"'):
            raise ValueError(f"Expected BQN string literal, got: {text!r}")
        return text[1:-1].replace('""', '"')

    def _cbqn_repr(self, expr: str) -> str:
        proc = subprocess.run(
            [CBQN_BIN, "-p", f"•Repr ({expr})"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(stderr or f"CBQN failed for expression: {expr}")
        return self._decode_bqn_string_literal(proc.stdout)

    def test_matrix_against_cbqn_repr(self) -> None:
        from bqn_jax import evaluate

        for case in ALL_MATRIX_CASES:
            with self.subTest(symbol=case.symbol, valence=case.valence, expr=case.expr):
                ours = evaluate(case.expr)
                cbqn_repr_expr = self._cbqn_repr(case.expr)
                cbqn_as_ours = evaluate(cbqn_repr_expr)
                self._assert_close_value(ours, cbqn_as_ours)


if __name__ == "__main__":
    unittest.main()
