from __future__ import annotations

import importlib.util
import math
import time
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for system-surface tests")
class SystemSurfaceTests(unittest.TestCase):
    SYSTEM_KEYS = {
        "pi",
        "e",
        "inf",
        "nan",
        "i",
        "true",
        "false",
        "cwd",
        "time",
        "BQN",
        "version",
        "policy",
        "Type",
        "Repr",
        "Fmt",
        "ParseFloat",
        "UnixTime",
        "MonoTime",
        "Delay",
        "type",
        "repr",
        "fmt",
        "parse_float",
        "unix_time",
        "mono_time",
        "delay",
    }

    def _decode_text(self, value) -> str:
        arr = value
        if hasattr(arr, "tolist"):
            arr = arr.tolist()
        return "".join(chr(int(c)) for c in arr)

    def test_system_namespace_surfaces_all_documented_entries(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.evaluator import Namespace

        ns = evaluate("•")
        self.assertIsInstance(ns, Namespace)
        self.assertEqual(set(ns.values.keys()), self.SYSTEM_KEYS)

        for key in self.SYSTEM_KEYS:
            with self.subTest(key=key):
                evaluate(f"•{key}")
                evaluate(f"•.{key}")

    def test_system_constants_and_metadata(self) -> None:
        from bqn_jax import evaluate

        self.assertAlmostEqual(float(evaluate("•pi")), math.pi, places=6)
        self.assertAlmostEqual(float(evaluate("•e")), math.e, places=6)
        self.assertTrue(math.isinf(float(evaluate("•inf"))))
        self.assertTrue(math.isnan(float(evaluate("•nan"))))
        self.assertEqual(int(evaluate("•true")), 1)
        self.assertEqual(int(evaluate("•false")), 0)
        self.assertEqual(complex(evaluate("•i").item()), 1j)

        cwd = self._decode_text(evaluate("•cwd"))
        version = self._decode_text(evaluate("•version"))
        policy = self._decode_text(evaluate("•policy"))
        bqn_name = self._decode_text(evaluate("•BQN"))
        self.assertTrue(cwd)
        self.assertIn("bgn-jax", version)
        self.assertIn("bgn-jax", bqn_name)
        self.assertIn("read-only", policy)
        self.assertIn("no file/network/process mutation", policy)

    def test_system_functions(self) -> None:
        from bqn_jax import evaluate

        self.assertEqual(int(evaluate("•Type 1")), 1)
        self.assertEqual(int(evaluate("•Type 1‿2")), 0)
        self.assertEqual(int(evaluate("•Type (0 ⊑ ⟨•⟩)")), 4)
        self.assertEqual(int(evaluate("•Type (0 ⊑ ⟨+⟩)")), 3)
        self.assertTrue(callable(evaluate("•type")))

        repr_text = self._decode_text(evaluate("•Repr 1‿2"))
        fmt_text = self._decode_text(evaluate("•Fmt 1‿2"))
        self.assertIn("1", repr_text)
        self.assertIn("‿", repr_text)
        self.assertIn("⟨", fmt_text)
        self.assertIn("⟩", fmt_text)
        self.assertTrue(callable(evaluate("•repr")))
        self.assertTrue(callable(evaluate("•fmt")))

        self.assertAlmostEqual(float(evaluate('•ParseFloat "1.25"')), 1.25, places=6)
        self.assertAlmostEqual(float(evaluate('•ParseFloat "¯2e¯1"')), -0.2, places=6)
        self.assertTrue(callable(evaluate("•parse_float")))

        unix_t = float(evaluate("•UnixTime 0"))
        mono_t = float(evaluate("•MonoTime 0"))
        self.assertGreater(unix_t, 1_000_000_000.0)
        self.assertGreater(mono_t, 0.0)
        self.assertTrue(callable(evaluate("•unix_time")))
        self.assertTrue(callable(evaluate("•mono_time")))

        requested = 0.005
        start = time.perf_counter()
        delayed = float(evaluate("•Delay 0.005"))
        wall_elapsed = time.perf_counter() - start
        self.assertGreaterEqual(delayed, 0.0)
        self.assertGreaterEqual(wall_elapsed, requested * 0.6)
        self.assertTrue(callable(evaluate("•delay")))

    def test_system_function_errors_and_read_only_policy(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate('•ParseFloat "not-a-number"')
        with self.assertRaises(ValueError):
            evaluate("•Delay ¯0.01")
        with self.assertRaises(ValueError):
            evaluate("•pi ← 3")
        with self.assertRaises(ValueError):
            evaluate("•Type ← +")


if __name__ == "__main__":
    unittest.main()
