from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for CBQN catalog tests")
class CbqnCatalogTests(unittest.TestCase):
    def _to_python(self, value):
        if isinstance(value, list):
            return [self._to_python(v) for v in value]
        if hasattr(value, "tolist"):
            return self._to_python(value.tolist())
        return value

    def test_catalog_has_all_applicability_buckets(self) -> None:
        from bqn_jax.cbqn_catalog import CATALOG

        buckets = {case.applicability for case in CATALOG}
        self.assertEqual(buckets, {"shared_supported", "interpreter_only", "unsupported"})

    def test_shared_supported_cases_match_expected(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.cbqn_catalog import CATALOG

        for case in CATALOG:
            if case.applicability != "shared_supported":
                continue
            assert case.expected_expr is not None
            with self.subTest(case=case.id):
                got = evaluate(case.expr)
                want = evaluate(case.expected_expr)
                self.assertEqual(self._to_python(got), self._to_python(want))

    def test_interpreter_only_cases_are_runtime_only_for_ir(self) -> None:
        from bqn_jax import evaluate, lower_to_ir
        from bqn_jax.cbqn_catalog import CATALOG
        from bqn_jax.errors import BQNUnsupportedError

        for case in CATALOG:
            if case.applicability != "interpreter_only":
                continue
            assert case.expected_expr is not None
            with self.subTest(case=case.id):
                got = evaluate(case.expr)
                want = evaluate(case.expected_expr)
                self.assertEqual(self._to_python(got), self._to_python(want))

                with self.assertRaises(BQNUnsupportedError) as cm:
                    lower_to_ir(case.expr)
                self.assertIn("Runtime-only feature", str(cm.exception))

    def test_unsupported_cases_raise_in_interpreter(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.cbqn_catalog import CATALOG

        for case in CATALOG:
            if case.applicability != "unsupported":
                continue
            with self.subTest(case=case.id):
                with self.assertRaises(Exception):
                    evaluate(case.expr)


if __name__ == "__main__":
    unittest.main()
