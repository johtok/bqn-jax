from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for runtime value-model tests")
class RuntimeValueModelTests(unittest.TestCase):
    def test_char_is_first_class_value(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.values import BQNChar

        out = evaluate("'a'")
        self.assertIsInstance(out, BQNChar)
        self.assertEqual(out.value, "a")

        plus_one = evaluate("'a' + 1")
        self.assertEqual(int(plus_one), ord("a") + 1)

    def test_boxed_arrays_use_explicit_container(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.values import BoxedArray

        boxed = evaluate("⟨1⋄2‿3⟩")
        self.assertIsInstance(boxed, BoxedArray)
        self.assertIsInstance(boxed, list)

    def test_value_info_atom_array_boxed_and_char(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.values import ValueKind, value_info

        atom = value_info(evaluate("1"))
        arr = value_info(evaluate("↕ 4"))
        boxed = value_info(evaluate("⟨1⋄2‿3⟩"))
        ch = value_info(evaluate("'z'"))

        self.assertEqual(atom.kind, ValueKind.ATOM)
        self.assertEqual(atom.rank, 0)

        self.assertEqual(arr.kind, ValueKind.ARRAY)
        self.assertEqual(arr.rank, 1)
        self.assertEqual(arr.shape, (4,))

        self.assertEqual(boxed.kind, ValueKind.BOXED_ARRAY)
        self.assertEqual(boxed.rank, 1)
        self.assertGreaterEqual(boxed.depth, 1)

        self.assertEqual(ch.kind, ValueKind.CHARACTER)
        self.assertEqual(ch.rank, 0)

    def test_operation_values_have_consistent_metadata(self) -> None:
        from bqn_jax.evaluator import DerivedMod1, DerivedMod2, PrimitiveFunction
        from bqn_jax.values import OperationInfo, operation_info

        primitive = PrimitiveFunction(op="+")
        fold = DerivedMod1(op="´", operand=primitive)
        compose = DerivedMod2(op="∘", left=primitive, right=primitive)
        py_callable = operation_info(lambda x: x)

        self.assertIsInstance(primitive.info, OperationInfo)
        self.assertEqual(primitive.info.kind, "primitive")
        self.assertEqual(primitive.info.symbol, "+")

        self.assertIsInstance(fold.info, OperationInfo)
        self.assertEqual(fold.info.kind, "modifier1")
        self.assertEqual(fold.info.symbol, "´")

        self.assertIsInstance(compose.info, OperationInfo)
        self.assertEqual(compose.info.kind, "modifier2")
        self.assertEqual(compose.info.symbol, "∘")

        assert py_callable is not None
        self.assertEqual(py_callable.kind, "callable")

    def test_validator_rejects_unsupported_runtime_value(self) -> None:
        from bqn_jax.values import validate_value

        class Unsupported:
            pass

        with self.assertRaises(TypeError):
            validate_value(Unsupported(), where="unsupported")


if __name__ == "__main__":
    unittest.main()
