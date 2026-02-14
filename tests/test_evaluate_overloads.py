from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for evaluate overload tests")
class EvaluateOverloadsTests(unittest.TestCase):
    def test_evaluate_expression_still_returns_value(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1 + 2 × 3")
        self.assertEqual(float(out), 7.0)

    def test_evaluate_env_returns_stateful_evaluator(self) -> None:
        from bqn_jax import EvaluationEnvironment, StatefulEvaluate, evaluate

        env = EvaluationEnvironment({"x": 1})
        stateful = evaluate(env)
        self.assertIsInstance(stateful, StatefulEvaluate)
        self.assertIs(stateful.env, env)
        self.assertEqual(float(stateful("x + 2")), 3.0)

    def test_stateful_evaluator_persists_var_and_function_definitions(self) -> None:
        from bqn_jax import EvaluationEnvironment, evaluate

        stateful = evaluate(EvaluationEnvironment())
        self.assertEqual(float(stateful("a ← 10 ⋄ a")), 10.0)
        self.assertEqual(float(stateful("a + 5")), 15.0)
        self.assertEqual(float(stateful("F ← {𝕩 + a} ⋄ F 2")), 12.0)
        self.assertEqual(float(stateful("F 3")), 13.0)

    def test_evaluate_expression_with_stateful_env_returns_result_and_env(self) -> None:
        from bqn_jax import EvaluationEnvironment, evaluate

        env = EvaluationEnvironment({"x": 4})
        out, same_env = evaluate("x + 2", env)
        self.assertEqual(float(out), 6.0)
        self.assertIs(same_env, env)

        updated, _ = evaluate("x ↩ 9 ⋄ x", env)
        self.assertEqual(float(updated), 9.0)
        persisted, _ = evaluate("x", env)
        self.assertEqual(float(persisted), 9.0)

    def test_plain_mapping_env_keeps_legacy_one_shot_behavior(self) -> None:
        from bqn_jax import evaluate

        env = {"x": 4}
        out = evaluate("x ↩ 7 ⋄ x", env=env)
        self.assertEqual(float(out), 7.0)
        self.assertEqual(env["x"], 4)
        self.assertEqual(float(evaluate("x", env=env)), 4.0)

    def test_mapping_first_argument_builds_stateful_environment(self) -> None:
        from bqn_jax import StatefulEvaluate, evaluate

        stateful = evaluate({"x": 2})
        self.assertIsInstance(stateful, StatefulEvaluate)
        self.assertEqual(float(stateful("x + 1")), 3.0)
        stateful("x ↩ 10")
        self.assertEqual(float(stateful("x")), 10.0)

    def test_stateful_export_result_does_not_leak_to_next_call(self) -> None:
        from bqn_jax import EvaluationEnvironment, evaluate
        from bqn_jax.evaluator import Namespace

        stateful = evaluate(EvaluationEnvironment())
        ns = stateful("a ⇐ 1")
        self.assertIsInstance(ns, Namespace)
        self.assertEqual(float(ns.values["a"]), 1.0)

        plain = stateful("2 + 3")
        self.assertNotIsInstance(plain, Namespace)
        self.assertEqual(float(plain), 5.0)

