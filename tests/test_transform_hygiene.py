from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = ("src/bqn_jax", "benchmarks", "scripts")
TRANSFORM_ATTRS = {"jit", "vmap", "grad", "value_and_grad", "jacfwd", "jacrev"}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for rel in TARGET_DIRS:
        root = REPO_ROOT / rel
        files.extend(sorted(root.rglob("*.py")))
    return files


def _is_jax_transform_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id == "jax" and func.attr in TRANSFORM_ATTRS
    return False


class TransformHygieneTests(unittest.TestCase):
    def test_no_nested_jax_transform_construction(self) -> None:
        violations: list[str] = []

        for path in _iter_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not _is_jax_transform_call(node):
                    continue
                call = node
                assert isinstance(call, ast.Call)
                nested_args = [arg for arg in call.args if _is_jax_transform_call(arg)]
                if nested_args:
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{call.lineno}")

        self.assertEqual(
            [],
            violations,
            msg="Nested jax transform constructors found:\n" + "\n".join(violations),
        )

    def test_no_lambda_body_builds_jax_transform(self) -> None:
        violations: list[str] = []

        for path in _iter_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Lambda):
                    continue
                if any(_is_jax_transform_call(inner) for inner in ast.walk(node.body)):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{node.lineno}")

        self.assertEqual(
            [],
            violations,
            msg="Lambdas that construct jax transforms found:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
