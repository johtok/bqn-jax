"""Conformance and dashboard helpers for suite-level pass-rate reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
import io
import json
import unittest


@dataclass(frozen=True)
class SuiteStats:
    name: str
    tests_run: int
    passed: int
    failed: int
    errors: int
    skipped: int
    executable: int
    pass_rate: float | None
    status: str

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "skipped"}


@dataclass(frozen=True)
class SpecSection:
    page: str
    title: str
    patterns: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://mlochbaum.github.io/BQN/spec/{self.page}"


_SPEC_SECTIONS: Final[tuple[SpecSection, ...]] = (
    SpecSection(page="types.html", title="Types", patterns=("test_runtime_value_model.py",)),
    SpecSection(page="token.html", title="Token", patterns=("test_lexer_and_literal_coverage.py",)),
    SpecSection(page="literal.html", title="Literal", patterns=("test_parser_language_syntax.py",)),
    SpecSection(
        page="grammar.html",
        title="Grammar",
        patterns=("test_parser_language_syntax.py", "test_parser_grammar_conformance.py"),
    ),
    SpecSection(page="scope.html", title="Scope", patterns=("test_runtime_language_semantics.py",)),
    SpecSection(
        page="evaluate.html",
        title="Evaluate",
        patterns=("test_runtime_language_semantics.py", "test_tutorial_and_cbqn_examples.py"),
    ),
    SpecSection(
        page="primitive.html",
        title="Primitive",
        patterns=("test_primitive_modifier_matrix.py", "test_bqncrate_examples.py"),
    ),
    SpecSection(page="inferred.html", title="Inferred", patterns=("test_inferred_properties.py",)),
    SpecSection(page="complex.html", title="Complex", patterns=("test_runtime_language_semantics.py",)),
    SpecSection(page="system.html", title="System", patterns=("test_system_surface.py",)),
)


def default_spec_sections() -> tuple[SpecSection, ...]:
    return _SPEC_SECTIONS


def _discover_suite(patterns: tuple[str, ...], *, tests_dir: Path) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for pattern in patterns:
        discovered = loader.discover(start_dir=str(tests_dir), pattern=pattern, top_level_dir=str(tests_dir))
        suite.addTests(discovered)
    return suite


def run_patterns(name: str, patterns: tuple[str, ...], *, tests_dir: Path = Path("tests")) -> SuiteStats:
    suite = _discover_suite(patterns, tests_dir=tests_dir)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    executed = result.testsRun - skipped
    passed = result.testsRun - failures - errors - skipped

    if failures or errors:
        status = "fail"
    elif executed == 0:
        status = "skipped"
    else:
        status = "pass"

    pass_rate = None if executed == 0 else (passed / executed) * 100.0
    return SuiteStats(
        name=name,
        tests_run=result.testsRun,
        passed=passed,
        failed=failures,
        errors=errors,
        skipped=skipped,
        executable=executed,
        pass_rate=pass_rate,
        status=status,
    )


def run_spec_sections(*, tests_dir: Path = Path("tests")) -> list[tuple[SpecSection, SuiteStats]]:
    rows: list[tuple[SpecSection, SuiteStats]] = []
    for section in default_spec_sections():
        stats = run_patterns(section.page, section.patterns, tests_dir=tests_dir)
        rows.append((section, stats))
    return rows


def aggregate(name: str, stats: list[SuiteStats]) -> SuiteStats:
    tests_run = sum(s.tests_run for s in stats)
    passed = sum(s.passed for s in stats)
    failed = sum(s.failed for s in stats)
    errors = sum(s.errors for s in stats)
    skipped = sum(s.skipped for s in stats)
    executable = sum(s.executable for s in stats)
    pass_rate = None if executable == 0 else (passed / executable) * 100.0
    status = "fail" if (failed or errors) else ("skipped" if executable == 0 else "pass")
    return SuiteStats(
        name=name,
        tests_run=tests_run,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        executable=executable,
        pass_rate=pass_rate,
        status=status,
    )


def stats_to_markdown_table(rows: list[SuiteStats]) -> str:
    lines = [
        "| Suite | Run | Passed | Skipped | Failed | Errors | Pass Rate | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        rate = "n/a" if row.pass_rate is None else f"{row.pass_rate:.2f}%"
        lines.append(
            f"| `{row.name}` | {row.tests_run} | {row.passed} | {row.skipped} | {row.failed} | {row.errors} | {rate} | {row.status} |"
        )
    return "\n".join(lines)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stats_payload(rows: list[SuiteStats]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]
