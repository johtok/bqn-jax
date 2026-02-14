"""End-to-end spec audit for conformance mapping and strict parser probes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bqn_jax.conformance import aggregate, default_spec_sections, run_patterns, run_spec_sections
from bqn_jax.parser import ParseError, parse


SPEC_INDEX_URL = "https://mlochbaum.github.io/BQN/spec/index.html"


@dataclass(frozen=True)
class StrictProbe:
    id: str
    page: str
    description: str
    source: str
    expectation: str  # parse_ok | parse_error


STRICT_PROBES: tuple[StrictProbe, ...] = (
    StrictProbe(
        id="token_word_requires_letter",
        page="token.html",
        description="Word names must include alphabetic content; `_99` is not a valid identifier.",
        source="_99",
        expectation="parse_error",
    ),
    StrictProbe(
        id="literal_char_no_backslash_escapes",
        page="token.html",
        description="Character literals require no escaping; backslash escapes should not be accepted as a char literal.",
        source="'\\n'",
        expectation="parse_error",
    ),
    StrictProbe(
        id="grammar_header_allows_lhscomp_list",
        page="grammar.html",
        description="Case headers accept `LHS_COMP`, including list destructuring forms.",
        source="{⟨x⋄y⟩: x+y}",
        expectation="parse_ok",
    ),
    StrictProbe(
        id="grammar_header_allows_lhscomp_strand",
        page="grammar.html",
        description="Case headers accept strand destructuring forms.",
        source="{x‿y: x+y}",
        expectation="parse_ok",
    ),
    StrictProbe(
        id="grammar_general_case_order",
        page="grammar.html",
        description="A general body cannot appear before a non-general body in a block.",
        source="{1; F x: x}",
        expectation="parse_error",
    ),
    StrictProbe(
        id="grammar_general_case_count_imm",
        page="grammar.html",
        description="Immediate blocks cannot contain multiple general bodies.",
        source="{1;2}",
        expectation="parse_error",
    ),
    StrictProbe(
        id="token_system_literal_must_be_defined",
        page="token.html",
        description="A system literal token is valid only if it matches a defined system value.",
        source="•unknown",
        expectation="parse_error",
    ),
    StrictProbe(
        id="token_identifier_role_function_uppercase",
        page="token.html",
        description="A lowercase identifier has subject role and must not parse as direct function call.",
        source="f 1",
        expectation="parse_error",
    ),
    StrictProbe(
        id="token_identifier_role_function_uppercase_allowed",
        page="token.html",
        description="An uppercase identifier has function role and may parse as direct function call.",
        source="F 1",
        expectation="parse_ok",
    ),
    StrictProbe(
        id="grammar_monadic_header_omit_function_requires_non_name_argument",
        page="grammar.html",
        description="Monadic function header may omit function only when argument is not a plain name.",
        source="{x: x+1}",
        expectation="parse_error",
    ),
    StrictProbe(
        id="grammar_header_function_role_required",
        page="grammar.html",
        description="Function-header function token must use function role (uppercase identifier).",
        source="{f x: x}",
        expectation="parse_error",
    ),
    StrictProbe(
        id="grammar_header_function_role_uppercase_allowed",
        page="grammar.html",
        description="Uppercase function-role identifier should be allowed in function header.",
        source="{F x: x}",
        expectation="parse_ok",
    ),
    StrictProbe(
        id="grammar_inference_header_undo_form",
        page="grammar.html",
        description="Inference header using ⁼ should parse for function blocks.",
        source="{F⁼x: x}",
        expectation="parse_ok",
    ),
    StrictProbe(
        id="grammar_inference_header_under_undo_form",
        page="grammar.html",
        description="Inference header using ˜⁼ should parse when left argument header is present.",
        source="{w F˜⁼x: x}",
        expectation="parse_ok",
    ),
)


def _fetch_spec_index_pages() -> list[str]:
    html: str
    try:
        with urllib.request.urlopen(SPEC_INDEX_URL, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        # Fallback to curl in environments where Python's CA bundle is unavailable.
        proc = subprocess.run(["curl", "-fsSL", SPEC_INDEX_URL], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"Failed to fetch {SPEC_INDEX_URL}")
        html = proc.stdout
    pages = sorted(set(re.findall(r"\b([a-z]+\.html)\b", html)))
    return [page for page in pages if page != "index.html"]


def _suite_definitions() -> dict[str, tuple[str, ...]]:
    return {
        "runtime": (
            "test_runtime_language_semantics.py",
            "test_runtime_value_model.py",
            "test_system_surface.py",
            "test_inferred_properties.py",
            "test_primitive_modifier_matrix.py",
        ),
        "jax-ir": ("test_jax_ir_pipeline.py", "test_jax_ir_equivalence.py"),
        "diff-cbqn": ("test_cbqn_differential_matrix.py",),
    }


def _run_strict_probes() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for probe in STRICT_PROBES:
        parsed_ok = False
        error: str | None = None
        try:
            parse(probe.source)
            parsed_ok = True
        except (ParseError, SyntaxError, ValueError) as exc:
            error = str(exc)

        if probe.expectation == "parse_ok":
            passed = parsed_ok
            observed = "parse_ok" if parsed_ok else "parse_error"
        else:
            passed = not parsed_ok
            observed = "parse_ok" if parsed_ok else "parse_error"

        rows.append(
            {
                "id": probe.id,
                "page": probe.page,
                "description": probe.description,
                "source": probe.source,
                "expectation": probe.expectation,
                "observed": observed,
                "passed": passed,
                "error": error,
            }
        )
    return rows


def _print_markdown(
    *,
    index_pages: list[str],
    section_pages: list[str],
    spec_summary,
    group_stats: list,
    strict_rows: list[dict[str, object]],
) -> str:
    index_set = set(index_pages)
    section_set = set(section_pages)

    missing_in_sections = sorted(index_set - section_set)
    extra_in_sections = sorted(section_set - index_set)

    lines = [
        "# Full Spec Audit",
        "",
        f"- Spec index URL: {SPEC_INDEX_URL}",
        f"- Spec pages in index: {len(index_pages)}",
        f"- Spec pages in conformance mapping: {len(section_pages)}",
        "",
        "## Coverage Cross-Check",
        "",
        f"- Missing in conformance mapping: `{missing_in_sections if missing_in_sections else 'none'}`",
        f"- Extra in conformance mapping: `{extra_in_sections if extra_in_sections else 'none'}`",
        "- Compliance reporting is derived from executable conformance and strict probes.",
        "",
        "## Conformance Summary",
        "",
        f"- `spec`: run={spec_summary.tests_run}, pass_rate={'n/a' if spec_summary.pass_rate is None else f'{spec_summary.pass_rate:.2f}%'} status={spec_summary.status}",
    ]
    for row in group_stats:
        lines.append(
            f"- `{row.name}`: run={row.tests_run}, pass_rate={'n/a' if row.pass_rate is None else f'{row.pass_rate:.2f}%'} status={row.status}"
        )

    lines.extend(["", "## Strict Compatibility Probes", ""])
    strict_failures = [row for row in strict_rows if not row["passed"]]
    lines.append(f"- Total probes: {len(strict_rows)}")
    lines.append(f"- Failures: {len(strict_failures)}")
    lines.append("")
    lines.append("| Probe | Page | Expect | Observed | Status |")
    lines.append("|---|---|---|---|---|")
    for row in strict_rows:
        lines.append(
            f"| `{row['id']}` | `{row['page']}` | `{row['expectation']}` | `{row['observed']}` | {'pass' if row['passed'] else 'fail'} |"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="tests")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail the audit if strict compatibility probes detect spec mismatches",
    )
    parser.add_argument("--json-out", default="benchmarks/output/conformance/full_spec_audit.json")
    parser.add_argument("--markdown-out", default="benchmarks/output/conformance/full_spec_audit.md")
    args = parser.parse_args()

    index_pages = _fetch_spec_index_pages()
    section_pages = [section.page for section in default_spec_sections()]

    section_rows = run_spec_sections(tests_dir=Path(args.tests_dir))
    spec_summary = aggregate("spec", [stats for _, stats in section_rows])
    group_stats = [run_patterns(name, patterns, tests_dir=Path(args.tests_dir)) for name, patterns in _suite_definitions().items()]
    strict_rows = _run_strict_probes()

    report = _print_markdown(
        index_pages=index_pages,
        section_pages=section_pages,
        spec_summary=spec_summary,
        group_stats=group_stats,
        strict_rows=strict_rows,
    )
    print(report)

    payload = {
        "spec_index_url": SPEC_INDEX_URL,
        "spec_index_pages": index_pages,
        "conformance_pages": section_pages,
        "summary": {
            "spec": {
                "tests_run": spec_summary.tests_run,
                "pass_rate": spec_summary.pass_rate,
                "status": spec_summary.status,
            },
            "groups": [
                {"name": row.name, "tests_run": row.tests_run, "pass_rate": row.pass_rate, "status": row.status}
                for row in group_stats
            ],
            "strict_probes": strict_rows,
        },
    }
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md = Path(args.markdown_out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report + "\n", encoding="utf-8")

    index_set = set(index_pages)
    section_set = set(section_pages)
    missing_in_sections = bool(index_set - section_set)
    conformance_fail = spec_summary.status == "fail" or any(row.status == "fail" for row in group_stats)
    strict_fail = any(not row["passed"] for row in strict_rows)

    fail = missing_in_sections or conformance_fail
    if args.strict:
        fail = fail or strict_fail
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
