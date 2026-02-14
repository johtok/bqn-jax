"""Catalog of imported CBQN test cases and applicability categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


Applicability = Literal["shared_supported", "interpreter_only", "unsupported"]
ExpectedOutcome = Literal["matches_expected", "runtime_only_backend", "interpreter_error"]


@dataclass(frozen=True)
class CbqnCatalogCase:
    id: str
    source_file: str
    expr: str
    expected_expr: str | None
    applicability: Applicability
    expected_outcome: ExpectedOutcome
    note: str


CATALOG: Final[tuple[CbqnCatalogCase, ...]] = (
    CbqnCatalogCase(
        id="prims_box_add_left",
        source_file="test/cases/prims.bqn",
        expr="(<1)+ 1",
        expected_expr="<2",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="boxed scalar dyadic addition",
    ),
    CbqnCatalogCase(
        id="prims_box_add_right",
        source_file="test/cases/prims.bqn",
        expr="1 +<1",
        expected_expr="<2",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="boxed scalar dyadic addition, right boxed",
    ),
    CbqnCatalogCase(
        id="prims_null_subtract",
        source_file="test/cases/prims.bqn",
        expr="@-@",
        expected_expr="0",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="null arithmetic compatibility",
    ),
    CbqnCatalogCase(
        id="prims_replicate_counts",
        source_file="test/cases/prims.bqn",
        expr="2‿3‿0‿1/↕4",
        expected_expr="6⥊0‿0‿1‿1‿1‿3",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="replicate behavior with vector counts",
    ),
    CbqnCatalogCase(
        id="prims_depth_vector",
        source_file="test/cases/prims.bqn",
        expr="≡↕10",
        expected_expr="1",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="depth of rank-1 value",
    ),
    CbqnCatalogCase(
        id="prims_depth_atom",
        source_file="test/cases/prims.bqn",
        expr="≡0",
        expected_expr="0",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="depth of atom value",
    ),
    CbqnCatalogCase(
        id="prims_find_empty",
        source_file="test/cases/prims.bqn",
        expr="⟨⟩⍷1‿0⥊\"\"",
        expected_expr="1‿1⥊1",
        applicability="shared_supported",
        expected_outcome="matches_expected",
        note="find over empty pattern",
    ),
    CbqnCatalogCase(
        id="runtime_block_call",
        source_file="test/cases/blocks.bqn",
        expr="{𝕩+1} 3",
        expected_expr="4",
        applicability="interpreter_only",
        expected_outcome="runtime_only_backend",
        note="block functions are interpreter-only (not lowered to JAX IR)",
    ),
    CbqnCatalogCase(
        id="runtime_assignment",
        source_file="test/cases/scope.bqn",
        expr="a ← 1",
        expected_expr="1",
        applicability="interpreter_only",
        expected_outcome="runtime_only_backend",
        note="assignment is interpreter-only in current JAX IR path",
    ),
    CbqnCatalogCase(
        id="runtime_namespace_member",
        source_file="test/cases/scope.bqn",
        expr="({a ⇐ 1} 0).a",
        expected_expr="1",
        applicability="interpreter_only",
        expected_outcome="runtime_only_backend",
        note="namespace access is interpreter-only in current JAX IR path",
    ),
    CbqnCatalogCase(
        id="runtime_case_dispatch",
        source_file="test/cases/blocks.bqn",
        expr="2 ({𝕩+1; 𝕨+𝕩}) 3",
        expected_expr="5",
        applicability="interpreter_only",
        expected_outcome="runtime_only_backend",
        note="multi-case block dispatch is interpreter-only in current JAX IR path",
    ),
    CbqnCatalogCase(
        id="unsupported_system_import",
        source_file="test/cases/system.bqn",
        expr='•Import "mod"',
        expected_expr=None,
        applicability="unsupported",
        expected_outcome="interpreter_error",
        note="module/system import surface is intentionally not exposed",
    ),
    CbqnCatalogCase(
        id="unsupported_system_exit",
        source_file="test/cases/system.bqn",
        expr="•Exit 0",
        expected_expr=None,
        applicability="unsupported",
        expected_outcome="interpreter_error",
        note="process mutation APIs are excluded by runtime policy",
    ),
)
