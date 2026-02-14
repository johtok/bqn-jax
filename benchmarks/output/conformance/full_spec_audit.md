# Full Spec Audit

- Spec index URL: https://mlochbaum.github.io/BQN/spec/index.html
- Spec pages in index: 10
- Spec pages in conformance mapping: 10

## Coverage Cross-Check

- Missing in conformance mapping: `none`
- Extra in conformance mapping: `none`
- Compliance reporting is derived from executable conformance and strict probes.

## Conformance Summary

- `spec`: run=470, pass_rate=100.00% status=pass
- `runtime`: run=126, pass_rate=100.00% status=pass
- `jax-ir`: run=21, pass_rate=100.00% status=pass
- `diff-cbqn`: run=1, pass_rate=100.00% status=pass

## Strict Compatibility Probes

- Total probes: 14
- Failures: 0

| Probe | Page | Expect | Observed | Status |
|---|---|---|---|---|
| `token_word_requires_letter` | `token.html` | `parse_error` | `parse_error` | pass |
| `literal_char_no_backslash_escapes` | `token.html` | `parse_error` | `parse_error` | pass |
| `grammar_header_allows_lhscomp_list` | `grammar.html` | `parse_ok` | `parse_ok` | pass |
| `grammar_header_allows_lhscomp_strand` | `grammar.html` | `parse_ok` | `parse_ok` | pass |
| `grammar_general_case_order` | `grammar.html` | `parse_error` | `parse_error` | pass |
| `grammar_general_case_count_imm` | `grammar.html` | `parse_error` | `parse_error` | pass |
| `token_system_literal_must_be_defined` | `token.html` | `parse_error` | `parse_error` | pass |
| `token_identifier_role_function_uppercase` | `token.html` | `parse_error` | `parse_error` | pass |
| `token_identifier_role_function_uppercase_allowed` | `token.html` | `parse_ok` | `parse_ok` | pass |
| `grammar_monadic_header_omit_function_requires_non_name_argument` | `grammar.html` | `parse_error` | `parse_error` | pass |
| `grammar_header_function_role_required` | `grammar.html` | `parse_error` | `parse_error` | pass |
| `grammar_header_function_role_uppercase_allowed` | `grammar.html` | `parse_ok` | `parse_ok` | pass |
| `grammar_inference_header_undo_form` | `grammar.html` | `parse_ok` | `parse_ok` | pass |
| `grammar_inference_header_under_undo_form` | `grammar.html` | `parse_ok` | `parse_ok` | pass |
