from __future__ import annotations

import math
import unittest

from bqn_jax.ast import Assign, Block, Call, Char, Export, Infix, Member, Mod1, Mod2, Name, Nothing, Null, Number, Prefix, String, Train, Vector
from bqn_jax.parser import ParseError, parse, parse_program


def _escape_string(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
    return "".join(out)


def _escape_char(ch: str) -> str:
    if ch == "\\":
        return "\\\\"
    if ch == "'":
        return "\\'"
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return ch


def _render_number(value) -> str:
    if isinstance(value, complex):
        if value.real != 0:
            raise AssertionError("Round-trip renderer only supports pure imaginary complex literals")
        imag = float(value.imag)
        if imag == 1:
            return "I"
        if imag == -1:
            return "¯I"
        return f"{imag}I".replace("-", "¯")

    real = float(value)
    if math.isinf(real):
        return "∞" if real > 0 else "¯∞"
    if real.is_integer():
        return str(int(real))
    return repr(real)


def _render(expr) -> str:
    if isinstance(expr, Number):
        return _render_number(expr.value)
    if isinstance(expr, Char):
        return f"'{_escape_char(expr.value)}'"
    if isinstance(expr, String):
        return f"\"{_escape_string(expr.value)}\""
    if isinstance(expr, Null):
        return "@"
    if isinstance(expr, Nothing):
        return "·"
    if isinstance(expr, Name):
        return expr.value
    if isinstance(expr, Member):
        value = _render(expr.value)
        if not isinstance(expr.value, (Name, Member)):
            value = f"({value})"
        return f"{value}.{expr.attr}"
    if isinstance(expr, Vector):
        return f"⟨{' ⋄ '.join(_render(item) for item in expr.items)}⟩"
    if isinstance(expr, Prefix):
        return f"({expr.op} {_render(expr.right)})"
    if isinstance(expr, Infix):
        return f"({_render(expr.left)} {expr.op} {_render(expr.right)})"
    if isinstance(expr, Mod1):
        return f"({_render(expr.operand)}{expr.op})"
    if isinstance(expr, Mod2):
        return f"({_render(expr.left)}{expr.op}{_render(expr.right)})"
    if isinstance(expr, Train):
        return f"({' '.join(_render(part) for part in expr.parts)})"
    if isinstance(expr, Assign):
        return f"({_render(expr.left)} {expr.op} {_render(expr.right)})"
    if isinstance(expr, Export):
        if expr.target is None:
            return "⇐"
        return f"({_render(expr.target)} ⇐)"
    if isinstance(expr, Block):
        rendered_cases: list[str] = []
        def _render_header_target(target) -> str:
            if isinstance(target, Name):
                return target.value
            if isinstance(target, Vector):
                return "‿".join(_render_header_target(item) for item in target.items)
            raise AssertionError(f"Unsupported case header target in renderer: {type(target)!r}")

        for case in expr.cases:
            head = ""
            if case.header is not None:
                if len(case.header) == 1:
                    rendered = _render_header_target(case.header[0])
                    if isinstance(case.header[0], Name):
                        head = f"F {rendered}: "
                    else:
                        head = f"{rendered}: "
                elif len(case.header) == 2:
                    left = _render_header_target(case.header[0])
                    right = _render_header_target(case.header[1])
                    head = f"{left} F {right}: "
                else:
                    raise AssertionError("Unsupported case header arity in renderer")
            pieces = [f"{_render(predicate)}?" for predicate in case.predicates]
            pieces.extend(_render(stmt) for stmt in case.body.statements)
            rendered_cases.append(head + " ⋄ ".join(pieces))
        return "{" + "; ".join(rendered_cases) + "}"
    if isinstance(expr, Call):
        if expr.left is None:
            return f"({_render(expr.func)} {_render(expr.right)})"
        return f"({_render(expr.left)} {_render(expr.func)} {_render(expr.right)})"
    raise AssertionError(f"Unsupported node in round-trip renderer: {type(expr)!r}")


class ParserGrammarConformanceTests(unittest.TestCase):
    # Source forms are derived from grammar productions in:
    # https://mlochbaum.github.io/BQN/spec/grammar.html
    def test_spec_grammar_fixture_parse_success(self) -> None:
        expression_fixtures = (
            "⇐",
            "x ⇐",
            "x ← y ← 1",
            "x ↩ 1",
            "+",
            "+´",
            "F∘+ 3",
            "F∘G H 3",
            "F G H I 3",
            "{F x: x = 0? ⋄ x + 1; w F x: w + x}",
            "{x‿y: x + y}",
            "⟨1 ⋄ 2 ⋄ 3⟩",
            "[1 ⋄ 2 ⋄ 3]",
            "ns.member.submember",
        )
        for source in expression_fixtures:
            with self.subTest(source=source):
                parse(source)

        program_fixtures = (
            "a ← 1 ⋄ b ← 2 ⋄ a + b",
            "a ← 1\nb ← 2\na + b",
            "a ← 1,b ← 2,a + b",
        )
        for source in program_fixtures:
            with self.subTest(source=source):
                parse_program(source)

    def test_spec_grammar_fixture_parse_rejections(self) -> None:
        invalid_fixtures = (
            "⟨1 2⟩",
            "F∘",
            "{x + 1 ⋄ x = 0? ⋄ x}",
            "{F x:}",
            "1 + )",
        )
        for source in invalid_fixtures:
            with self.subTest(source=source):
                with self.assertRaises((ParseError, SyntaxError)):
                    parse(source)

    def test_parser_output_is_canonical_for_equivalent_forms(self) -> None:
        expression_pairs = (
            ("1 <= 2", "1 ≤ 2"),
            ("*´ 1‿2‿3", "×´ 1‿2‿3"),
            ("(x)", "x"),
            ("F∘G H 3", "(F∘G) H 3"),
            ("F G H I 3", "F G (H I) 3"),
            ("⟨1 ⋄ 2 ⋄ 3⟩", "[1 ⋄ 2 ⋄ 3]"),
        )
        for left, right in expression_pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(parse(left), parse(right))

        program_pairs = (
            ("a ← 1 ⋄ b ← 2 ⋄ a + b", "a ← 1\nb ← 2\na + b"),
            ("a ← 1,b ← 2,a + b", "a ← 1 ⋄ b ← 2 ⋄ a + b"),
        )
        for left, right in program_pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(parse_program(left), parse_program(right))

    def test_round_trip_for_precedence_and_association_cases(self) -> None:
        cases = (
            "1 + 2 × 3",
            "F∘G H 3",
            "F G H I 3",
            "2 F○G 3",
            "+´⎉1 (2‿3 ⥊ ↕6)",
            "{F x: x = 0? ⋄ x + 1; w F x: w + x}",
            "x ← y ← 1",
            "a.b.c",
        )
        for source in cases:
            with self.subTest(source=source):
                parsed = parse(source)
                rendered = _render(parsed)
                reparsed = parse(rendered)
                self.assertEqual(reparsed, parsed)


if __name__ == "__main__":
    unittest.main()
