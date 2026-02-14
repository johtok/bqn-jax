from __future__ import annotations

import unittest

from bqn_jax.ast import Assign, Block, Call, Char, Export, Infix, Member, Mod1, Mod2, Name, Nothing, Number, Prefix, Program, String, Train, Vector
from bqn_jax.parser import ParseError, parse, parse_program


class ParserLanguageSyntaxTests(unittest.TestCase):
    def test_right_associative_parse(self) -> None:
        expr = parse("1 + 2 × 3")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "+")
        self.assertIsInstance(expr.right, Infix)
        assert isinstance(expr.right, Infix)
        self.assertEqual(expr.right.op, "×")

    def test_monadic_parse(self) -> None:
        expr = parse("≢ ↕ 5")
        self.assertIsInstance(expr, Prefix)
        assert isinstance(expr, Prefix)
        self.assertEqual(expr.op, "≢")
        self.assertIsInstance(expr.right, Prefix)
        assert isinstance(expr.right, Prefix)
        self.assertEqual(expr.right.op, "↕")

    def test_strand_parse(self) -> None:
        expr = parse("1‿2‿3")
        self.assertIsInstance(expr, Vector)
        assert isinstance(expr, Vector)
        self.assertEqual(len(expr.items), 3)

    def test_list_literal_parse(self) -> None:
        expr = parse("⟨1⋄2⋄3⟩")
        self.assertIsInstance(expr, Vector)
        assert isinstance(expr, Vector)
        self.assertEqual(len(expr.items), 3)

    def test_named_alias_parse(self) -> None:
        expr = parse("shape (2‿3)")
        self.assertIsInstance(expr, Prefix)
        assert isinstance(expr, Prefix)
        self.assertEqual(expr.op, "≢")

    def test_assignment_parse(self) -> None:
        expr = parse("x ← 1")
        self.assertIsInstance(expr, Assign)
        assert isinstance(expr, Assign)
        self.assertEqual(expr.op, "←")

    def test_program_parse(self) -> None:
        program = parse_program("a ← 1\nb ← 2\na + b")
        self.assertIsInstance(program, Program)
        self.assertEqual(len(program.statements), 3)

    def test_string_parse(self) -> None:
        expr = parse('"ab""c"')
        self.assertIsInstance(expr, String)
        assert isinstance(expr, String)
        self.assertEqual(expr.value, 'ab"c')

    def test_numeric_literal_forms(self) -> None:
        pi_expr = parse("π")
        self.assertIsInstance(pi_expr, Number)
        assert isinstance(pi_expr, Number)
        self.assertAlmostEqual(pi_expr.value, 3.1415926, places=5)

        inf_expr = parse("∞")
        self.assertIsInstance(inf_expr, Number)
        assert isinstance(inf_expr, Number)
        self.assertTrue(inf_expr.value > 1e100)

        underscore_expr = parse("1_2_3")
        self.assertIsInstance(underscore_expr, Number)
        assert isinstance(underscore_expr, Number)
        self.assertEqual(underscore_expr.value, 123.0)

    def test_complex_numeric_literal_parse(self) -> None:
        pure_imag = parse("2i")
        self.assertIsInstance(pure_imag, Number)
        assert isinstance(pure_imag, Number)
        self.assertEqual(complex(pure_imag.value), 2j)

        pair_complex = parse("1i2")
        self.assertIsInstance(pair_complex, Number)
        assert isinstance(pair_complex, Number)
        self.assertEqual(complex(pair_complex.value), complex(1, 2))

        pair_complex_upper = parse("¯1I¯2")
        self.assertIsInstance(pair_complex_upper, Number)
        assert isinstance(pair_complex_upper, Number)
        self.assertEqual(complex(pair_complex_upper.value), complex(-1, -2))

        expr = parse("1 + 2i")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "+")
        self.assertIsInstance(expr.right, Number)
        assert isinstance(expr.right, Number)
        self.assertEqual(complex(expr.right.value), 2j)

    def test_char_parse(self) -> None:
        expr = parse("'a'")
        self.assertIsInstance(expr, Char)
        assert isinstance(expr, Char)
        self.assertEqual(expr.value, "a")

    def test_nothing_parse(self) -> None:
        expr = parse("·")
        self.assertIsInstance(expr, Nothing)

    def test_member_access_parse(self) -> None:
        expr = parse("ns.a.b")
        self.assertIsInstance(expr, Member)
        assert isinstance(expr, Member)
        self.assertEqual(expr.attr, "b")
        self.assertIsInstance(expr.value, Member)
        assert isinstance(expr.value, Member)
        self.assertEqual(expr.value.attr, "a")

    def test_export_statement_parse(self) -> None:
        target = parse_program("a ⇐")
        self.assertIsInstance(target.statements[0], Export)
        assert isinstance(target.statements[0], Export)
        self.assertIsNotNone(target.statements[0].target)

        bare = parse_program("⇐")
        self.assertIsInstance(bare.statements[0], Export)
        assert isinstance(bare.statements[0], Export)
        self.assertIsNone(bare.statements[0].target)

    def test_parse_error_contains_span_and_expected(self) -> None:
        with self.assertRaises(ParseError) as cm:
            parse("1 + )")
        err = cm.exception
        self.assertGreaterEqual(err.start, 0)
        self.assertGreater(err.end, err.start)
        self.assertTrue(err.expected)

    def test_fold_modifier_parse(self) -> None:
        expr = parse("+´ 1‿2‿3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod1)
        assert isinstance(expr.func, Mod1)
        self.assertEqual(expr.func.op, "´")
        self.assertIsInstance(expr.func.operand, Name)
        assert isinstance(expr.func.operand, Name)
        self.assertEqual(expr.func.operand.value, "+")

    def test_chained_primitive_modifier_parse(self) -> None:
        expr = parse("+´⎉1 (2‿3 ⥊ ↕6)")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertEqual(expr.func.op, "⎉")
        self.assertIsInstance(expr.func.left, Mod1)
        assert isinstance(expr.func.left, Mod1)
        self.assertEqual(expr.func.left.op, "´")

    def test_structural_dyad_parse(self) -> None:
        expr = parse("1 ∾ 2")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "∾")

    def test_select_parse(self) -> None:
        expr = parse("1 ⊏ 2‿3")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "⊏")

    def test_index_of_parse(self) -> None:
        expr = parse("1‿2 ⊐ 2")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "⊐")

    def test_group_parse(self) -> None:
        mono = parse("⊔ 1‿0‿1")
        self.assertIsInstance(mono, Prefix)
        assert isinstance(mono, Prefix)
        self.assertEqual(mono.op, "⊔")

        dyad = parse("1‿0‿1 ⊔ 10‿20‿30")
        self.assertIsInstance(dyad, Infix)
        assert isinstance(dyad, Infix)
        self.assertEqual(dyad.op, "⊔")

    def test_sort_and_boolean_parse(self) -> None:
        mono = parse("∧ 3‿1‿2")
        self.assertIsInstance(mono, Prefix)
        assert isinstance(mono, Prefix)
        self.assertEqual(mono.op, "∧")

        dyad = parse("1‿0‿1 ∨ 0‿0‿1")
        self.assertIsInstance(dyad, Infix)
        assert isinstance(dyad, Infix)
        self.assertEqual(dyad.op, "∨")

        grade_up = parse("⍋ 3‿1‿2")
        self.assertIsInstance(grade_up, Prefix)
        assert isinstance(grade_up, Prefix)
        self.assertEqual(grade_up.op, "⍋")

        grade_down = parse("⍒ 3‿1‿2")
        self.assertIsInstance(grade_down, Prefix)
        assert isinstance(grade_down, Prefix)
        self.assertEqual(grade_down.op, "⍒")

        dyad_grade_up = parse("1‿3‿5 ⍋ 4")
        self.assertIsInstance(dyad_grade_up, Infix)
        assert isinstance(dyad_grade_up, Infix)
        self.assertEqual(dyad_grade_up.op, "⍋")

        dyad_grade_down = parse("5‿3‿1 ⍒ 4")
        self.assertIsInstance(dyad_grade_down, Infix)
        assert isinstance(dyad_grade_down, Infix)
        self.assertEqual(dyad_grade_down.op, "⍒")

    def test_additional_primitive_parse(self) -> None:
        less = parse("1 < 2")
        self.assertIsInstance(less, Infix)
        assert isinstance(less, Infix)
        self.assertEqual(less.op, "<")

        right = parse("1 ⊢ 2")
        self.assertIsInstance(right, Infix)
        assert isinstance(right, Infix)
        self.assertEqual(right.op, "⊢")

        rotate = parse("1 ⌽ 2‿3")
        self.assertIsInstance(rotate, Infix)
        assert isinstance(rotate, Infix)
        self.assertEqual(rotate.op, "⌽")

        nudge_after = parse("1 » 2‿3")
        self.assertIsInstance(nudge_after, Infix)
        assert isinstance(nudge_after, Infix)
        self.assertEqual(nudge_after.op, "»")

        nudge_back = parse("1 « 2‿3")
        self.assertIsInstance(nudge_back, Infix)
        assert isinstance(nudge_back, Infix)
        self.assertEqual(nudge_back.op, "«")

        pair = parse("1 ⋈ 2")
        self.assertIsInstance(pair, Infix)
        assert isinstance(pair, Infix)
        self.assertEqual(pair.op, "⋈")

    def test_search_family_parse(self) -> None:
        mono_mark = parse("∊ 1‿2‿1")
        self.assertIsInstance(mono_mark, Prefix)
        assert isinstance(mono_mark, Prefix)
        self.assertEqual(mono_mark.op, "∊")

        mono_class = parse("⊐ 1‿2‿1")
        self.assertIsInstance(mono_class, Prefix)
        assert isinstance(mono_class, Prefix)
        self.assertEqual(mono_class.op, "⊐")

        mono_occ = parse("⊒ 1‿2‿1")
        self.assertIsInstance(mono_occ, Prefix)
        assert isinstance(mono_occ, Prefix)
        self.assertEqual(mono_occ.op, "⊒")

        mono_dedup = parse("⍷ 1‿2‿1")
        self.assertIsInstance(mono_dedup, Prefix)
        assert isinstance(mono_dedup, Prefix)
        self.assertEqual(mono_dedup.op, "⍷")

        dyad_member = parse("1‿2 ∊ 2‿3")
        self.assertIsInstance(dyad_member, Infix)
        assert isinstance(dyad_member, Infix)
        self.assertEqual(dyad_member.op, "∊")

        dyad_prog = parse("1‿2‿1 ⊒ 1‿1‿2")
        self.assertIsInstance(dyad_prog, Infix)
        assert isinstance(dyad_prog, Infix)
        self.assertEqual(dyad_prog.op, "⊒")

        dyad_find = parse("1‿2 ⍷ 0‿1‿2")
        self.assertIsInstance(dyad_find, Infix)
        assert isinstance(dyad_find, Infix)
        self.assertEqual(dyad_find.op, "⍷")

    def test_couple_parse(self) -> None:
        mono = parse("≍ 3")
        self.assertIsInstance(mono, Prefix)
        assert isinstance(mono, Prefix)
        self.assertEqual(mono.op, "≍")

        dyad = parse("1 ≍ 2")
        self.assertIsInstance(dyad, Infix)
        assert isinstance(dyad, Infix)
        self.assertEqual(dyad.op, "≍")

    def test_replicate_alias_parse(self) -> None:
        expr = parse("1‿0‿2 replicate 10‿20‿30")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "/")

    def test_replicate_slash_parse(self) -> None:
        expr = parse("1‿0‿2 / 10‿20‿30")
        self.assertIsInstance(expr, Infix)
        assert isinstance(expr, Infix)
        self.assertEqual(expr.op, "/")

    def test_self_swap_modifier_parse(self) -> None:
        expr = parse("F˜ 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod1)
        assert isinstance(expr.func, Mod1)
        self.assertEqual(expr.func.op, "˜")

    def test_constant_modifier_parse(self) -> None:
        expr = parse("3˙ 9")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod1)
        assert isinstance(expr.func, Mod1)
        self.assertEqual(expr.func.op, "˙")

    def test_mod2_compose_parse(self) -> None:
        expr = parse("F∘G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertEqual(expr.func.op, "∘")

    def test_mod2_over_parse(self) -> None:
        expr = parse("2 F○G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsNotNone(expr.left)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertEqual(expr.func.op, "○")

    def test_mod2_valences_parse(self) -> None:
        expr = parse("F⊘G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertEqual(expr.func.op, "⊘")

    def test_mod2_choose_parse(self) -> None:
        expr = parse("F◶G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertEqual(expr.func.op, "◶")

    def test_additional_mod1_parse(self) -> None:
        for op in ("¨", "˘", "⌜", "˝", "⁼", "`"):
            expr = parse(f"F{op} 3")
            self.assertIsInstance(expr, Call)
            assert isinstance(expr, Call)
            self.assertIsInstance(expr.func, Mod1)
            assert isinstance(expr.func, Mod1)
            self.assertEqual(expr.func.op, op)

    def test_additional_mod2_parse(self) -> None:
        for op in ("⌾", "⎉", "⚇", "⍟", "⎊"):
            expr = parse(f"F{op}G 3")
            self.assertIsInstance(expr, Call)
            assert isinstance(expr, Call)
            self.assertIsInstance(expr.func, Mod2)
            assert isinstance(expr.func, Mod2)
            self.assertEqual(expr.func.op, op)

    def test_two_train_parse(self) -> None:
        expr = parse("F G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertEqual(len(expr.func.parts), 2)

    def test_three_train_parse(self) -> None:
        expr = parse("F G H 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertEqual(len(expr.func.parts), 3)

    def test_dyadic_train_parse(self) -> None:
        expr = parse("2 F G 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsNotNone(expr.left)
        self.assertIsInstance(expr.func, Train)

    def test_four_train_parse(self) -> None:
        expr = parse("F G H I 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertEqual(len(expr.func.parts), 3)
        self.assertIsInstance(expr.func.parts[2], Train)

    def test_parenthesized_train_term_parse(self) -> None:
        expr = parse("F (G H) 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertIsInstance(expr.func.parts[1], Train)

    def test_block_parse(self) -> None:
        expr = parse("{𝕩 + 1}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        self.assertEqual(len(expr.cases), 1)
        self.assertEqual(len(expr.cases[0].body.statements), 1)

    def test_block_case_header_monadic_parse(self) -> None:
        expr = parse("{F x: x + 1}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        case = expr.cases[0]
        self.assertIsNotNone(case.header)
        assert case.header is not None
        self.assertEqual(case.header[0], Name(value="x"))
        self.assertEqual(len(case.predicates), 0)
        self.assertEqual(len(case.body.statements), 1)

    def test_block_case_header_dyadic_parse(self) -> None:
        expr = parse("{w F x: w + x}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        case = expr.cases[0]
        self.assertIsNotNone(case.header)
        assert case.header is not None
        self.assertEqual(case.header[0], Name(value="w"))
        self.assertEqual(case.header[1], Name(value="x"))
        self.assertEqual(len(case.predicates), 0)
        self.assertEqual(len(case.body.statements), 1)

    def test_block_case_header_destructuring_parse(self) -> None:
        expr = parse("{x‿y: x + y}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        case = expr.cases[0]
        self.assertIsNotNone(case.header)
        assert case.header is not None
        self.assertEqual(case.header[0], Vector(items=(Name(value="x"), Name(value="y"))))
        self.assertEqual(len(case.predicates), 0)
        self.assertEqual(len(case.body.statements), 1)

    def test_block_case_header_list_destructuring_parse(self) -> None:
        expr = parse("{⟨x⋄y⟩: x + y}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        case = expr.cases[0]
        self.assertIsNotNone(case.header)
        assert case.header is not None
        self.assertEqual(case.header[0], Vector(items=(Name(value="x"), Name(value="y"))))
        self.assertEqual(len(case.predicates), 0)
        self.assertEqual(len(case.body.statements), 1)

    def test_block_case_predicate_parse(self) -> None:
        expr = parse("{x = 0? ⋄ x + 1}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        case = expr.cases[0]
        self.assertEqual(case.header, None)
        self.assertEqual(len(case.predicates), 1)
        self.assertEqual(len(case.body.statements), 1)

    def test_block_case_predicate_after_statement_rejected(self) -> None:
        with self.assertRaises(SyntaxError):
            parse("{x + 1 ⋄ x = 0? ⋄ x}")

    def test_block_case_requires_trailing_statement(self) -> None:
        with self.assertRaises(SyntaxError):
            parse("{x = 0?}")

    def test_block_two_cases_parse(self) -> None:
        expr = parse("{𝕩+1; 𝕨+𝕩}")
        self.assertIsInstance(expr, Block)
        assert isinstance(expr, Block)
        self.assertEqual(len(expr.cases), 2)

    def test_block_rejects_general_case_before_non_general(self) -> None:
        with self.assertRaises(ParseError):
            parse("{1; F x: x}")

    def test_block_rejects_multiple_general_cases_for_immediate_block(self) -> None:
        with self.assertRaises(ParseError):
            parse("{1;2}")

    def test_monadic_call_parse(self) -> None:
        expr = parse("F 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsNone(expr.left)

    def test_dyadic_call_parse(self) -> None:
        expr = parse("2 F 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsNotNone(expr.left)

    def test_standalone_primitive_function_expression_parse(self) -> None:
        expr = parse("+")
        self.assertIsInstance(expr, Name)
        assert isinstance(expr, Name)
        self.assertEqual(expr.value, "+")

    def test_mod2_chain_left_associative_parse(self) -> None:
        expr = parse("F∘G∘H 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Mod2)
        assert isinstance(expr.func, Mod2)
        self.assertIsInstance(expr.func.left, Mod2)
        assert isinstance(expr.func.left, Mod2)
        self.assertEqual(expr.func.left.left, Name(value="F"))
        self.assertEqual(expr.func.left.right, Name(value="G"))
        self.assertEqual(expr.func.right, Name(value="H"))

    def test_mod2_binds_tighter_than_train_parse(self) -> None:
        expr = parse("F∘G H 3")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertEqual(len(expr.func.parts), 2)
        self.assertIsInstance(expr.func.parts[0], Mod2)
        self.assertEqual(expr.func.parts[1], Name(value="H"))

    def test_train_parses_literal_derived_callable_term(self) -> None:
        expr = parse("F 3˙ 2")
        self.assertIsInstance(expr, Call)
        assert isinstance(expr, Call)
        self.assertIsInstance(expr.func, Train)
        assert isinstance(expr.func, Train)
        self.assertEqual(len(expr.func.parts), 2)
        self.assertEqual(expr.func.parts[0], Name(value="F"))
        self.assertIsInstance(expr.func.parts[1], Mod1)
        assert isinstance(expr.func.parts[1], Mod1)
        self.assertEqual(expr.func.parts[1].op, "˙")


if __name__ == "__main__":
    unittest.main()
