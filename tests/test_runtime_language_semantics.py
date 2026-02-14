from __future__ import annotations

import importlib.util
import unittest


JAX_AVAILABLE = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(JAX_AVAILABLE, "jax is required for runtime semantics tests")
class RuntimeLanguageSemanticsTests(unittest.TestCase):
    def test_right_associative_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1 + 2 × 3")
        self.assertEqual(float(out), 7.0)

    def test_monadic_shape(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("≢ (2‿3 ⥊ ↕6)")
        self.assertEqual(list(out.tolist()), [2, 3])

    def test_ascii_aliases(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1 + 2 * 3")
        self.assertEqual(float(out), 7.0)

    def test_comparison(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("3 <= 4")
        self.assertEqual(int(out), 1)

    def test_additional_comparisons(self) -> None:
        from bqn_jax import evaluate

        lt = evaluate("3 < 4")
        gt = evaluate("3 > 4")
        ge = evaluate("4 ≥ 4")
        self.assertEqual(int(lt), 1)
        self.assertEqual(int(gt), 0)
        self.assertEqual(int(ge), 1)

    def test_env_name_lookup(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("x + 2", env={"x": 4})
        self.assertEqual(float(out), 6.0)

    def test_program_assignment(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("a ← 3 ⋄ b ← 4 ⋄ a × b")
        self.assertEqual(float(out), 12.0)

    def test_destructuring_assignment(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("a‿b ← 5‿6 ⋄ a + b")
        self.assertEqual(float(out), 11.0)

    def test_modified_assignment_requires_existing(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(NameError):
            evaluate("x ↩ 1")

    def test_name_equivalence_ignores_case_and_underscores(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("A_B ← 4 ⋄ a_b + AB")
        self.assertEqual(float(out), 8.0)

    def test_duplicate_definition_same_scope_rejected(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(NameError):
            evaluate("x ← 1 ⋄ X_ ← 2")

    def test_update_name_equivalence_ignores_case_and_underscores(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("x ← 1 ⋄ X_ ↩ 5 ⋄ x")
        self.assertEqual(float(out), 5.0)

    def test_infix_evaluation_order_right_before_left(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("x ← 0 ⋄ (x ↩ 1) + (x ↩ 2) ⋄ x")
        self.assertEqual(int(out), 1)

    def test_dyadic_call_evaluation_order_right_before_left(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("x ← 0 ⋄ F ← {𝕨 + 𝕩} ⋄ (x ↩ 1) F (x ↩ 2) ⋄ x")
        self.assertEqual(int(out), 1)

    def test_assignment_to_nothing_target_is_noop(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("· ← 3 ⋄ 7")
        self.assertEqual(float(out), 7.0)

    def test_namespace_export_and_member_access(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.evaluator import Namespace

        out = evaluate("F ← {a ⇐ 1 ⋄ b ⇐ 2} ⋄ ns ← F 0 ⋄ ns.b")
        self.assertEqual(float(out), 2.0)

        ns = evaluate("a ⇐ 1 ⋄ b ← 2")
        self.assertIsInstance(ns, Namespace)
        self.assertIn("a", ns.values)
        self.assertNotIn("b", ns.values)
        self.assertEqual(float(ns.values["a"]), 1.0)

    def test_namespace_destructuring_assignment_from_namespace(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("ns ← {a ⇐ 1 ⋄ b ⇐ 2} 0 ⋄ a‿b ← ns ⋄ a + b")
        self.assertEqual(float(out), 3.0)

    def test_namespace_export_statement(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.evaluator import Namespace

        ns = evaluate("a ← 1 ⋄ b ← 2 ⋄ a ⇐")
        self.assertIsInstance(ns, Namespace)
        self.assertEqual(set(ns.values.keys()), {"a"})
        self.assertEqual(float(ns.values["a"]), 1.0)

    def test_bare_export_forces_namespace_result_without_exporting_all(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.evaluator import Namespace

        ns = evaluate("a ← 1 ⋄ ⇐")
        self.assertIsInstance(ns, Namespace)
        self.assertEqual(ns.values, {})

    def test_export_requires_local_definition(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(NameError):
            evaluate("a ← 1 ⋄ F ← {a ⇐} ⋄ F 0")

    def test_export_before_definition_rejected(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(NameError):
            evaluate("a ⇐ ⋄ a ← 1")

    def test_namespace_member_missing_raises(self) -> None:
        from bqn_jax import evaluate
        from bqn_jax.evaluator import Namespace

        with self.assertRaises(NameError):
            evaluate("ns.x", env={"ns": Namespace(values={"y": 1})})

    def test_numeric_literals_pi_and_infinity(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("π + 1")
        self.assertAlmostEqual(float(out), 4.1415926, places=5)
        inf_out = evaluate("∞")
        self.assertTrue(float(inf_out) > 1e100)

    def test_string_literal_shape(self) -> None:
        from bqn_jax import evaluate

        out = evaluate('≢ "abc"')
        self.assertEqual(list(out.tolist()), [3])

    def test_fold_modifier_monadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("+´ 1‿2‿3‿4")
        self.assertEqual(float(out), 10.0)

    def test_fold_modifier_with_initial(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("10 +´ 1‿2‿3")
        self.assertEqual(float(out), 16.0)

    def test_mixed_rank_boxed_vector_and_depth(self) -> None:
        from bqn_jax import evaluate

        depth = evaluate("≡ 1‿(2‿3)")
        self.assertEqual(int(depth), 2)
        mapped = evaluate("- 1‿(2‿3)")
        self.assertEqual(int(mapped[0]), -1)
        self.assertEqual(list(mapped[1].tolist()), [-2, -3])

    def test_block_monadic_call(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F ← {𝕩 + 1} ⋄ F 4")
        self.assertEqual(float(out), 5.0)

    def test_block_dyadic_call(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F ← {𝕨 + 𝕩} ⋄ 2 F 3")
        self.assertEqual(float(out), 5.0)

    def test_lexical_closure_reference(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("a ← 10 ⋄ F ← {a + 𝕩} ⋄ a ↩ 20 ⋄ F 1")
        self.assertEqual(float(out), 21.0)

    def test_python_callable_env(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F 7", env={"F": lambda x: x + 2})
        self.assertEqual(float(out), 9.0)

    def test_block_two_case_dispatch(self) -> None:
        from bqn_jax import evaluate

        mono = evaluate("F ← {𝕩 + 1; 𝕨 + 𝕩} ⋄ F 4")
        dyad = evaluate("F ← {𝕩 + 1; 𝕨 + 𝕩} ⋄ 2 F 3")
        self.assertEqual(float(mono), 5.0)
        self.assertEqual(float(dyad), 5.0)

    def test_block_case_header_monadic_binding(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F ← {F x: x + 1} ⋄ F 4")
        self.assertEqual(float(out), 5.0)

    def test_block_case_header_dyadic_binding(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F ← {w F x: w + x} ⋄ 2 F 3")
        self.assertEqual(float(out), 5.0)

    def test_block_case_header_destructuring_binding(self) -> None:
        from bqn_jax import evaluate

        mono = evaluate("F ← {x‿y: x + y} ⋄ F 2‿3")
        dyad = evaluate("F ← {w‿z F x: w + z + x} ⋄ 2‿3 F 4")
        self.assertEqual(float(mono), 5.0)
        self.assertEqual(float(dyad), 9.0)

    def test_block_case_header_destructuring_length_mismatch(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("F ← {x‿y: x + y} ⋄ F 2‿3‿4")

    def test_block_case_header_dyadic_requires_left_argument(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("F ← {w F x: w + x} ⋄ F 3")

    def test_block_case_header_monadic_rejects_dyadic_call(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("F ← {F x: x + 1} ⋄ 2 F 3")

    def test_block_case_header_duplicate_names_rejected(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(NameError):
            evaluate("F ← {x F x: x} ⋄ 1 F 2")

    def test_block_case_predicate_fallthrough(self) -> None:
        from bqn_jax import evaluate

        yes = evaluate("F ← {𝕩 = 0? ⋄ 42; 𝕩 + 1} ⋄ F 0")
        no = evaluate("F ← {𝕩 = 0? ⋄ 42; 𝕩 + 1} ⋄ F 4")
        self.assertEqual(float(yes), 42.0)
        self.assertEqual(float(no), 5.0)

    def test_block_general_case_constraints(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(SyntaxError):
            evaluate("F ← {1; F x: x} ⋄ F 0")
        with self.assertRaises(SyntaxError):
            evaluate("F ← {1;2} ⋄ F 0")

    def test_block_case_predicate_requires_binary_scalar(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("F ← {2? ⋄ 0; 1} ⋄ F 3")

    def test_dyadic_join(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1‿2 ∾ 3‿4")
        self.assertEqual(list(out.tolist()), [1.0, 2.0, 3.0, 4.0])

    def test_dyadic_take(self) -> None:
        from bqn_jax import evaluate

        front = evaluate("2 ↑ 10‿20‿30‿40")
        back = evaluate("¯2 ↑ 10‿20‿30‿40")
        self.assertEqual(list(front.tolist()), [10.0, 20.0])
        self.assertEqual(list(back.tolist()), [30.0, 40.0])

    def test_dyadic_take_fill_extension(self) -> None:
        from bqn_jax import evaluate

        front = evaluate("5 ↑ 10‿20‿30")
        back = evaluate("¯5 ↑ 10‿20‿30")
        matrix = evaluate("4 ↑ (2‿2 ⥊ ↕4)")
        self.assertEqual(list(front.tolist()), [10.0, 20.0, 30.0, 0.0, 0.0])
        self.assertEqual(list(back.tolist()), [0.0, 0.0, 10.0, 20.0, 30.0])
        self.assertEqual(list(matrix.tolist()), [[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]])

    def test_dyadic_drop(self) -> None:
        from bqn_jax import evaluate

        front = evaluate("2 ↓ 10‿20‿30‿40")
        back = evaluate("¯1 ↓ 10‿20‿30‿40")
        self.assertEqual(list(front.tolist()), [30.0, 40.0])
        self.assertEqual(list(back.tolist()), [10.0, 20.0, 30.0])

    def test_dyadic_pick(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("2 ⊑ 10‿20‿30")
        self.assertEqual(float(out), 30.0)

    def test_additional_structural_primitives(self) -> None:
        from bqn_jax import evaluate

        mon_reverse = evaluate("⌽ 1‿2‿3")
        dyad_rotate = evaluate("1 ⌽ 1‿2‿3")
        mon_transpose = evaluate("⍉ (2‿3 ⥊ ↕6)")
        dyad_transpose = evaluate("1‿0 ⍉ (2‿3 ⥊ ↕6)")
        assert_monad = evaluate("! 1")
        assert_dyad = evaluate('"ok" ! 1')
        left = evaluate("1 ⊣ 2")
        right = evaluate("1 ⊢ 2")
        sign = evaluate("× ¯2‿0‿3")
        logical_not = evaluate("¬ 0‿2‿0")
        self.assertEqual(list(mon_reverse.tolist()), [3.0, 2.0, 1.0])
        self.assertEqual(list(dyad_rotate.tolist()), [2.0, 3.0, 1.0])
        self.assertEqual(list(mon_transpose.tolist()), [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
        self.assertEqual(list(dyad_transpose.tolist()), [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
        self.assertEqual(float(assert_monad), 1.0)
        self.assertEqual(float(assert_dyad), 1.0)
        self.assertEqual(float(left), 1.0)
        self.assertEqual(float(right), 2.0)
        self.assertEqual(list(sign.tolist()), [-1.0, 0.0, 1.0])
        self.assertEqual(list(logical_not.tolist()), [1.0, -1.0, 1.0])

    def test_shift_primitives(self) -> None:
        from bqn_jax import evaluate

        mon_after = evaluate("» 1‿2‿3")
        mon_back = evaluate("« 1‿2‿3")
        dyad_after = evaluate("9 » 1‿2‿3")
        dyad_back = evaluate("9 « 1‿2‿3")
        matrix_after = evaluate("» (2‿2 ⥊ ↕4)")
        matrix_dyad = evaluate("9‿9 » (2‿2 ⥊ ↕4)")
        self.assertEqual(list(mon_after.tolist()), [0.0, 1.0, 2.0])
        self.assertEqual(list(mon_back.tolist()), [2.0, 3.0, 0.0])
        self.assertEqual(list(dyad_after.tolist()), [9.0, 1.0, 2.0])
        self.assertEqual(list(dyad_back.tolist()), [2.0, 3.0, 9.0])
        self.assertEqual(list(matrix_after.tolist()), [[0.0, 0.0], [0.0, 1.0]])
        self.assertEqual(list(matrix_dyad.tolist()), [[9.0, 9.0], [0.0, 1.0]])

    def test_pair_primitive(self) -> None:
        from bqn_jax import evaluate

        mon = evaluate("⋈ 5")
        dyad = evaluate("1 ⋈ 2")
        self.assertEqual(list(mon.tolist()), [5.0])
        self.assertEqual(list(dyad.tolist()), [1.0, 2.0])

    def test_modifier_self_swap_monadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F˜ 3", env={"F": lambda w, x: w - x})
        self.assertEqual(float(out), 0.0)

    def test_modifier_self_swap_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("2 F˜ 5", env={"F": lambda w, x: 10 * w + x})
        self.assertEqual(float(out), 52.0)

    def test_modifier_constant(self) -> None:
        from bqn_jax import evaluate

        mono = evaluate("3˙ 9")
        dyad = evaluate("2 3˙ 9")
        self.assertEqual(float(mono), 3.0)
        self.assertEqual(float(dyad), 3.0)

    def test_additional_modifiers(self) -> None:
        from bqn_jax import evaluate

        env = {"F": lambda x: x + 1}
        each = evaluate("F¨ 1‿2‿3", env=env)
        cells = evaluate("F˘ (2‿3 ⥊ ↕6)", env=env)
        table = evaluate("1‿2 F⌜ 10‿20", env={"F": lambda w, x: w + x})
        insert = evaluate("F˝ 1‿2‿3", env={"F": lambda w, x: w + x})
        insert_init = evaluate("10 F˝ 1‿2‿3", env={"F": lambda w, x: w + x})
        undo = evaluate("F⁼ 3", env=env)
        self.assertEqual(list(each.tolist()), [2.0, 3.0, 4.0])
        self.assertEqual(list(cells.tolist()), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(list(table.tolist()), [[11.0, 21.0], [12.0, 22.0]])
        self.assertEqual(list(insert.tolist()), [6.0])
        self.assertEqual(list(insert_init.tolist()), [16.0])
        self.assertEqual(float(undo), 4.0)

    def test_scan_modifier(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("+` 1‿2‿3")
        with_init = evaluate("10 (+`) 1‿2‿3")
        self.assertEqual(list(out.tolist()), [1.0, 3.0, 6.0])
        self.assertEqual(list(with_init.tolist()), [11.0, 13.0, 16.0])

    def test_undo_modifier_for_invertible_primitives(self) -> None:
        from bqn_jax import evaluate

        add = evaluate("2 (+⁼) 5")
        sub = evaluate("2 (-⁼) 5")
        mul = evaluate("2 (×⁼) 10")
        div = evaluate("2 (÷⁼) 0.5")
        self.assertEqual(float(add), 3.0)
        self.assertEqual(float(sub), -3.0)
        self.assertEqual(float(mul), 5.0)
        self.assertEqual(float(div), 4.0)

    def test_modifier_atop_monadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F∘G 3", env={"F": lambda x: 10 * x, "G": lambda x: x + 1})
        self.assertEqual(float(out), 40.0)

    def test_modifier_atop_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("2 F∘G 3", env={"F": lambda x: 2 * x, "G": lambda w, x: w + x})
        self.assertEqual(float(out), 10.0)

    def test_modifier_over_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("2 F○G 3", env={"F": lambda w, x: w - x, "G": lambda x: 10 * x})
        self.assertEqual(float(out), -10.0)

    def test_modifier_before(self) -> None:
        from bqn_jax import evaluate

        env = {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}
        mono = evaluate("F⊸G 2", env=env)
        dyad = evaluate("2 F⊸G 3", env=env)
        self.assertEqual(float(mono), 302.0)
        self.assertEqual(float(dyad), 303.0)

    def test_modifier_after(self) -> None:
        from bqn_jax import evaluate

        env = {"F": lambda w, x: 100 * w + x, "G": lambda x: x + 1}
        mono = evaluate("F⟜G 2", env=env)
        dyad = evaluate("2 F⟜G 3", env=env)
        self.assertEqual(float(mono), 203.0)
        self.assertEqual(float(dyad), 204.0)

    def test_modifier_valences(self) -> None:
        from bqn_jax import evaluate

        env = {"F": lambda x: x + 1, "G": lambda w, x: 100 * w + x}
        mono = evaluate("F⊘G 2", env=env)
        dyad = evaluate("2 F⊘G 3", env=env)
        self.assertEqual(float(mono), 3.0)
        self.assertEqual(float(dyad), 203.0)

    def test_modifier_choose_monadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "Sel": lambda x: x % 2,
            "A": lambda x: x + 10,
            "B": lambda x: x + 100,
        }
        out = evaluate("Sel◶A‿B 3", env=env)
        self.assertEqual(float(out), 103.0)

    def test_modifier_choose_dyadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "Sel": lambda w, x: 1 if w < x else 0,
            "M": lambda w, x: w - x,
            "P": lambda w, x: w + x,
        }
        less = evaluate("2 Sel◶M‿P 5", env=env)
        not_less = evaluate("5 Sel◶M‿P 2", env=env)
        self.assertEqual(float(less), 7.0)
        self.assertEqual(float(not_less), 3.0)

    def test_additional_combinators(self) -> None:
        from bqn_jax import evaluate

        under = evaluate("F⌾G 3", env={"F": lambda x: x + 1, "G": lambda x: 2 * x})
        under_dyad = evaluate("2 F⌾G 3", env={"F": lambda w, x: w + x, "G": lambda x: 2 * x})
        rank = evaluate("F⎉1 3", env={"F": lambda x: x + 1})
        depth = evaluate("F⚇1 ⟨1⋄2‿3⟩", env={"F": lambda x: x + 1})
        self.assertEqual(float(under), 14.0)
        self.assertEqual(float(under_dyad), 16.0)
        self.assertEqual(float(rank), 4.0)
        self.assertIsInstance(depth, list)
        self.assertEqual(float(depth[0]), 2.0)
        self.assertEqual(list(depth[1].tolist()), [3.0, 4.0])

    def test_repeat_and_catch_modifiers(self) -> None:
        from bqn_jax import evaluate

        inv = lambda x: x + 2
        setattr(inv, "_bqn_inverse", lambda y: y - 2)
        mono_repeat = evaluate("F⍟2 5", env={"F": lambda x: x + 1})
        dyad_repeat = evaluate("3 F⍟2 1", env={"F": lambda w, x: w + x})
        neg_repeat = evaluate("F⍟¯2 10", env={"F": inv})
        fallback = evaluate("F⎊G 0", env={"F": lambda x: (_ for _ in ()).throw(ValueError("boom")), "G": lambda x: 99})
        normal = evaluate("F⎊G 2", env={"F": lambda x: 1 / x, "G": lambda x: 99})
        self.assertEqual(float(mono_repeat), 7.0)
        self.assertEqual(float(dyad_repeat), 7.0)
        self.assertEqual(float(neg_repeat), 6.0)
        self.assertEqual(float(fallback), 99.0)
        self.assertAlmostEqual(float(normal), 0.5, places=6)

    def test_under_uses_undo_when_available(self) -> None:
        from bqn_jax import evaluate

        G = lambda x: 2 * x
        setattr(G, "_bqn_inverse", lambda y: y / 2)
        mono = evaluate("F⌾G 3", env={"F": lambda x: x + 1, "G": G})
        dyad = evaluate("2 F⌾G 5", env={"F": lambda w, x: w + x, "G": G})
        self.assertEqual(float(mono), 3.5)
        self.assertEqual(float(dyad), 6.0)

    def test_rank_combinator_monadic_cells(self) -> None:
        from bqn_jax import evaluate

        row_sums = evaluate("+´⎉1 (2‿3 ⥊ ↕6)")
        atoms = evaluate("F⎉0 (2‿3 ⥊ ↕6)", env={"F": lambda x: 2 * x})
        self.assertEqual(list(row_sums.tolist()), [3.0, 12.0])
        self.assertEqual(list(atoms.tolist()), [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]])

    def test_rank_combinator_dyadic_cells_and_frame(self) -> None:
        from bqn_jax import evaluate

        cellwise = evaluate("1‿2 F⎉0 3‿4", env={"F": lambda w, x: 10 * w + x})
        left_row_right_atom = evaluate("(2‿2 ⥊ ↕4) F⎉1‿0 5‿6", env={"F": lambda w, x: w + x})
        self.assertEqual(list(cellwise.tolist()), [13.0, 24.0])
        self.assertEqual(list(left_row_right_atom.tolist()), [[5.0, 6.0], [8.0, 9.0]])

    def test_rank_combinator_frame_mismatch_raises(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("(2‿2 ⥊ ↕4) F⎉1 (3‿2 ⥊ ↕6)", env={"F": lambda w, x: w + x})

    def test_two_train_monadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("F G 3", env={"F": lambda x: 10 * x, "G": lambda x: x + 1})
        self.assertEqual(float(out), 40.0)

    def test_two_train_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("2 F G 3", env={"F": lambda x: 2 * x, "G": lambda w, x: w + x})
        self.assertEqual(float(out), 10.0)

    def test_three_train_monadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "F": lambda x: x + 1,
            "G": lambda w, x: 100 * w + x,
            "H": lambda x: 2 * x,
        }
        out = evaluate("F G H 3", env=env)
        self.assertEqual(float(out), 406.0)

    def test_three_train_call_order_right_before_left(self) -> None:
        from bqn_jax import evaluate

        calls: list[str] = []

        def left(x):
            calls.append("L")
            return x + 1

        def center(w, x):
            calls.append("C")
            return 10 * w + x

        def right(x):
            calls.append("R")
            return x + 2

        out = evaluate("Left Center Right 3", env={"Left": left, "Center": center, "Right": right})
        self.assertEqual(float(out), 45.0)
        self.assertEqual(calls, ["R", "L", "C"])

    def test_three_train_dyadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "F": lambda w, x: w + x,
            "G": lambda w, x: 100 * w + x,
            "H": lambda w, x: w * x,
        }
        out = evaluate("2 F G H 3", env=env)
        self.assertEqual(float(out), 506.0)

    def test_four_train_monadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "F": lambda x: x + 1,
            "G": lambda w, x: 100 * w + x,
            "H": lambda x: 2 * x,
            "I": lambda x: x + 3,
        }
        out = evaluate("F G H I 4", env=env)
        self.assertEqual(float(out), 514.0)

    def test_four_train_dyadic(self) -> None:
        from bqn_jax import evaluate

        env = {
            "F": lambda w, x: w + x,
            "G": lambda w, x: 10 * w + x,
            "H": lambda x: 2 * x,
            "I": lambda w, x: w - x,
        }
        out = evaluate("8 F G H I 3", env=env)
        self.assertEqual(float(out), 120.0)

    def test_parenthesized_train_term(self) -> None:
        from bqn_jax import evaluate

        env = {
            "F": lambda x: x + 1,
            "G": lambda x: 2 * x,
            "H": lambda x: x + 3,
        }
        out = evaluate("F (G H) 4", env=env)
        self.assertEqual(float(out), 15.0)

    def test_dyadic_select(self) -> None:
        from bqn_jax import evaluate

        scalar = evaluate("2 ⊏ 10‿20‿30‿40")
        vector = evaluate("1‿3 ⊏ 10‿20‿30‿40")
        matrix = evaluate("(2‿2 ⥊ 0‿2‿1‿3) ⊏ 10‿20‿30‿40")
        boxed_matrix = evaluate("(2‿2 ⥊ 0‿2‿1‿3) ⊏ ⟨10⋄20⋄30⋄40⋄50‿51⟩")
        multi_axis = evaluate("⟨2‿1⋄3‿0‿0⟩ ⊏ (3‿4 ⥊ ↕12)")
        multi_axis_scalar = evaluate("⟨2‿1⋄3⟩ ⊏ (3‿4 ⥊ ↕12)")
        self.assertEqual(float(scalar), 30.0)
        self.assertEqual(list(vector.tolist()), [20.0, 40.0])
        self.assertEqual(list(matrix.tolist()), [[10.0, 30.0], [20.0, 40.0]])
        self.assertEqual(list(boxed_matrix.tolist()), [[10.0, 30.0], [20.0, 40.0]])
        self.assertEqual(list(multi_axis.tolist()), [[11.0, 8.0, 8.0], [7.0, 4.0, 4.0]])
        self.assertEqual(list(multi_axis_scalar.tolist()), [11.0, 7.0])

    def test_dyadic_select_multi_axis_rank_error(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("⟨0‿1⋄0⟩ ⊏ 10‿20‿30")

    def test_dyadic_index_of(self) -> None:
        from bqn_jax import evaluate

        scalar = evaluate("3‿1‿4 ⊐ 4")
        vector = evaluate("3‿1‿4 ⊐ 1‿5‿4")
        matrix = evaluate("3‿1‿4 ⊐ (2‿2 ⥊ 1‿5‿4‿3)")
        self.assertEqual(int(scalar), 2)
        self.assertEqual(list(vector.tolist()), [1, 3, 2])
        self.assertEqual(list(matrix.tolist()), [[1, 3], [2, 0]])

    def test_sort_up_and_down(self) -> None:
        from bqn_jax import evaluate

        up = evaluate("∧ 3‿1‿2")
        down = evaluate("∨ 3‿1‿2")
        self.assertEqual(list(up.tolist()), [1.0, 2.0, 3.0])
        self.assertEqual(list(down.tolist()), [3.0, 2.0, 1.0])

    def test_sort_nested_boxed_values(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("∧ ⟨⟨1⋄2‿3⟩⋄⟨1⋄2‿2⟩⋄0⟩")
        self.assertIsInstance(out, list)
        self.assertEqual(float(out[0]), 0.0)
        self.assertIsInstance(out[1], list)
        self.assertIsInstance(out[2], list)
        self.assertEqual(float(out[1][0]), 1.0)
        self.assertEqual(list(out[1][1].tolist()), [2.0, 2.0])
        self.assertEqual(float(out[2][0]), 1.0)
        self.assertEqual(list(out[2][1].tolist()), [2.0, 3.0])

    def test_grade_up_and_down(self) -> None:
        from bqn_jax import evaluate

        up = evaluate("⍋ 3‿1‿2")
        down = evaluate("⍒ 3‿1‿2")
        self.assertEqual(list(up.tolist()), [1, 2, 0])
        self.assertEqual(list(down.tolist()), [0, 2, 1])

    def test_grade_on_matrix_rows(self) -> None:
        from bqn_jax import evaluate

        up = evaluate("⍋ (3‿2 ⥊ 2‿9‿1‿5‿2‿3)")
        down = evaluate("⍒ (3‿2 ⥊ 2‿9‿1‿5‿2‿3)")
        self.assertEqual(list(up.tolist()), [1, 2, 0])
        self.assertEqual(list(down.tolist()), [0, 2, 1])

    def test_grade_nested_boxed_values(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("⍋ ⟨⟨1⋄2‿3⟩⋄⟨1⋄2‿2⟩⋄0⟩")
        self.assertEqual(list(out.tolist()), [2, 1, 0])

    def test_grade_bins_up_and_down(self) -> None:
        from bqn_jax import evaluate

        bins_up = evaluate("2‿4‿6 ⍋ 1‿2‿3‿7")
        bins_up_scalar = evaluate("2‿4‿6 ⍋ 5")
        bins_down = evaluate("6‿4‿2 ⍒ 7‿6‿5‿1")
        self.assertEqual(list(bins_up.tolist()), [0, 1, 1, 3])
        self.assertEqual(int(bins_up_scalar), 2)
        self.assertEqual(list(bins_down.tolist()), [0, 1, 1, 3])

    def test_boolean_and_or(self) -> None:
        from bqn_jax import evaluate

        conj = evaluate("1‿0‿2 ∧ 0‿4‿5")
        disj = evaluate("1‿0‿0 ∨ 0‿0‿2")
        self.assertEqual(list(conj.tolist()), [0, 0, 10])
        self.assertEqual(list(disj.tolist()), [1, 0, 2])

    def test_search_family_monadic(self) -> None:
        from bqn_jax import evaluate

        marks = evaluate("∊ 3‿1‿3‿2‿1")
        classify = evaluate("⊐ 3‿1‿3‿2‿1")
        occ = evaluate("⊒ 3‿1‿3‿2‿1‿3")
        dedup = evaluate("⍷ 3‿1‿3‿2‿1")
        self.assertEqual(list(marks.tolist()), [1, 1, 0, 1, 0])
        self.assertEqual(list(classify.tolist()), [0, 1, 0, 2, 1])
        self.assertEqual(list(occ.tolist()), [0, 0, 1, 0, 1, 2])
        self.assertEqual(list(dedup.tolist()), [3.0, 1.0, 2.0])

    def test_search_family_dyadic(self) -> None:
        from bqn_jax import evaluate

        member_scalar = evaluate("4 ∊ 3‿1‿4")
        member_vector = evaluate("1‿5‿4 ∊ 3‿1‿4")
        member_matrix = evaluate("(2‿2 ⥊ 1‿5‿4‿3) ∊ 3‿1‿4")
        progressive = evaluate("3‿1‿3 ⊒ 3‿3‿3")
        progressive_matrix = evaluate("3‿1‿3 ⊒ (2‿2 ⥊ 3‿3‿3‿1)")
        find = evaluate("1‿2 ⍷ 0‿1‿2‿1‿2‿3")
        find_empty = evaluate("⟨⟩ ⍷ 1‿2‿3")
        self.assertEqual(int(member_scalar), 1)
        self.assertEqual(list(member_vector.tolist()), [1, 0, 1])
        self.assertEqual(list(member_matrix.tolist()), [[1, 0], [1, 1]])
        self.assertEqual(list(progressive.tolist()), [0, 2, 3])
        self.assertEqual(list(progressive_matrix.tolist()), [[0, 2], [3, 1]])
        self.assertEqual(list(find.tolist()), [0, 1, 0, 1, 0])
        self.assertEqual(list(find_empty.tolist()), [1, 1, 1, 1])

    def test_find_dyadic_ranked_windows(self) -> None:
        from bqn_jax import evaluate

        matrix = evaluate("(2‿2 ⥊ 6‿7‿10‿11) ⍷ (3‿4 ⥊ ↕12)")
        framed = evaluate("(2‿2 ⥊ 6‿7‿10‿11) ⍷ (2‿3‿4 ⥊ ↕24)")
        atom_pair = evaluate("3 ⍷ 3")
        atom_in_vector = evaluate("3 ⍷ 1‿3‿3")
        too_wide = evaluate("1‿2‿3‿4 ⍷ 3")
        self.assertEqual(list(matrix.tolist()), [[0, 0, 0], [0, 0, 1]])
        self.assertEqual(list(framed[0].tolist()), [[0, 0, 0], [0, 0, 1]])
        self.assertEqual(list(framed[1].tolist()), [[0, 0, 0], [0, 0, 0]])
        self.assertEqual(list(atom_pair.tolist()), [1])
        self.assertEqual(list(atom_in_vector.tolist()), [0, 1, 1])
        self.assertEqual(list(too_wide.tolist()), [])

    def test_find_dyadic_rank_error(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("(2‿1 ⥊ 1‿2) ⍷ 3")

    def test_search_dyadic_rank_compatibility(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(ValueError):
            evaluate("1 ⊐ 1")
        with self.assertRaises(ValueError):
            evaluate("1 ⊒ 1")
        with self.assertRaises(ValueError):
            evaluate("1 ∊ 1")
        with self.assertRaises(ValueError):
            evaluate("(2‿2 ⥊ ↕4) ⊐ 1")
        with self.assertRaises(ValueError):
            evaluate("1 ∊ (2‿2 ⥊ ↕4)")

    def test_search_dyadic_cell_rank_behavior(self) -> None:
        from bqn_jax import evaluate

        index_rows = evaluate("(3‿2 ⥊ 1‿2‿3‿4‿1‿2) ⊐ (2‿2 ⥊ 1‿2‿5‿6)")
        member_rows = evaluate("(2‿2 ⥊ 1‿2‿5‿6) ∊ (3‿2 ⥊ 1‿2‿3‿4‿1‿2)")
        progressive_rows = evaluate("(3‿2 ⥊ 1‿2‿3‿4‿1‿2) ⊒ (3‿2 ⥊ 1‿2‿1‿2‿3‿4)")
        self.assertEqual(list(index_rows.tolist()), [0, 3])
        self.assertEqual(list(member_rows.tolist()), [1, 0])
        self.assertEqual(list(progressive_rows.tolist()), [0, 2, 1])

    def test_group_monadic_and_dyadic(self) -> None:
        from bqn_jax import evaluate

        mono = evaluate("⊔ 1‿0‿1‿2")
        dyad = evaluate("1‿0‿1‿2 ⊔ 10‿20‿30‿40")
        dyad_matrix_keys = evaluate("(2‿2 ⥊ 1‿0‿1‿2) ⊔ 10‿20‿30‿40")
        mono_omit = evaluate("⊔ ¯1‿1‿0‿¯1")
        dyad_omit = evaluate("¯1‿1‿0‿¯1 ⊔ 10‿20‿30‿40")
        self.assertIsInstance(mono, list)
        self.assertIsInstance(dyad, list)
        self.assertIsInstance(dyad_matrix_keys, list)
        self.assertIsInstance(mono_omit, list)
        self.assertIsInstance(dyad_omit, list)
        self.assertEqual([list(x.tolist()) for x in mono], [[1], [0, 2], [3]])
        self.assertEqual([list(x.tolist()) for x in dyad], [[20.0], [10.0, 30.0], [40.0]])
        self.assertEqual([list(x.tolist()) for x in dyad_matrix_keys], [[20.0], [10.0, 30.0], [40.0]])
        self.assertEqual([list(x.tolist()) for x in mono_omit], [[2], [1]])
        self.assertEqual([list(x.tolist()) for x in dyad_omit], [[30.0], [20.0]])

    def test_couple_monadic_and_dyadic(self) -> None:
        from bqn_jax import evaluate

        mono = evaluate("≍ 10‿20")
        dyad = evaluate("1‿2 ≍ 3‿4")
        mixed = evaluate("1 ≍ 2‿3")
        mixed_rank = evaluate("(2‿2 ⥊ ↕4) ≍ 5‿6")
        self.assertEqual(list(mono.tolist()), [[10.0, 20.0]])
        self.assertEqual(list(dyad.tolist()), [[1.0, 2.0], [3.0, 4.0]])
        self.assertIsInstance(mixed, list)
        self.assertEqual(float(mixed[0]), 1.0)
        self.assertEqual(list(mixed[1].tolist()), [2.0, 3.0])
        self.assertIsInstance(mixed_rank, list)
        self.assertEqual(list(mixed_rank[0].tolist()), [[0.0, 1.0], [2.0, 3.0]])
        self.assertEqual(list(mixed_rank[1].tolist()), [5.0, 6.0])

    def test_replicate_alias_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1‿0‿2 replicate 10‿20‿30")
        self.assertEqual(list(out.tolist()), [10.0, 30.0, 30.0])

    def test_replicate_slash_dyadic(self) -> None:
        from bqn_jax import evaluate

        out = evaluate("1‿0‿2 / 10‿20‿30")
        matrix_counts = evaluate("(2‿2 ⥊ 1‿0‿2‿1) / 10‿20‿30‿40")
        self.assertEqual(list(out.tolist()), [10.0, 30.0, 30.0])
        self.assertEqual(list(matrix_counts.tolist()), [10.0, 30.0, 30.0, 40.0])

    def test_complex_extension(self) -> None:
        from bqn_jax import evaluate

        literal = evaluate("1 + 2i")
        pair_literal = evaluate("1i2")
        pair_literal_upper = evaluate("¯1I¯2")
        unit = evaluate("i × i")
        sqrt_neg = evaluate("√ (¯1 + 0i)")
        conjugated = evaluate("+ 1i2")
        self.assertEqual(complex(literal.item()), complex(1, 2))
        self.assertEqual(complex(pair_literal.item()), complex(1, 2))
        self.assertEqual(complex(pair_literal_upper.item()), complex(-1, -2))
        self.assertEqual(complex(unit.item()), complex(-1, 0))
        self.assertAlmostEqual(complex(sqrt_neg.item()).imag, 1.0, places=5)
        self.assertEqual(complex(conjugated.item()), complex(1, -2))

    def test_system_values_and_policy(self) -> None:
        from bqn_jax import evaluate

        pi = evaluate("•pi")
        e = evaluate("•.e")
        imag = evaluate("•i")
        policy = evaluate("•policy")
        self.assertAlmostEqual(float(pi), 3.1415926, places=5)
        self.assertAlmostEqual(float(e), 2.7182818, places=5)
        self.assertEqual(complex(imag.item()), 1j)
        policy_text = "".join(chr(int(c)) for c in policy.tolist())
        self.assertIn("read-only", policy_text)
        self.assertIn("no file/network/process mutation", policy_text)

    def test_system_value_errors(self) -> None:
        from bqn_jax import evaluate

        with self.assertRaises(SyntaxError):
            evaluate("•missing")
        with self.assertRaises(ValueError):
            evaluate("•pi ← 3")

    def test_fold_identity_for_boolean_primitives(self) -> None:
        from bqn_jax import evaluate

        conj_empty = evaluate("∧´ ⟨⟩")
        disj_empty = evaluate("∨´ ⟨⟩")
        self.assertEqual(int(conj_empty), 1)
        self.assertEqual(int(disj_empty), 0)


if __name__ == "__main__":
    unittest.main()
