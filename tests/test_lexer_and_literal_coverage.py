from __future__ import annotations

import math
import unittest

from bqn_jax.ast import Char, Nothing, Null, Number, String, Vector
from bqn_jax.lexer import tokenize
from bqn_jax.parser import ParseError, parse


class LexerAndLiteralCoverageTests(unittest.TestCase):
    _EXPR_EXPECTED = ("NUMBER", "CHAR", "STRING", "NULL", "NOTHING", "NAME", "LPAREN", "LANGLE", "LBRACK", "LBRACE")

    def _tokens(self, source: str, *, with_spans: bool = False):
        if with_spans:
            return [(tok.kind, tok.text, tok.pos, tok.end) for tok in tokenize(source) if tok.kind != "EOF"]
        return [(tok.kind, tok.text) for tok in tokenize(source) if tok.kind != "EOF"]

    def test_token_golden_delimiters_assignments_and_member_spans(self) -> None:
        tokens = self._tokens('(){}⟨⟩[]‿;:?·←↩⇐@"s"\'c\'a.b', with_spans=True)
        self.assertEqual(
            tokens,
            [
                ("LPAREN", "(", 0, 1),
                ("RPAREN", ")", 1, 2),
                ("LBRACE", "{", 2, 3),
                ("RBRACE", "}", 3, 4),
                ("LANGLE", "⟨", 4, 5),
                ("RANGLE", "⟩", 5, 6),
                ("LBRACK", "[", 6, 7),
                ("RBRACK", "]", 7, 8),
                ("STRAND", "‿", 8, 9),
                ("SEMI", ";", 9, 10),
                ("COLON", ":", 10, 11),
                ("QMARK", "?", 11, 12),
                ("NOTHING", "·", 12, 13),
                ("ASSIGN", "←", 13, 14),
                ("ASSIGN", "↩", 14, 15),
                ("ASSIGN", "⇐", 15, 16),
                ("NULL", "@", 16, 17),
                ("STRING", "s", 17, 20),
                ("CHAR", "c", 20, 23),
                ("NAME", "a", 23, 24),
                ("DOT", ".", 24, 25),
                ("NAME", "b", 25, 26),
            ],
        )

    def test_token_golden_separators_comments_and_ascii_aliases(self) -> None:
        tokens = self._tokens("a,⋄\n# comment\nb * ^ <= .5", with_spans=True)
        self.assertEqual(
            tokens,
            [
                ("NAME", "a", 0, 1),
                ("SEP", "⋄", 1, 4),
                ("SEP", "⋄", 13, 14),
                ("NAME", "b", 14, 15),
                ("PRIM_FN", "×", 16, 17),
                ("PRIM_FN", "⋆", 18, 19),
                ("PRIM_FN", "≤", 20, 22),
                ("NUMBER", "0.5", 23, 25),
            ],
        )

    def test_token_classes_for_all_supported_primitive_glyphs(self) -> None:
        primitive_glyphs = "+-×÷⋆√⌊⌈|¬∧∨<≤=≥>≠≡≢⊣⊢↕↑↓⌽⍉/«»⍋⍒⥊∾≍⋈⊏⊑⊐⊒∊⍷⊔!"
        for glyph in primitive_glyphs:
            with self.subTest(glyph=glyph):
                kind, text = self._tokens(glyph)[0]
                self.assertEqual(kind, "PRIM_FN")
                self.assertEqual(text, glyph)

        self.assertEqual(self._tokens("↩"), [("ASSIGN", "↩")])

    def test_token_classes_for_all_supported_modifier_glyphs(self) -> None:
        mod1_glyphs = "˙˜˘¨⌜⁼´˝`"
        mod2_glyphs = "∘○⊸⟜⌾◶⎉⚇⊘⍟⎊"

        for glyph in mod1_glyphs:
            with self.subTest(glyph=glyph):
                self.assertEqual(self._tokens(glyph), [("PRIM_MOD1", glyph)])

        for glyph in mod2_glyphs:
            with self.subTest(glyph=glyph):
                self.assertEqual(self._tokens(glyph), [("PRIM_MOD2", glyph)])

    def test_token_classes_for_names_and_system_names(self) -> None:
        tokens = self._tokens("• •pi •Type _x x1")
        self.assertEqual(
            tokens,
            [
                ("NAME", "•"),
                ("NAME", "•pi"),
                ("NAME", "•Type"),
                ("NAME", "_x"),
                ("NAME", "x1"),
            ],
        )
        with self.assertRaises(SyntaxError):
            tokenize("•_tmp")
        with self.assertRaises(SyntaxError):
            tokenize("_99")

    def test_string_escape_and_double_quote_lexing(self) -> None:
        tokens = self._tokens('"a""b\\n\\t\\x41\\u03B1\\U0001F600"')
        self.assertEqual(tokens, [("STRING", 'a"b\n\tAα😀')])

    def test_char_literal_requires_single_unescaped_codepoint(self) -> None:
        self.assertEqual(self._tokens("'a'"), [("CHAR", "a")])
        self.assertEqual(self._tokens("'α'"), [("CHAR", "α")])
        for source in ("'\\n'", "'\\x41'", "'\\u03B1'", "'\\U0001F600'"):
            with self.subTest(source=source):
                with self.assertRaises(SyntaxError):
                    tokenize(source)

    def test_numeric_literal_lexing_edge_forms(self) -> None:
        cases = {
            ".5": 0.5,
            "1.": 1.0,
            "1e+3": 1000.0,
            "¯2e¯1": -0.2,
            "2i": 2j,
            "1i2": complex(1, 2),
            "1I¯2": complex(1, -2),
            "¯i": -1j,
            "1_2_3": 123.0,
            "π": math.pi,
            "¯∞": -math.inf,
            ".5i": 0.5j,
        }
        for literal, expected in cases.items():
            tokens = self._tokens(literal)
            self.assertEqual(len(tokens), 1)
            kind, text = tokens[0]
            self.assertEqual(kind, "NUMBER")
            if isinstance(expected, complex):
                self.assertEqual(complex(text), expected)
            elif math.isinf(expected):
                self.assertTrue(math.isinf(float(text)))
                self.assertLess(float(text), 0.0)
            else:
                self.assertAlmostEqual(float(text), expected, places=12)

    def test_parser_literal_forms_with_escapes_and_vector_items(self) -> None:
        parsed_str = parse('"a\\n\\u03B1"')
        self.assertIsInstance(parsed_str, String)
        assert isinstance(parsed_str, String)
        self.assertEqual(parsed_str.value, "a\nα")

        parsed_char = parse("'t'")
        self.assertIsInstance(parsed_char, Char)
        assert isinstance(parsed_char, Char)
        self.assertEqual(parsed_char.value, "t")

        parsed_num = parse(".5")
        self.assertIsInstance(parsed_num, Number)
        assert isinstance(parsed_num, Number)
        self.assertAlmostEqual(parsed_num.value, 0.5, places=12)

        parsed_vector = parse('⟨"x" ⋄ \'y\' ⋄ .5 ⋄ @ ⋄ ·⟩')
        self.assertIsInstance(parsed_vector, Vector)
        assert isinstance(parsed_vector, Vector)
        self.assertEqual(len(parsed_vector.items), 5)
        self.assertIsInstance(parsed_vector.items[0], String)
        self.assertIsInstance(parsed_vector.items[1], Char)
        self.assertIsInstance(parsed_vector.items[2], Number)
        self.assertIsInstance(parsed_vector.items[3], Null)
        self.assertIsInstance(parsed_vector.items[4], Nothing)

    def test_parser_nothing_and_null_literals(self) -> None:
        self.assertIsInstance(parse("·"), Nothing)
        self.assertIsInstance(parse("@"), Null)

    def test_parse_error_reports_found_expected_and_spans(self) -> None:
        with self.assertRaises(ParseError) as cm:
            parse("1 + )")
        err = cm.exception
        self.assertEqual(err.message, "Unexpected token")
        self.assertEqual((err.start, err.end), (4, 5))
        self.assertEqual(err.expected, self._EXPR_EXPECTED)
        self.assertEqual(err.found, "RPAREN())")
        rendered = str(err)
        self.assertIn("expected", rendered)
        self.assertIn("found", rendered)

    def test_parse_error_for_vector_separator_requirement(self) -> None:
        with self.assertRaises(ParseError) as cm:
            parse("⟨1 2⟩")
        err = cm.exception
        self.assertEqual(err.message, "Unexpected token")
        self.assertEqual((err.start, err.end), (3, 4))
        self.assertEqual(err.expected, ("SEP", "RANGLE"))
        self.assertEqual(err.found, "NUMBER(2.0)")

    def test_parse_error_for_unsupported_case_header_token(self) -> None:
        with self.assertRaises(ParseError) as cm:
            parse("{1:2}")
        err = cm.exception
        self.assertEqual(err.message, "Invalid case header")
        self.assertEqual((err.start, err.end), (1, 2))
        self.assertEqual(err.expected, ())
        self.assertEqual(err.found, "NUMBER(1.0)")

    def test_invalid_escape_reports_source_location(self) -> None:
        error_cases = (
            ('"bad\\qescape"', "Unknown escape sequence"),
            ('"\\U11000000"', "Invalid \\U escape"),
            ("'ab'", "exactly one code point"),
            ("1e", "Invalid numeric literal"),
        )
        for source, message in error_cases:
            with self.subTest(source=source):
                with self.assertRaises(SyntaxError) as cm:
                    parse(source)
                self.assertIn(message, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
