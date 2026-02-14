"""Tokenization for an expanded (still partial) BQN subset."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int
    end: int


_SINGLE_TOKENS = {
    "(": "LPAREN",
    ")": "RPAREN",
    "{": "LBRACE",
    "}": "RBRACE",
    "⟨": "LANGLE",
    "⟩": "RANGLE",
    "[": "LBRACK",
    "]": "RBRACK",
    "‿": "STRAND",
    ";": "SEMI",
    ":": "COLON",
    "?": "QMARK",
    ".": "DOT",
    "·": "NOTHING",
}

_OP_ALIASES = {
    "*": "×",
    "^": "⋆",
    "<=": "≤",
}

_ASSIGN_TOKENS = {"←", "↩", "⇐"}
_NUMERIC_START = set("¯∞π.0123456789")
_NUMERIC_BODY = set("¯∞π._0123456789eE")

_PRIMITIVE_FUNCTIONS = set(
    "+-×÷⋆√⌊⌈|¬∧∨"
    "<≤=≥>≠≡≢"
    "⊣⊢"
    "↕↑↓↩"
    "⌽⍉/«»"
    "⍋⍒⥊∾≍⋈"
    "⊏⊑⊐⊒∊⍷⊔!"
)
_PRIMITIVE_MOD1 = set("˙˜˘¨⌜⁼´˝`")
_PRIMITIVE_MOD2 = set("∘○⊸⟜⌾◶⎉⚇⊘⍟⎊")
_OP_ASCII = set("*^")

_SYSTEM_VALUES = {
    "pi",
    "e",
    "inf",
    "nan",
    "i",
    "true",
    "false",
    "cwd",
    "time",
    "BQN",
    "version",
    "policy",
    "Type",
    "Repr",
    "Fmt",
    "ParseFloat",
    "UnixTime",
    "MonoTime",
    "Delay",
    "type",
    "repr",
    "fmt",
    "parse_float",
    "unix_time",
    "mono_time",
    "delay",
}
_SYSTEM_VALUES_NORM = {name.replace("_", "").casefold() for name in _SYSTEM_VALUES}

_NUMBER_RE = re.compile(
    r"""
    ^
    (?P<sign>[¯-]?)                           # leading sign
    (?:
        (?P<infinity>∞)                       # infinity
      |
        (?P<mantissa>
            π
          |
            (?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)
        )
        (?:
            [eE]
            (?P<exp_sign>[¯+\-]?)
            (?P<exponent>[0-9]+)
        )?
    )
    $
    """,
    re.VERBOSE,
)


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def _is_ident_continue(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def _scan_while(source: str, start: int, predicate) -> tuple[str, int]:
    i = start
    while i < len(source) and predicate(source[i]):
        i += 1
    return source[start:i], i


def _normalize_word_name(name: str) -> str:
    return name.replace("_", "").casefold()


def _scan_digits_and_underscores(source: str, start: int) -> int:
    i = start
    while i < len(source) and (source[i].isdigit() or source[i] == "_"):
        i += 1
    return i


def _scan_real_number_component(source: str, start: int) -> int:
    i = start

    if source[i] in {"¯", "-"}:
        i += 1
        if i >= len(source):
            raise SyntaxError(f"Invalid numeric literal {source[start:i]!r} at index {start}")

    if source[i] == "∞":
        return i + 1

    if source[i] == "π":
        i += 1
        if i < len(source) and source[i] in {"e", "E"}:
            i += 1
            if i < len(source) and source[i] in {"¯", "-", "+"}:
                i += 1
            exp_start = i
            i = _scan_digits_and_underscores(source, i)
            if i == exp_start:
                raise SyntaxError(f"Invalid numeric literal {source[start:i]!r} at index {start}")
        return i

    if source[i] == ".":
        i += 1
        frac_start = i
        i = _scan_digits_and_underscores(source, i)
        if i == frac_start:
            raise SyntaxError(f"Invalid numeric literal {source[start:i]!r} at index {start}")
    else:
        int_start = i
        i = _scan_digits_and_underscores(source, i)
        if i == int_start:
            raise SyntaxError(f"Invalid numeric literal {source[start:i]!r} at index {start}")
        if i < len(source) and source[i] == ".":
            i += 1
            i = _scan_digits_and_underscores(source, i)

    if i < len(source) and source[i] in {"e", "E"}:
        i += 1
        if i < len(source) and source[i] in {"¯", "-", "+"}:
            i += 1
        exp_start = i
        i = _scan_digits_and_underscores(source, i)
        if i == exp_start:
            raise SyntaxError(f"Invalid numeric literal {source[start:i]!r} at index {start}")

    return i


def _scan_number(source: str, start: int) -> tuple[str, int]:
    if source[start] in {"i", "I"}:
        return source[start : start + 1], start + 1

    if source[start] in {"¯", "-"} and start + 1 < len(source) and source[start + 1] in {"i", "I"}:
        return source[start : start + 2], start + 2

    i = _scan_real_number_component(source, start)

    if i < len(source) and source[i] in {"i", "I"}:
        i += 1
        if i < len(source) and source[i] in _NUMERIC_START:
            i = _scan_real_number_component(source, i)

    return source[start:i], i


def _parse_real_number_text(text: str, pos: int) -> float:
    cleaned = text.replace("_", "")
    m = _NUMBER_RE.match(cleaned)
    if not m:
        raise SyntaxError(f"Invalid numeric literal {text!r} at index {pos}")

    sign = -1.0 if m.group("sign") in {"¯", "-"} else 1.0
    if m.group("infinity"):
        return sign * math.inf

    mantissa_text = m.group("mantissa")
    if mantissa_text == "π":
        mantissa = math.pi
    else:
        mantissa = float(mantissa_text)

    exponent_text = m.group("exponent")
    if exponent_text is not None:
        exp_sign = -1 if m.group("exp_sign") in {"¯", "-"} else 1
        exponent = exp_sign * int(exponent_text)
        mantissa *= 10.0**exponent

    return sign * mantissa


def _parse_number_text(text: str, pos: int) -> float | complex:
    cleaned = text.replace("_", "")
    if cleaned in {"i", "I"}:
        return complex(0.0, 1.0)

    if cleaned in {"¯i", "-i", "¯I", "-I"}:
        return complex(0.0, -1.0)

    imag_sep = next((ch for ch in ("i", "I") if ch in cleaned), None)
    if imag_sep is not None:
        left, right = cleaned.split(imag_sep, maxsplit=1)
        if right:
            real = 0.0 if left in {"", "+", "¯", "-"} else _parse_real_number_text(left, pos)
            imag = _parse_real_number_text(right, pos)
            return complex(real, imag)

    if cleaned.endswith(("i", "I")):
        imag_text = cleaned[:-1]
        if imag_text in {"", "+"}:
            imag = 1.0
        elif imag_text in {"¯", "-"}:
            imag = -1.0
        else:
            imag = _parse_real_number_text(imag_text, pos)
        return complex(0.0, imag)
    return _parse_real_number_text(cleaned, pos)


def _parse_escaped_codepoint(source: str, start: int) -> tuple[str, int]:
    if start >= len(source):
        raise SyntaxError("Escape sequence is incomplete at end of input")

    esc = source[start]
    if esc == "n":
        return "\n", start + 1
    if esc == "r":
        return "\r", start + 1
    if esc == "t":
        return "\t", start + 1
    if esc == "0":
        return "\0", start + 1
    if esc == "\\":
        return "\\", start + 1
    if esc == '"':
        return '"', start + 1
    if esc == "'":
        return "'", start + 1

    if esc == "x":
        hex_end = start + 3
        if hex_end > len(source):
            raise SyntaxError(f"Incomplete \\x escape at index {start - 1}")
        digits = source[start + 1 : hex_end]
        if not all(ch in "0123456789abcdefABCDEF" for ch in digits):
            raise SyntaxError(f"Invalid \\x escape at index {start - 1}")
        codepoint = int(digits, 16)
        try:
            return chr(codepoint), hex_end
        except ValueError as exc:
            raise SyntaxError(f"Invalid \\x escape at index {start - 1}") from exc

    if esc == "u":
        hex_end = start + 5
        if hex_end > len(source):
            raise SyntaxError(f"Incomplete \\u escape at index {start - 1}")
        digits = source[start + 1 : hex_end]
        if not all(ch in "0123456789abcdefABCDEF" for ch in digits):
            raise SyntaxError(f"Invalid \\u escape at index {start - 1}")
        codepoint = int(digits, 16)
        try:
            return chr(codepoint), hex_end
        except ValueError as exc:
            raise SyntaxError(f"Invalid \\u escape at index {start - 1}") from exc

    if esc == "U":
        hex_end = start + 9
        if hex_end > len(source):
            raise SyntaxError(f"Incomplete \\U escape at index {start - 1}")
        digits = source[start + 1 : hex_end]
        if not all(ch in "0123456789abcdefABCDEF" for ch in digits):
            raise SyntaxError(f"Invalid \\U escape at index {start - 1}")
        codepoint = int(digits, 16)
        try:
            return chr(codepoint), hex_end
        except ValueError as exc:
            raise SyntaxError(f"Invalid \\U escape at index {start - 1}") from exc

    raise SyntaxError(f"Unknown escape sequence \\{esc} at index {start - 1}")


def _scan_string(source: str, start: int) -> tuple[str, int]:
    assert source[start] == '"'
    i = start + 1
    out: list[str] = []
    while i < len(source):
        ch = source[i]
        if ch == '"':
            if i + 1 < len(source) and source[i + 1] == '"':
                out.append('"')
                i += 2
                continue
            return "".join(out), i + 1
        if ch == "\\":
            escaped, end = _parse_escaped_codepoint(source, i + 1)
            out.append(escaped)
            i = end
            continue
        out.append(ch)
        i += 1
    raise SyntaxError(f"Unterminated string literal at index {start}")


def _scan_char(source: str, start: int) -> tuple[str, int]:
    assert source[start] == "'"
    i = start + 1
    if i >= len(source):
        raise SyntaxError(f"Unterminated character literal at index {start}")

    ch = source[i]
    i += 1

    if i >= len(source) or source[i] != "'":
        raise SyntaxError(f"Character literal must contain exactly one code point at index {start}")
    return ch, i + 1


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    while i < len(source):
        ch = source[i]

        if ch in {" ", "\t", "\f", "\v"}:
            i += 1
            continue

        if ch == "#":
            while i < len(source) and source[i] not in {"\n", "\r"}:
                i += 1
            continue

        if ch in {"\n", "\r", "⋄", ","}:
            start = i
            while i < len(source) and source[i] in {"\n", "\r", "⋄", ","}:
                i += 1
            tokens.append(Token("SEP", "⋄", start, i))
            continue

        if source.startswith("<=", i):
            tokens.append(Token("PRIM_FN", _OP_ALIASES["<="], i, i + 2))
            i += 2
            continue

        if ch == "." and (i + 1 >= len(source) or not source[i + 1].isdigit()):
            tokens.append(Token("DOT", ch, i, i + 1))
            i += 1
            continue

        if ch in _SINGLE_TOKENS:
            if ch == "." and i + 1 < len(source) and source[i + 1].isdigit():
                text, end = _scan_number(source, i)
                value = _parse_number_text(text, i)
                tokens.append(Token("NUMBER", repr(value), i, end))
                i = end
                continue
            tokens.append(Token(_SINGLE_TOKENS[ch], ch, i, i + 1))
            i += 1
            continue

        if ch in _ASSIGN_TOKENS:
            tokens.append(Token("ASSIGN", ch, i, i + 1))
            i += 1
            continue

        if ch in _PRIMITIVE_MOD1:
            tokens.append(Token("PRIM_MOD1", ch, i, i + 1))
            i += 1
            continue

        if ch == "@":
            tokens.append(Token("NULL", ch, i, i + 1))
            i += 1
            continue

        if ch == '"':
            value, end = _scan_string(source, i)
            tokens.append(Token("STRING", value, i, end))
            i = end
            continue

        if ch == "'":
            value, end = _scan_char(source, i)
            tokens.append(Token("CHAR", value, i, end))
            i = end
            continue

        if ch in _PRIMITIVE_FUNCTIONS:
            tokens.append(Token("PRIM_FN", ch, i, i + 1))
            i += 1
            continue

        if ch in _PRIMITIVE_MOD2:
            tokens.append(Token("PRIM_MOD2", ch, i, i + 1))
            i += 1
            continue

        if ch in _OP_ASCII:
            tokens.append(Token("PRIM_FN", _OP_ALIASES[ch], i, i + 1))
            i += 1
            continue

        if ch == "•":
            if i + 1 < len(source) and _is_ident_start(source[i + 1]):
                ident_start = i + 1
                ident, end = _scan_while(source, ident_start, _is_ident_continue)
                if not any(part.isalpha() for part in ident):
                    raise SyntaxError(f"Invalid word name {ident!r} at index {ident_start}")
                if _normalize_word_name(ident) not in _SYSTEM_VALUES_NORM:
                    raise SyntaxError(f"Undefined system value {'•' + ident!r} at index {i}")
                tokens.append(Token("NAME", f"•{ident}", i, end))
                i = end
                continue
            tokens.append(Token("NAME", "•", i, i + 1))
            i += 1
            continue

        if ch in _NUMERIC_START:
            text, end = _scan_number(source, i)
            value = _parse_number_text(text, i)
            tokens.append(Token("NUMBER", repr(value), i, end))
            i = end
            continue

        if _is_ident_start(ch):
            start = i
            i += 1
            while i < len(source) and _is_ident_continue(source[i]):
                i += 1
            ident = source[start:i]
            if not any(part.isalpha() for part in ident):
                raise SyntaxError(f"Invalid word name {ident!r} at index {start}")
            tokens.append(Token("NAME", ident, start, i))
            continue

        raise SyntaxError(f"Unexpected character {ch!r} at index {i}")

    tokens.append(Token("EOF", "", len(source), len(source)))
    return tokens
