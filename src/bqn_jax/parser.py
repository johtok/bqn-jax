"""Parser for an expanded (still partial) BQN subset."""

from __future__ import annotations

from dataclasses import dataclass

from .ast import Assign, Block, Call, Case, Char, Export, Expr, HeaderTarget, Infix, Member, Mod1, Mod2, Name, Nothing, Null, Number, Prefix, Program, String, Train, Vector
from .lexer import Token, tokenize

_NAMED_ALIASES = {
    "shape": "≢",
    "range": "↕",
    "ravel": "⥊",
    "reshape": "⥊",
    "replicate": "/",
    "floor": "⌊",
    "ceil": "⌈",
    "length": "≠",
    "depth": "≡",
}

_MONADIC_OPS = {
    "+",
    "-",
    "×",
    "÷",
    "⋆",
    "!",
    "¬",
    "⌽",
    "⍉",
    "≢",
    "↕",
    "⥊",
    "⌊",
    "⌈",
    "|",
    "√",
    "⋈",
    "=",
    "≠",
    "≡",
    "≍",
    "<",
    ">",
    "«",
    "»",
    "⊣",
    "⊢",
    "⊔",
    "∧",
    "∨",
    "⍋",
    "⍒",
    "∊",
    "⊐",
    "⊒",
    "⍷",
}
_DYADIC_OPS = {
    "+",
    "-",
    "×",
    "÷",
    "⋆",
    "<",
    "≤",
    ">",
    "≥",
    "=",
    "⥊",
    "⌊",
    "⌈",
    "|",
    "√",
    "≠",
    "≡",
    "≍",
    "⋈",
    "⊣",
    "⊢",
    "/",
    "«",
    "»",
    "↑",
    "↓",
    "⌽",
    "⍉",
    "!",
    "∾",
    "⊑",
    "⊏",
    "⊐",
    "⊒",
    "⊔",
    "∊",
    "⍷",
    "∧",
    "∨",
    "⍋",
    "⍒",
}
_MOD2_OPS = {"∘", "○", "⊸", "⟜", "⊘", "◶", "⌾", "⎉", "⚇", "⍟", "⎊"}
_MOD1_OPS = {"˙", "˜", "¨", "˘", "⌜", "˝", "⁼", "´", "`"}
_SPECIAL_HEADER_NAMES = {"𝕨", "𝕎", "𝕩", "𝕏", "𝕗", "𝔽", "𝕘", "𝔾", "𝕤", "𝕊", "𝕣", "_𝕣", "_𝕣_"}

_ROLE_SUBJECT = "subject"
_ROLE_FUNCTION = "function"
_ROLE_MOD1 = "modifier1"
_ROLE_MOD2 = "modifier2"


class ParseError(SyntaxError):
    def __init__(
        self,
        message: str,
        start: int,
        end: int,
        expected: tuple[str, ...] = (),
        found: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.start = start
        self.end = end
        self.expected = expected
        self.found = found

    def __str__(self) -> str:
        expected_text = ""
        if self.expected:
            expected_text = f"; expected {', '.join(self.expected)}"
        found_text = ""
        if self.found is not None:
            found_text = f"; found {self.found}"
        return f"{self.message} at span [{self.start}, {self.end}){expected_text}{found_text}"


@dataclass
class _Parser:
    tokens: list[Token]
    index: int = 0

    def parse_program(self) -> Program:
        statements: list[Expr] = []
        self._consume_separators()
        while self._peek().kind != "EOF":
            statements.append(self._parse_statement())
            self._consume_separators()
        return Program(statements=tuple(statements))

    def parse_expression_only(self) -> Expr:
        expr = self._parse_statement()
        self._expect("EOF")
        return expr

    def _peek(self) -> Token:
        return self.tokens[self.index]

    def _peek_next(self) -> Token:
        return self.tokens[self.index + 1]

    def _advance(self) -> Token:
        tok = self.tokens[self.index]
        self.index += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            self._error(tok, expected=(kind,))
        return self._advance()

    def _error(self, tok: Token | None = None, *, message: str | None = None, expected: tuple[str, ...] = ()) -> None:
        token = tok if tok is not None else self._peek()
        detail = message if message is not None else "Unexpected token"
        normalized_expected = tuple(dict.fromkeys(expected))
        if token.kind == "EOF":
            found = "EOF"
        elif token.text:
            found = f"{token.kind}({token.text})"
        else:
            found = token.kind
        raise ParseError(detail, token.pos, token.end, expected=normalized_expected, found=found)

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self._advance()
            return True
        return False

    def _consume_separators(self, *, include_semi: bool = True) -> None:
        if include_semi:
            while self._peek().kind in {"SEP", "SEMI"}:
                self._advance()
            return
        while self._peek().kind == "SEP":
            self._advance()

    def _can_start_prefix(self, tok: Token) -> bool:
        if tok.kind in {"NUMBER", "CHAR", "STRING", "NULL", "NOTHING", "NAME", "LPAREN", "LANGLE", "LBRACK", "LBRACE"}:
            return True
        primitive = self._as_primitive(tok)
        return primitive is not None and primitive in _MONADIC_OPS

    def _can_start_application_arg(self, tok: Token) -> bool:
        return tok.kind in {"NUMBER", "CHAR", "STRING", "NULL", "NOTHING", "NAME", "LPAREN", "LANGLE", "LBRACK", "LBRACE"}

    def _name_role(self, name: str) -> str:
        ident = name[1:] if name.startswith("•") else name
        if ident.startswith("_"):
            return _ROLE_MOD2 if ident.endswith("_") and len(ident) > 1 else _ROLE_MOD1
        if not ident:
            return _ROLE_SUBJECT
        first = ident[0]
        if first.isupper():
            return _ROLE_FUNCTION
        if first.islower():
            return _ROLE_SUBJECT
        return _ROLE_SUBJECT

    def _name_is_function_role(self, name: str) -> bool:
        return self._name_role(name) == _ROLE_FUNCTION

    def _expr_is_function_role(self, expr: Expr) -> bool:
        if isinstance(expr, Name):
            return (
                expr.value in _MONADIC_OPS
                or expr.value in _DYADIC_OPS
                or self._name_is_function_role(expr.value)
            )
        if isinstance(expr, Member):
            return self._name_is_function_role(expr.attr)
        return isinstance(expr, (Block, Call, Mod1, Mod2, Train))

    def _is_callable_expr(self, expr: Expr) -> bool:
        return self._expr_is_function_role(expr)

    def _can_start_callable_token(self, tok: Token) -> bool:
        if tok.kind == "NAME":
            return self._name_is_function_role(tok.text)
        if tok.kind in {"NUMBER", "CHAR", "STRING", "NULL", "NOTHING", "NAME", "LPAREN", "LANGLE", "LBRACK", "LBRACE"}:
            return True
        return self._as_primitive(tok) is not None

    def _as_primitive(self, tok: Token) -> str | None:
        if tok.kind == "PRIM_FN":
            return tok.text
        if tok.kind == "NAME":
            return _NAMED_ALIASES.get(tok.text)
        return None

    def _dyadic_binding_power(self, op: str) -> tuple[int, int]:
        if op in _DYADIC_OPS:
            # Right-associative with uniform precedence.
            return (10, 10)
        self._error(message=f"Unknown dyadic operator {op!r}")
        raise AssertionError("unreachable")

    def _parse_statement(self) -> Expr:
        if self._peek().kind == "ASSIGN" and self._peek().text == "⇐":
            self._advance()
            return Export()

        left = self._parse_expression(0)
        if self._peek().kind == "ASSIGN":
            arrow = self._advance()
            if arrow.text == "⇐" and self._peek().kind in {"SEP", "SEMI", "RBRACE", "EOF"}:
                return Export(target=left)
            right = self._parse_statement()
            return Assign(op=arrow.text, left=left, right=right)
        return left

    def _parse_expression(self, min_bp: int) -> Expr:
        left = self._parse_prefix()

        while True:
            updated = self._parse_callable_derivation(left)
            if updated is not None:
                left = updated
                continue

            trained = self._parse_train_tail(left)
            if trained is not left:
                left = trained
                continue

            peek_tok = self._peek()
            peek_next_tok = self.tokens[self.index + 1] if (self.index + 1) < len(self.tokens) else peek_tok
            peek_primitive = self._as_primitive(peek_tok)
            # Permit dyadic application with derived primitive functions such as -˜ or ⋆⁼.
            # Keep the +´/×´ infix fold path by excluding PRIM_MOD1 ´ here.
            starts_modified_primitive_callable = (
                peek_primitive is not None
                and (
                    (peek_next_tok.kind == "PRIM_MOD1" and peek_next_tok.text != "´")
                    or peek_next_tok.kind == "PRIM_MOD2"
                )
            )

            # Dyadic application: 𝕨 F 𝕩
            if (
                min_bp <= 15
                and (not self._is_callable_expr(left))
                and (
                    (peek_primitive is None and self._can_start_application_arg(peek_tok))
                    or starts_modified_primitive_callable
                )
            ):
                save = self.index
                try:
                    func_candidate = self._parse_callable_term()
                    func_candidate = self._parse_train_tail(func_candidate)
                    if self._is_callable_expr(func_candidate) and self._can_start_application_arg(self._peek()):
                        right_arg = self._parse_prefix()
                        left = Call(func=func_candidate, left=left, right=right_arg)
                        continue
                except SyntaxError:
                    pass
                self.index = save

            # Monadic application: F 𝕩
            if min_bp <= 15 and self._is_callable_expr(left) and self._can_start_application_arg(self._peek()):
                right_arg = self._parse_prefix()
                left = Call(func=left, right=right_arg)
                continue

            tok = self._peek()
            primitive = self._as_primitive(tok)
            if primitive is None:
                break

            is_fold = self._peek_next().kind == "PRIM_MOD1" and self._peek_next().text == "´"
            op_name = f"´{primitive}" if is_fold else primitive
            if primitive not in _DYADIC_OPS:
                break

            lbp, rbp = self._dyadic_binding_power(primitive)
            if lbp < min_bp:
                break

            self._advance()
            if is_fold:
                self._advance()
            right = self._parse_expression(rbp)
            left = Infix(op=op_name, left=left, right=right)

        return left

    def _build_train_expr(self, terms: list[Expr]) -> Expr:
        if len(terms) < 2:
            return terms[0]
        if len(terms) == 2:
            return Train(parts=(terms[0], terms[1]))

        right = self._build_train_expr(terms[2:]) if len(terms) > 3 else terms[2]
        return Train(parts=(terms[0], terms[1], right))

    def _parse_train_tail(self, expr: Expr) -> Expr:
        if not self._is_callable_expr(expr):
            return expr

        terms = [expr]
        while self._can_start_callable_token(self._peek()):
            save = self.index
            try:
                next_term = self._parse_callable_term()
            except SyntaxError:
                self.index = save
                break
            if not self._is_callable_expr(next_term):
                self.index = save
                break
            terms.append(next_term)

        if len(terms) == 1:
            return expr
        return self._build_train_expr(terms)

    def _parse_callable_derivation(self, expr: Expr) -> Expr | None:
        if self._peek().kind == "PRIM_MOD1":
            mod_text = self._peek().text
            if mod_text == "˙":
                mod_tok = self._advance()
                return Mod1(op=mod_tok.text, operand=expr)
            if mod_text in _MOD1_OPS and self._is_callable_expr(expr):
                mod_tok = self._advance()
                return Mod1(op=mod_tok.text, operand=expr)

        if not self._is_callable_expr(expr):
            return None

        if self._peek().kind == "PRIM_MOD2" and self._peek().text in _MOD2_OPS:
            op_tok = self._advance()
            # 2-modifier derivation is left-associative and binds tighter than trains.
            # Its right operand is a single immediate subject/function term (not a train).
            right = self._parse_callable_base()
            return Mod2(op=op_tok.text, left=expr, right=right)

        return None

    def _parse_callable_base(self) -> Expr:
        tok = self._peek()
        primitive = self._as_primitive(tok)
        if primitive is not None:
            self._advance()
            return Name(value=primitive)
        return self._parse_strand()

    def _parse_callable_term(self) -> Expr:
        term = self._parse_prefix()
        while True:
            updated = self._parse_callable_derivation(term)
            if updated is None:
                break
            term = updated
        return term

    def _parse_prefix(self) -> Expr:
        tok = self._peek()
        primitive = self._as_primitive(tok)

        if primitive is not None and self._peek_next().kind in {"PRIM_MOD1", "PRIM_MOD2"}:
            self._advance()
            return Name(value=primitive)

        if primitive is not None and not self._can_start_prefix(self._peek_next()):
            self._advance()
            return Name(value=primitive)

        if primitive is not None and primitive in _MONADIC_OPS:
            self._advance()
            right = self._parse_prefix()
            return Prefix(op=primitive, right=right)

        return self._parse_strand()

    def _parse_strand(self) -> Expr:
        first = self._parse_atom()
        items = [first]
        while self._match("STRAND"):
            items.append(self._parse_atom())
        if len(items) == 1:
            return first
        return Vector(items=tuple(items))

    def _parse_atom(self) -> Expr:
        base = self._parse_atom_base()
        while self._match("DOT"):
            name_tok = self._peek()
            if name_tok.kind != "NAME":
                self._error(name_tok, expected=("NAME",))
            self._advance()
            base = Member(value=base, attr=name_tok.text)
        return base

    def _parse_atom_base(self) -> Expr:
        tok = self._peek()

        if tok.kind == "NUMBER":
            self._advance()
            if "j" in tok.text:
                return Number(value=complex(tok.text))
            return Number(value=float(tok.text))

        if tok.kind == "CHAR":
            self._advance()
            return Char(value=tok.text)

        if tok.kind == "STRING":
            self._advance()
            return String(value=tok.text)

        if tok.kind == "NULL":
            self._advance()
            return Null()

        if tok.kind == "NOTHING":
            self._advance()
            return Nothing()

        if tok.kind == "NAME":
            self._advance()
            return Name(value=tok.text)

        if self._match("LPAREN"):
            expr = self._parse_statement()
            self._expect("RPAREN")
            return expr

        if self._match("LANGLE"):
            return self._parse_vector_literal("RANGLE")

        if self._match("LBRACK"):
            return self._parse_vector_literal("RBRACK")

        if self._match("LBRACE"):
            return self._parse_block()

        self._error(tok, expected=("NUMBER", "CHAR", "STRING", "NULL", "NOTHING", "NAME", "LPAREN", "LANGLE", "LBRACK", "LBRACE"))
        raise AssertionError("unreachable")

    def _parse_vector_literal(self, closing_kind: str) -> Expr:
        items: list[Expr] = []
        self._consume_separators(include_semi=False)

        if self._match(closing_kind):
            return Vector(items=tuple())

        while True:
            items.append(self._parse_statement())
            consumed_sep = False
            while self._peek().kind == "SEP":
                self._advance()
                consumed_sep = True

            if self._match(closing_kind):
                return Vector(items=tuple(items))

            if consumed_sep:
                continue

            tok = self._peek()
            self._error(tok, expected=("SEP", closing_kind))

    def _parse_block(self) -> Expr:
        cases: list[Case] = []
        self._consume_separators(include_semi=False)
        block_end_tok: Token | None = None

        while True:
            header = self._parse_case_header_if_present()
            statements: list[Expr] = []
            predicates: list[Expr] = []
            saw_statement = False
            while self._peek().kind not in {"SEMI", "RBRACE", "EOF"}:
                item = self._parse_statement()
                if self._match("QMARK"):
                    if saw_statement:
                        self._error(message="Predicates must appear before statements in a case body")
                    predicates.append(item)
                else:
                    saw_statement = True
                    statements.append(item)
                self._consume_separators(include_semi=False)

            if not statements:
                self._error(message="Case body must end with a non-predicate statement")

            cases.append(
                Case(
                    header=header,
                    predicates=tuple(predicates),
                    body=Program(statements=tuple(statements)),
                )
            )

            if self._match("SEMI"):
                self._consume_separators(include_semi=False)
                continue
            if self._peek().kind == "RBRACE":
                block_end_tok = self._advance()
                break
            self._error(expected=("RBRACE",))

        assert block_end_tok is not None
        self._validate_block_cases(tuple(cases), block_end_tok)
        return Block(cases=tuple(cases))

    def _find_case_colon(self) -> int | None:
        paren_depth = 0
        angle_depth = 0
        bracket_depth = 0
        brace_depth = 0
        i = self.index
        while i < len(self.tokens):
            kind = self.tokens[i].kind
            if kind == "LPAREN":
                paren_depth += 1
            elif kind == "RPAREN":
                paren_depth = max(0, paren_depth - 1)
            elif kind == "LANGLE":
                angle_depth += 1
            elif kind == "RANGLE":
                angle_depth = max(0, angle_depth - 1)
            elif kind == "LBRACK":
                bracket_depth += 1
            elif kind == "RBRACK":
                bracket_depth = max(0, bracket_depth - 1)
            elif kind == "LBRACE":
                brace_depth += 1
            elif kind == "RBRACE":
                if brace_depth == 0 and paren_depth == 0 and angle_depth == 0 and bracket_depth == 0:
                    return None
                brace_depth = max(0, brace_depth - 1)
            elif kind in {"SEMI", "EOF"} and paren_depth == 0 and angle_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                return None
            elif kind == "COLON" and paren_depth == 0 and angle_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                return i
            i += 1
        return None

    def _parse_case_header_target_atom(self) -> HeaderTarget:
        tok = self._peek()
        if tok.kind == "NAME":
            self._advance()
            return Name(value=tok.text)
        if self._match("LANGLE"):
            return self._parse_case_header_list_literal("RANGLE")
        if self._match("LBRACK"):
            return self._parse_case_header_list_literal("RBRACK")
        self._error(tok, message="Unsupported case header token; expected assignment target", expected=("NAME", "LANGLE", "LBRACK"))
        raise AssertionError("unreachable")

    def _parse_case_header_target(self) -> HeaderTarget:
        first = self._parse_case_header_target_atom()
        items = [first]
        while self._match("STRAND"):
            items.append(self._parse_case_header_target_atom())
        if len(items) == 1:
            return first
        return Vector(items=tuple(items))

    def _parse_case_header_list_literal(self, closing_kind: str) -> Vector:
        items: list[HeaderTarget] = []
        self._consume_separators(include_semi=False)

        if self._match(closing_kind):
            return Vector(items=tuple())

        while True:
            items.append(self._parse_case_header_target())
            consumed_sep = False
            while self._peek().kind == "SEP":
                self._advance()
                consumed_sep = True

            if self._match(closing_kind):
                return Vector(items=tuple(items))

            if consumed_sep:
                continue

            tok = self._peek()
            self._error(tok, expected=("SEP", closing_kind))

    def _header_target_is_plain_name(self, target: HeaderTarget) -> bool:
        return isinstance(target, Name)

    def _parse_header_targets_from_tokens(self, tokens: tuple[Token, ...]) -> tuple[HeaderTarget, ...] | None:
        if not tokens:
            return tuple()

        eof_pos = tokens[-1].end
        sub = _Parser(tokens=[*tokens, Token("EOF", "", eof_pos, eof_pos)])
        targets: list[HeaderTarget] = []
        try:
            while sub._peek().kind != "EOF":
                if sub._peek().kind == "SEP":
                    sub._advance()
                    continue
                targets.append(sub._parse_case_header_target())
            return tuple(targets)
        except ParseError:
            return None

    def _parse_single_header_target_from_tokens(self, tokens: tuple[Token, ...]) -> HeaderTarget | None:
        if not tokens:
            return None
        eof_pos = tokens[-1].end
        sub = _Parser(tokens=[*tokens, Token("EOF", "", eof_pos, eof_pos)])
        try:
            sub._consume_separators(include_semi=False)
            target = sub._parse_case_header_target()
            sub._consume_separators(include_semi=False)
            if sub._peek().kind != "EOF":
                return None
            return target
        except ParseError:
            return None

    def _parse_explicit_function_header_from_tokens(self, tokens: tuple[Token, ...]) -> tuple[HeaderTarget, ...] | None:
        compact = tuple(tok for tok in tokens if tok.kind != "SEP")
        if not compact:
            return None

        for func_index, func_tok in enumerate(compact):
            if func_tok.kind != "NAME" or not self._name_is_function_role(func_tok.text):
                continue

            marker_index = func_index + 1
            saw_under = False

            if (
                marker_index < len(compact)
                and compact[marker_index].kind == "PRIM_MOD1"
                and compact[marker_index].text == "˜"
            ):
                saw_under = True
                marker_index += 1

            if (
                marker_index < len(compact)
                and compact[marker_index].kind == "PRIM_MOD1"
                and compact[marker_index].text == "⁼"
            ):
                marker_index += 1
            elif saw_under:
                continue

            right_tokens = compact[marker_index:]
            if not right_tokens:
                continue
            right_target = self._parse_single_header_target_from_tokens(right_tokens)
            if right_target is None:
                continue

            left_tokens = compact[:func_index]
            if not left_tokens:
                return (right_target,)

            left_target = self._parse_single_header_target_from_tokens(left_tokens)
            if left_target is None:
                continue
            return (left_target, right_target)

        return None

    def _parse_case_header_if_present(self) -> tuple[HeaderTarget, ...] | None:
        colon_index = self._find_case_colon()
        if colon_index is None:
            return None

        header_tokens = tuple(self.tokens[self.index:colon_index])
        first_header_tok = next((tok for tok in header_tokens if tok.kind != "SEP"), self.tokens[colon_index])
        if all(tok.kind == "SEP" for tok in header_tokens):
            self._error(first_header_tok, message="Case header cannot be empty")

        explicit = self._parse_explicit_function_header_from_tokens(header_tokens)
        if explicit is not None:
            targets = explicit
        else:
            parsed = self._parse_header_targets_from_tokens(header_tokens)
            if parsed is None:
                self._error(first_header_tok, message="Invalid case header")
            targets = parsed

            if not targets:
                self._error(first_header_tok, message="Case header cannot be empty")
            if len(targets) > 2:
                self._error(first_header_tok, message="Only monadic and dyadic case headers are supported")
            if len(targets) == 1 and self._header_target_is_plain_name(targets[0]):
                self._error(first_header_tok, message="Monadic header without function requires a non-name argument")
            if len(targets) == 2:
                self._error(first_header_tok, message="Dyadic header requires an explicit function token")

        self.index = colon_index
        self._expect("COLON")
        self._consume_separators(include_semi=False)
        return tuple(targets)

    def _header_target_uses_special_name(self, target: HeaderTarget) -> bool:
        if isinstance(target, Name):
            return target.value in _SPECIAL_HEADER_NAMES
        if isinstance(target, Vector):
            return any(self._header_target_uses_special_name(item) for item in target.items if isinstance(item, (Name, Vector)))
        return False

    def _expr_uses_special_name(self, expr: Expr) -> bool:
        if isinstance(expr, Name):
            return expr.value in _SPECIAL_HEADER_NAMES
        if isinstance(expr, Member):
            return self._expr_uses_special_name(expr.value)
        if isinstance(expr, Vector):
            return any(self._expr_uses_special_name(item) for item in expr.items)
        if isinstance(expr, Prefix):
            return self._expr_uses_special_name(expr.right)
        if isinstance(expr, Infix):
            return self._expr_uses_special_name(expr.left) or self._expr_uses_special_name(expr.right)
        if isinstance(expr, Mod1):
            return self._expr_uses_special_name(expr.operand)
        if isinstance(expr, Mod2):
            return self._expr_uses_special_name(expr.left) or self._expr_uses_special_name(expr.right)
        if isinstance(expr, Train):
            return any(self._expr_uses_special_name(part) for part in expr.parts)
        if isinstance(expr, Assign):
            return self._expr_uses_special_name(expr.left) or self._expr_uses_special_name(expr.right)
        if isinstance(expr, Export):
            return expr.target is not None and self._expr_uses_special_name(expr.target)
        if isinstance(expr, Block):
            return any(self._case_uses_special_name(case) for case in expr.cases)
        if isinstance(expr, Call):
            return (
                self._expr_uses_special_name(expr.func)
                or self._expr_uses_special_name(expr.right)
                or (expr.left is not None and self._expr_uses_special_name(expr.left))
            )
        return False

    def _is_general_case(self, case: Case) -> bool:
        return case.header is None and len(case.predicates) == 0

    def _case_uses_special_name(self, case: Case) -> bool:
        if case.header is not None and any(self._header_target_uses_special_name(target) for target in case.header):
            return True
        if any(self._expr_uses_special_name(pred) for pred in case.predicates):
            return True
        return any(self._expr_uses_special_name(stmt) for stmt in case.body.statements)

    def _validate_block_cases(self, cases: tuple[Case, ...], error_tok: Token) -> None:
        saw_general = False
        general_count = 0
        for case in cases:
            is_general = self._is_general_case(case)
            if is_general:
                saw_general = True
                general_count += 1
                continue
            if saw_general:
                self._error(error_tok, message="General block case cannot appear before non-general case")

        if general_count > 2:
            self._error(error_tok, message="Block cannot contain more than two general cases")

        if general_count > 1 and not any(self._case_uses_special_name(case) for case in cases):
            self._error(error_tok, message="Immediate block cannot contain multiple general cases")


def parse(source: str) -> Expr:
    tokens = tokenize(source)
    parser = _Parser(tokens=tokens)
    return parser.parse_expression_only()


def parse_program(source: str) -> Program:
    tokens = tokenize(source)
    parser = _Parser(tokens=tokens)
    return parser.parse_program()
