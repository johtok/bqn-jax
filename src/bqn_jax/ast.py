"""AST nodes for an expanded (still partial) BQN subset."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number as Numeric
from typing import Union


@dataclass(frozen=True)
class Number:
    value: Numeric


@dataclass(frozen=True)
class Char:
    value: str


@dataclass(frozen=True)
class String:
    value: str


@dataclass(frozen=True)
class Null:
    pass


@dataclass(frozen=True)
class Nothing:
    pass


@dataclass(frozen=True)
class Name:
    value: str


@dataclass(frozen=True)
class Member:
    value: "Expr"
    attr: str


@dataclass(frozen=True)
class Vector:
    items: tuple["Expr", ...]


@dataclass(frozen=True)
class Prefix:
    op: str
    right: "Expr"


@dataclass(frozen=True)
class Infix:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Mod1:
    op: str
    operand: "Expr"


@dataclass(frozen=True)
class Mod2:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Train:
    parts: tuple["Expr", ...]


@dataclass(frozen=True)
class Assign:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Export:
    target: "Expr | None" = None


@dataclass(frozen=True)
class Program:
    statements: tuple["Expr", ...]


@dataclass(frozen=True)
class Case:
    header: tuple["HeaderTarget", ...] | None
    predicates: tuple["Expr", ...]
    body: Program


@dataclass(frozen=True)
class Block:
    cases: tuple[Case, ...]


@dataclass(frozen=True)
class Call:
    func: "Expr"
    right: "Expr"
    left: "Expr | None" = None


HeaderTarget = Union[Name, Vector]
Expr = Union[Number, Char, String, Null, Nothing, Name, Member, Vector, Prefix, Infix, Mod1, Mod2, Train, Assign, Export, Block, Call]
