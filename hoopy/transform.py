"""
This module is responsible for all operator-related token & AST transformations.
"""

from __future__ import annotations

import abc
import ast
import keyword
import token
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import Any, Mapping, NamedTuple, Sequence

from . import tokens
from .runtime import (
    BLACKLISTED_OPERATOR_STRINGS,
    OperatorKind,
    is_builtin_with_kind,
    operator_proxy_for,
)


def desugar_infix(op: str, left: ast.expr, right: ast.expr) -> ast.Call:
    """Generates an AST node of the form
    ```
    __operator__(__name__, op)(left, right)
    ```
    given values of `op`, `left` and `right`.
    """
    return ast.Call(
        ast.Call(
            ast.Name("__operator__", ctx=ast.Load()),
            args=[ast.Name("__name__", ctx=ast.Load()), ast.Constant(op)],
            keywords=[],
        ),
        args=[left, right],
        keywords=[],
    )


def magic_imports() -> ast.ImportFrom:
    """Generates an AST node importing the special dunder methods from `hoopy.magic`."""
    return ast.ImportFrom(module="hoopy.magic", names=[ast.alias(name="*")])


def mangle_operator_string(toks: Sequence[TokenInfo], nonce: str) -> str | None:
    """Returns the mangled representation of a token, or None if the tokens
    don't constitute a valid operator.

    Reasons for failure include:

    * Disallowed token types (`$2$` is invalid, `$.$` is valid)
    * Spaces between non-keyword tokens (`+ +` is invalid, `is not` is valid)
    * Specifically blacklisted tokens (`:=` is invalid, `:=:` is valid)
    """
    if not toks:
        return None

    if 1 <= len(toks) <= 2 and all(
        tokens.is_keyword_operator_token(tok) for tok in toks
    ):
        op_string = " ".join(tok.string for tok in toks)
        if not is_builtin_with_kind(op_string, OperatorKind.Keyword):
            return None

    else:
        prev = None
        operator: list[str] = []
        for tok in toks:
            if not tokens.is_custom_operator_token(tok) or (
                prev is not None and not tokens.is_adjacent(prev, tok)
            ):
                return None
            operator.append(tok.string)
            prev = tok

        op_string = "".join(operator)
        if op_string in BLACKLISTED_OPERATOR_STRINGS:
            return None

    return f"__operator_{nonce}_{''.join(f'{ord(c):x}' for c in op_string)}"


def mangle_operator_objects_inplace(toks: list[TokenInfo], nonce: str) -> None:
    """Substitutes all instances of objectified operators (e.g. `(+%+)`)
    with their hex representations, such as ` __operator_0f198f1a_2b252b `
    given the nonce `"0f198f1a"`. Notice the extra spaces around the identifiers.
    This is kept to ensure no tokens are accidentally joined together after this
    transformation.
    """
    tokens.remove_error_whitespace_inplace(toks)
    start = None
    operators: list[tuple[int, int, str]] = []
    for i in range(len(toks)):
        if toks[i].exact_type == token.LPAR:
            start = i
        if toks[i].exact_type == token.RPAR and start is not None:
            mangled = mangle_operator_string(toks[start + 1 : i], nonce)
            if mangled is not None:
                operators.append((start + 1, i, mangled))
            start = None
    for start, end, mangled in reversed(operators):
        tokens.remove_inplace(toks, start - 1, end + 1)
        tokens.insert_inplace(
            toks, start - 1, token.NAME, mangled, left_offset=1, right_offset=1
        )


class Spans(NamedTuple):
    left: tuple[int, int]
    right: tuple[int, int]


# No ergonomic ADTs? :'c


class Operator(abc.ABC):
    pass


@dataclass
class Application(Operator):
    pass


@dataclass
class Identifier(Operator):
    content: str


@dataclass
class Custom(Operator):
    content: str


@dataclass
class Inplace(Operator):
    content: str


DEFAULT_OPERATOR = "*"


def collect_operator_tokens_inplace(
    toks: list[TokenInfo],
) -> Mapping[Spans, Operator]:
    """Substitutes instances of infix identifiers (e.g. \\`foo`),
    implicit function application (e.g. `f x`), custom operators
    (e.g. `<$>`) with generic operators, and in-place operators
    (e.g. `>>=`) with their non-inplace variants, returning
    information about the transformed regions.
    """
    tokens.remove_error_whitespace_inplace(toks)

    # TODO: this is an imperative implementation of a parser for a regular
    # language. If only there were some form of regex for token streams...
    collected_spans: dict[Spans, Operator] = {}
    i = 0
    while i < len(toks):
        # Function application
        # TODO: handle newline-delimited application by keeping a stack of [](){}
        match toks[i : i + 2]:
            case [
                TokenInfo(end=left) as first,
                TokenInfo() as second,
            ] if tokens.is_expression_ender(
                first
            ) and tokens.is_always_expression_starter(
                second
            ):
                tokens.insert_inplace(
                    toks, i + 1, token.OP, DEFAULT_OPERATOR, left_offset=1
                )
                right = toks[i + 2].start
                collected_spans[Spans(left, right)] = Application()
                # 2 since we added 1 extra token we don't need to process anymore
                i += 2
                continue

        # Infixified identifiers:
        # a 3-token pattern replaced with a 1-token pattern,
        # checking 1 token on each end for lookaround
        match toks[i - 1 : i + 4]:
            case [
                TokenInfo(end=(left_row, left_col)),
                TokenInfo(string="`", start=(_, start_col)),
                TokenInfo(type=token.NAME, string=string),
                TokenInfo(string="`"),
                TokenInfo(),
            ] if not keyword.iskeyword(string):
                tokens.remove_inplace(toks, i, i + 3)
                # after mutation
                right = toks[i + 1].start
                offset = start_col - left_col
                tokens.insert_inplace(
                    toks,
                    i,
                    token.OP,
                    DEFAULT_OPERATOR,
                    left_offset=offset,
                    right_offset=-offset,
                )
                collected_spans[Spans((left_row, left_col), right)] = Identifier(string)
                i += 1
                continue

        # Custom infix operators can be of variable width
        # This also catches in-place operators
        current = toks[i]
        operator = [current]
        if 0 < i < len(toks) - 1 and tokens.is_custom_operator_token(current):
            # slurp the next token while we can
            while i + len(operator) + 1 < len(toks):
                next = toks[i + len(operator)]
                if tokens.is_custom_operator_token(next) and tokens.is_adjacent(
                    current, next
                ):
                    operator.append(next)
                    current = next
                else:
                    break

            op_string = "".join(tok.string for tok in operator)
            if (
                op_string not in BLACKLISTED_OPERATOR_STRINGS
                and not is_builtin_with_kind(op_string, OperatorKind.Symbolic)
            ):

                tokens.remove_inplace(toks, i, i + len(operator))
                left_row, left_col = toks[i - 1].end
                offset = operator[0].start[1] - left_col

                proxy = operator_proxy_for(op_string)
                bonus = proxy.isalpha()

                tokens.insert_inplace(
                    # We need an extra space of offset for token hygeine,
                    # since the proxy may return `and` and `or`
                    toks,
                    i,
                    token.OP,
                    proxy,
                    left_offset=offset + bonus,
                    right_offset=-offset + bonus,
                )
                right = toks[i + 1].start

                kind = (
                    Inplace
                    if is_builtin_with_kind(op_string, OperatorKind.Inplace)
                    else Custom
                )

                collected_spans[Spans((left_row, left_col), right)] = kind(op_string)

                i += 1
                continue

        # no patterns succeeded
        i += 1

    return collected_spans


class HoopyTransformer(ast.NodeTransformer):
    """Class responsible for AST node manipulation."""

    def __init__(self, custom_spans: Mapping[Spans, Operator]) -> None:
        super().__init__()
        self.spans = custom_spans

    def visit_Module(self, node: ast.Module) -> Any:
        # visit child nodes first
        node = self.generic_visit(node)  # type: ignore

        # The magic import `from hoopy.magic import *` must go after
        # the module docstring and any __future__ imports

        stmts = node.body
        # this evaluates to 0 or 1 depending on if there's a module docstring
        offset = 0
        match stmts:
            case [ast.Expr(value=ast.Constant(value=str()))]:
                offset += 1

        # skip past every __future__ import
        while offset < len(stmts):
            match stmts[offset]:
                case ast.ImportFrom(module="__future__"):
                    offset += 1

        stmts.insert(offset, magic_imports())
        return ast.Module(body=stmts, type_ignores=node.type_ignores)
