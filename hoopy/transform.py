"""
This module is responsible for all operator-related token & AST transformations.
"""

from __future__ import annotations
import abc

import ast
from dataclasses import dataclass
from enum import IntEnum
import token
from tokenize import TokenInfo
from typing import Mapping, NamedTuple, Sequence
import keyword

from . import tokens


def desugar_infix(op: str, left: ast.expr, right: ast.expr) -> ast.Call:
    """
    Generates an AST node of the form
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


SYMBOL_TOKEN_STRINGS = set(token.EXACT_TOKEN_TYPES)
for non_op in ",;()[]{}":
    SYMBOL_TOKEN_STRINGS.remove(non_op)
for extra_op in "$!?":
    SYMBOL_TOKEN_STRINGS.add(extra_op)

KEYWORD_TOKEN_STRINGS = {"is", "not", "and", "or", "in"}

BLACKLISTED_OPERATOR_STRINGS = {"...", ".", ":", "::", ":=", "=", "~"}


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

    # TODO: handle keyword operators in a cleaner way
    # This matches:
    # * not
    # * and
    # * or
    # * is
    # * in
    # * is not
    # * not in
    if (len(toks) == 1 and toks[0].string in KEYWORD_TOKEN_STRINGS) or (
        len(toks) == 2
        and (
            (toks[0].string == "not" and toks[1].string == "in")
            or (toks[0].string == "is" and toks[1].string == "not")
        )
    ):
        op_string = " ".join(tok.string for tok in toks)

    else:
        end = None
        operator: list[str] = []
        for tok in toks:
            if tok.string not in SYMBOL_TOKEN_STRINGS or (
                end is not None and tok.start[1] != end
            ):
                return None
            operator.append(tok.string)
            end = tok.end[1]
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


# No ergonomic ADTs?


class Operator(abc.ABC):
    pass


@dataclass
class Application(Operator):
    pass


@dataclass
class Identifier(Operator):
    content: str


@dataclass
class Custom:
    content: str


def collect_operator_tokens_inplace(
    toks: list[TokenInfo],
) -> Mapping[Spans, Operator]:
    """Substitutes instances of infix identifiers (e.g. \\`foo`),
    implicit function application (e.g. `f x`), and custom operators
    (e.g. `<$>`) with generic operators, returning information about
    the transformed regions.
    """
    tokens.remove_error_whitespace_inplace(toks)

    # TODO: this is an imperative implementation of a parser for a regular
    # language. If only there were some form of regex for token streams...
    collected_spans: dict[Spans, Operator] = {}
    i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            first = toks[i]
            second = toks[i + 1]
            if tokens.expression_ender(first) and tokens.always_expression_starter(
                second
            ):
                left = first.end
                tokens.insert_inplace(toks, i + 1, token.OP, "@", left_offset=1)
                right = toks[i + 2].start
                collected_spans[Spans(left, right)] = Application()
                # 2 since we added 1 extra token we don't want to process
                i += 2
                continue

        # Infixified identifiers:
        # a 3-token pattern replaced with a 1-token pattern,
        # checking 1 token on each end for lookaround
        if i > 0 and i + 3 < len(toks):
            first = toks[i]
            second = toks[i + 1]
            third = toks[i + 2]
            if (
                first.string == "`"
                and second.type == token.NAME
                and not keyword.iskeyword(second.string)
                and third.string == "`"
            ):

                tokens.remove_inplace(toks, i, i + 3)

                left_row, left_col = toks[i - 1].end
                offset = first.start[1] - left_col
                tokens.insert_inplace(
                    toks, i, token.OP, "@", left_offset=offset, right_offset=-offset
                )
                right = toks[i + 1].start

                collected_spans[Spans((left_row, left_col), right)] = Identifier(
                    second.string
                )

                i += 1
                continue

        i += 1

    return collected_spans
