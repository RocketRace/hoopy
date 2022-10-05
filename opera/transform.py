"""
This module is responsible for all operator-related token & AST transformations.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import token
from tokenize import TokenInfo
from typing import Sequence
import keyword

from .tokens import insert_inplace, remove_inplace, remove_error_whitespace_inplace


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
        remove_inplace(toks, start - 1, end + 1)
        insert_inplace(
            toks, start - 1, token.NAME, mangled, left_offset=1, right_offset=1
        )


@dataclass(unsafe_hash=True)
class OperatorSpan:
    """Left and right endpoints of the surrounding tokens"""

    left_row: int
    left_col: int
    right_row: int
    right_col: int


def collect_infix_identifiers_inplace(
    toks: list[TokenInfo],
) -> dict[OperatorSpan, str]:
    """Substitutes instances of infix identifiers (e.g. \\`foo`)
    with generic @ operators, returning the transformed
    identifiers mapped from the endpoints of the previous and next
    token spans (to maintain the information). Does not touch the
    spans of other tokens!

    # Example
    Transforms the input tokens "foo\\`bar\\`baz" with "foo@baz",
    returning the dictionary `{ OperatorSpan(1, 3, 1, 4): "bar" }`
    """
    remove_error_whitespace_inplace(toks)
    if len(toks) < 5:
        return {}

    # TODO: this is an imperative implementation of a fairly
    # simple functional iterator routine, but I'm not confident
    # in Python's ability to provide ergonomic or effectient
    # iterator ops even when using the itertools module...
    collected_spans: dict[OperatorSpan, str] = {}
    removed = 0
    i = 1
    while i < len(toks) - 4 - removed:
        first = toks[i]
        second = toks[i + 1]
        third = toks[i + 2]
        if (
            first.string == "`"
            and second.type == token.NAME
            and not keyword.iskeyword(second.string)
            and third.string == "`"
        ):
            remove_inplace(toks, i, i + 3)
            removed += 2

            left_row, left_col = toks[i - 1].end
            right_row, right_col = toks[i + 1].start
            offset = first.start[1] - left_col
            insert_inplace(
                toks, i, token.OP, "@", left_offset=offset, right_offset=-offset
            )

            collected_spans[
                OperatorSpan(left_row, left_col, right_row, right_col)
            ] = second.string

        i += 1

    return collected_spans
