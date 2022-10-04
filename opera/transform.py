"""
This module is responsible for all operator-related token & AST transformations.
"""

from __future__ import annotations

import ast
import token
from tokenize import TokenInfo
from typing import Sequence

from .tokens import insert_inplace, remove_inplace


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


def transform_operator_objects_inplace(toks: list[TokenInfo], nonce: str) -> None:
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
