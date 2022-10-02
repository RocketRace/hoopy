"""
Token transformations!
"""

from __future__ import annotations

import token
from io import StringIO
from tokenize import TokenInfo, generate_tokens
from typing import Iterator


def offset(tok: TokenInfo, by: int) -> TokenInfo:
    """Apply a horizontal offset to a token."""
    s_row, s_col = tok.start
    e_row, e_col = tok.end
    return TokenInfo(
        tok.type, tok.string, (s_row, s_col + by), (e_row, e_col + by), tok.line
    )


def offset_line_inplace(
    tokens: list[TokenInfo], *, line: int, by: int, starting: int = 0
) -> None:
    """Horizontally shift the spans of tokens in the given line right by `by` columns, mutating the list in-place.

    The optional `start` parameter (default 0) determines the first token index to be considered.
    """
    start = -1
    end = -1
    for i, tok in enumerate(tokens[starting:], starting):
        if tok.start[0] == line:
            if start == -1:
                start = i
            end = i
        elif start != -1:
            break
    if start == end == -1:
        return
    tokens[start : end + 1] = [offset(token, by) for token in tokens[start : end + 1]]


def token_eq(left: TokenInfo, right: TokenInfo) -> bool:
    """Convenience function to compare tokens by their relevant fields:

    * Type
    * String
    * Start
    * End
    """
    return (
        left.type == right.type
        and left.string == right.string
        and left.start == right.start
        and left.end == right.end
    )


def tokenize(source: str) -> Iterator[TokenInfo]:
    """Convenience function to iterate over tokens of source code"""
    return generate_tokens(StringIO(source).readline)


def insert_inplace(
    toks: list[TokenInfo],
    index: int,
    type: int,
    string: str,
    offset: int,
    next_row: bool = False,
) -> None:
    """Insert a token into a token stream, and shift all following tokens forward appropriately.

    * index: index into token list
    * type: token type (see `tokens` in the standard library)
    * string: token string
    * offset: offset from previous token
    * next_row: whether to skip forward to the next row
    """
    previous = toks[index - 1]
    row, col = previous.end
    row = row + 1 if next_row else row
    col = offset if next_row else col + offset
    toks.insert(
        index,
        # the line attribute is left blank here since it is not used anywhere
        TokenInfo(type, string, (row, col), (row, col + len(string)), ""),
    )
    offset_line_inplace(toks, line=row, by=offset + len(string), starting=index + 1)


class UnclosedParentheses(Exception):
    """No matching parenthesis found.

    depth: Number of unclosed parentheses (>= 1)
    """

    def __init__(self, depth: int):
        self.depth = depth


def find_matching_parentheses(tokens: list[TokenInfo], left: int) -> int:
    """Returns the the index of right-paren paired with left-paren at the given index, or None if not found."""
    depth = 1
    for i, tok in enumerate(tokens[left + 1 :]):
        if tok.type == token.RPAR:
            depth -= 1
        elif tok.type == token.LPAR:
            depth += 1
        if depth == 0:
            return i
    raise UnclosedParentheses(depth)


def matches_objectification(tokens: list[TokenInfo]) -> bool:
    """Tests against objectified patterns"""
    return False


def objectify_operators_inplace(tokens: list[TokenInfo]) -> None:
    """Substitutes all instances of objectified operators (e.g. `(+)`)
    with their desugared representations (e.g. `(__operator__(__name__, "+"))`).

    Returns True if any substitutions were made, and False otherwise.
    """
    i = -1
    while i < len(tokens):
        i += 1
        tok = tokens[i]
        if tok == token.LPAR:
            pass
