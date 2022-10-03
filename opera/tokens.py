"""
Token transformations and utility functions!
"""

from __future__ import annotations

import token
from io import StringIO
from tokenize import TokenInfo, generate_tokens
from typing import Iterator, Sequence


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

    The optional `starting` parameter (default 0) determines the first token index to be considered.
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


def remove_inplace(toks: list[TokenInfo], start: int, end: int | None = None) -> None:
    """Remove some slice within this the token stream, and shift following tokens backward appropriately.

    Endpoints are handled the same as with python slices.

    If `end` is missing or None, it defaults to `start + 1`."""
    if end is None:
        end = start + 1

    if end <= start:
        return

    s_row, s_col = toks[start].start
    e_row, e_col = toks[end - 1].end

    if s_row == e_row:
        # this is negative (i.e. leftwards)
        offset = s_col - e_col
        toks[start:end] = []
        offset_line_inplace(toks, line=s_row, by=offset, starting=start)
    else:
        raise ValueError(
            "Removing tokens spanning across multiple lines not supported at the moment"
        )


def insert_inplace(
    toks: list[TokenInfo],
    index: int,
    type: int,
    string: str,
    *,
    left_offset: int = 0,
    right_offset: int = 0,
    next_row: bool = False,
) -> None:
    """Insert a token into a token stream, and shift all following tokens forward appropriately.

    * index: index into token list
    * type: token type (see `tokens` in the standard library)
    * string: token string
    * left_offset: offset from previous token
    * right_offset: offset to next token
    * next_row: whether to skip forward to the next row
    """
    previous = toks[index - 1]
    row, col = previous.end
    row = row + 1 if next_row else row
    col = left_offset if next_row else col + left_offset
    toks.insert(
        index,
        # the line attribute is left blank here since it is not used anywhere
        TokenInfo(type, string, (row, col), (row, col + len(string)), ""),
    )
    offset_line_inplace(
        toks, line=row, by=left_offset + len(string) + right_offset, starting=index + 1
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
