"""
Utility functions for dealing with token streams.

This is used primarily by the `transform` module.
"""

from __future__ import annotations
import itertools

import tokenize
from io import StringIO
from tokenize import TokenInfo
from typing import Iterable, Iterator, Sequence


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


def lex(source: str) -> Iterator[TokenInfo]:
    """Convenience function to iterate over tokens of source code"""
    return tokenize.generate_tokens(StringIO(source).readline)


def unlex(toks: Iterable[TokenInfo]) -> str:
    """Convenience function to reconstruct a string stream from tokens.

    TODO: Also fix "broken" spans, i.e. whenever a token's start
    span precedes the previous token's end span.
    """
    return tokenize.untokenize(fix_spans(list(toks)))


def remove_inplace(
    toks: list[TokenInfo],
    start: int,
    end: int | None = None,
    *,
    shift_rest: bool = True,
) -> None:
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
        if shift_rest:
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


def fix_spans(toks: Sequence[TokenInfo]) -> Sequence[TokenInfo]:
    """Replaces broken token spans in a token stream, making
    the resulting tokens valid to pass into tokenize.untokenize()"""
    if len(toks) <= 1:
        return toks

    out = [x for x in toks]
    for i in range(len(toks) - 1):
        prev = out[i]
        current = out[i + 1]

        p_row, p_col = prev.end
        c_row, c_col = current.start
        if p_row < c_row:
            continue
        elif p_row > c_row:
            raise ValueError("Fixing vertical offsets not yet supported")
        else:
            if p_col <= c_col:
                continue
            else:
                delta = p_col - c_col
                out[i + 1] = offset(current, delta)
    return out
