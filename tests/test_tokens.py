import tokenize
import pytest
import token
from random import randint
from tokenize import TokenInfo

from hoopy import tokens


def test_offset():
    for i in range(1000):
        tok = TokenInfo(0, "", (1, 0), (1, 0), "")
        expected = TokenInfo(0, "", (1, i), (1, i), "")
        assert tokens.offset(tok, i) == expected


def test_offset_line_inplace():
    for _ in range(100):
        toks = list[TokenInfo]()
        for line in range(10):
            for i in range(10):
                tok = TokenInfo(0, "", (line + 1, i), (line + 1, i + 1), "")
                toks.append(tok)

        index = randint(1, 98)

        first = toks[0]
        prev = toks[index - 1]
        current = toks[index]
        next = toks[index + 1]
        last = toks[-1]

        tokens.offset_line_inplace(toks, line=index // 10 + 1, by=10, starting=index)

        assert first.start[1] == toks[0].start[1]
        assert first.end[1], toks[0].end[1]

        assert prev.start[1] == toks[index - 1].start[1]
        assert prev.end[1] == toks[index - 1].end[1]

        assert current.start[1] + 10 == toks[index].start[1]
        assert current.end[1] + 10 == toks[index].end[1]

        if next.start[0] == current.start[0]:
            assert next.start[1] + 10 == toks[index + 1].start[1]
            assert next.end[1] + 10 == toks[index + 1].end[1]
        else:
            assert next.start[1] == toks[index + 1].start[1]
            assert next.end[1] == toks[index + 1].end[1]

        if last.start[0] == current.start[0]:
            assert last.start[1] + 10 == toks[-1].start[1]
            assert last.end[1] + 10 == toks[-1].end[1]
        else:
            assert last.start[1] == toks[-1].start[1]
            assert last.end[1] == toks[-1].end[1]


def test_offset_line_newlines():
    toks = list(tokens.lex("\n++\n"))
    expected = list(tokens.lex("\n+  +\n"))
    tokens.offset_line_inplace(toks, line=2, by=2, starting=2)
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_simple():
    toks = list(tokens.lex("1+"))
    expected = list(tokens.lex("1:+"))
    tokens.insert_inplace(toks, 1, token.OP, ":", left_offset=0, right_offset=0)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_offset():
    toks = list(tokens.lex("++"))
    expected = list(tokens.lex("+ : +"))
    tokens.insert_inplace(toks, 1, token.OP, ":", left_offset=1, right_offset=1)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_newline():
    toks = list(tokens.lex("1\n+"))
    expected = list(tokens.lex("1\n:+"))
    tokens.insert_inplace(toks, 2, token.OP, ":", next_row=True)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_lex_unlex():
    src = "1 + 2 == 5; 4178937\nfhjsd[]!=(%)#ÅÖlm14kl3ori\n\n\n\taaaaa"
    assert tokens.unlex(tokens.lex(src)) == src


def test_fix_spans():
    toks = list(tokens.lex("123+"))
    tokens.offset_line_inplace(toks, line=1, by=-3, starting=1)

    with pytest.raises(ValueError):
        tokenize.untokenize(toks)

    assert tokenize.untokenize(tokens.fix_spans(toks)) == "123+"


def test_token_eq():
    assert tokens.token_eq(
        TokenInfo(0, "", (0, 0), (0, 0), ""),
        TokenInfo(0, "", (0, 0), (0, 0), ""),
    )
    assert tokens.token_eq(
        TokenInfo(0, "", (0, 0), (0, 0), "different"),
        TokenInfo(0, "", (0, 0), (0, 0), ""),
    )
    assert not (
        tokens.token_eq(
            TokenInfo(0, "different", (0, 0), (0, 0), "different"),
            TokenInfo(0, "", (0, 0), (0, 0), ""),
        )
    )
    assert not (
        tokens.token_eq(
            TokenInfo(0, "different", (0, 0), (0, 0), ""),
            TokenInfo(0, "", (0, 0), (0, 0), ""),
        )
    )
