import token
import tokenize
from random import randint
from tokenize import TokenInfo

from opera import tokens


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
    toks = list(tokens.tokenize("\n++\n"))
    expected = list(tokens.tokenize("\n+  +\n"))
    tokens.offset_line_inplace(toks, line=2, by=2, starting=2)
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_simple():
    toks = list(tokens.tokenize("1+"))
    expected = list(tokens.tokenize("1:+"))
    tokens.insert_inplace(toks, 1, token.OP, ":", 0, False)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_offset():
    toks = list(tokens.tokenize("1+"))
    expected = list(tokens.tokenize("1+ :"))
    tokens.insert_inplace(toks, 2, token.OP, ":", 1, False)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_insert_inplace_newline():
    toks = list(tokens.tokenize("1\n+"))
    expected = list(tokens.tokenize("1\n:+"))
    tokens.insert_inplace(toks, 2, token.OP, ":", 0, True)
    assert len(toks) == len(expected), "different count"
    for l, r in zip(toks, expected):
        assert tokens.token_eq(l, r), f"{l} and {r} not equal"


def test_tokenize():
    src = "1 + 2 == 5; 4178937\nfhjsd[]!=(%)#ÅÖlm14kl3ori\n\n\n\taaaaa"
    assert tokenize.untokenize(tokens.tokenize(src)), src


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


def test_transform_operator_objects_inplace():
    src = "(a) (...) (not in) ($$) (()) (+) (2+) (~~) (+ -) (is*)"
    toks = list(tokens.tokenize(src))
    tokens.transform_operator_objects_inplace(toks, "nonce")
    out = tokenize.untokenize(toks)
    expected = "(a) (...) (__operator_nonce_6e6f7420696e) (__operator_nonce_2424) (()) (__operator_nonce_2b) (2+) (__operator_nonce_7e7e) (+ -) (is*)"
    assert out == expected
