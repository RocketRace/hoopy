import tokenize
import pytest
from hoopy import tokens, transform


class TestMangleOperatorObjectsInplace:
    def expect_transformation(self, src, exp):
        toks = list(tokens.lex(src))
        transform.mangle_operator_objects_inplace(toks, "nonce")
        tokens.pretty_print(toks)
        out = tokens.unlex(toks)
        assert out == exp

    def test_non_operators(self):
        self.expect_transformation("(a)", "(a)")
        self.expect_transformation("(1)", "(1)")
        self.expect_transformation("(match)", "(match)")
        self.expect_transformation("(+1)", "(+1)")

    def test_builtin_operators(self):
        self.expect_transformation("(is)", " __operator_nonce_6973 ")
        self.expect_transformation("(+)", " __operator_nonce_2b ")
        self.expect_transformation("(not in)", " __operator_nonce_6e6f7420696e ")
        self.expect_transformation("(>>=)", " __operator_nonce_3e3e3d ")

    def test_custom_operators(self):
        self.expect_transformation("($)", " __operator_nonce_24 ")
        self.expect_transformation("(:::)", " __operator_nonce_3a3a3a ")
        self.expect_transformation("(?=>)", " __operator_nonce_3f3d3e ")

    def test_invalid_syntax(self):
        with pytest.raises(tokenize.TokenError):
            self.expect_transformation("(", "(")
        with pytest.raises(tokenize.TokenError):
            self.expect_transformation("((((((", "((((((")
        with pytest.raises(tokenize.TokenError):
            self.expect_transformation("(((((()", "(((((()")
        self.expect_transformation(")(", ")(")
        self.expect_transformation("(*is)", "(*is)")
        self.expect_transformation("(is and)", "(is and)")
        self.expect_transformation("(not   not)", "(not   not)")

    def test_whitespace(self):
        self.expect_transformation("(  $)", " __operator_nonce_24 ")
        self.expect_transformation("( $ )", " __operator_nonce_24 ")
        self.expect_transformation("($ $)", "($ $)")
        self.expect_transformation(" ($) ", "  __operator_nonce_24  ")
        self.expect_transformation("(not  in)", " __operator_nonce_6e6f7420696e ")
        self.expect_transformation("(not       in)", " __operator_nonce_6e6f7420696e ")

    def test_invalid_operator(self):
        self.expect_transformation("(...)", "(...)")
        self.expect_transformation("(=)", "(=)")
        self.expect_transformation("(.)", "(.)")
        self.expect_transformation("(:=)", "(:=)")
        self.expect_transformation("(::)", "(::)")

    def test_combination(self):
        src = "(a)(...)(not in)($$)(())(+)(2+)(~~)(+ -)(is*)"
        exp = "(a)(...) __operator_nonce_6e6f7420696e  __operator_nonce_2424 (()) __operator_nonce_2b (2+) __operator_nonce_7e7e (+ -)(is*)"
        self.expect_transformation(src, exp)


class TestCollectOperatorTokensInplace:
    def expect_transformation(self, src, exp, spans):
        toks = list(tokens.lex(src))
        _, exp_spans = transform.collect_operator_tokens_inplace(toks)
        out = tokens.unlex(toks)
        assert out == exp
        assert spans == exp_spans

    def test_strange_backticks(self):
        self.expect_transformation(
            "a cd  `efg` `123` 1`b`c`2 `  `",
            "a * cd  * `123` 1*c`2 `  `",
            {
                transform.Spans((1, 1), (1, 4)): transform.Application(),
                transform.Spans((1, 6), (1, 10)): transform.Identifier("efg"),
                transform.Spans((1, 17), (1, 18)): transform.Identifier("b"),
            },
        )

    def test_function_application_example(self):
        self.expect_transformation(
            "(fs[0]) True x {1, 2, 3} None 'hi'",
            "(fs[0]) * True * x * {1, 2, 3} * None * 'hi'",
            {
                transform.Spans((1, 7), (1, 10)): transform.Application(),
                transform.Spans((1, 14), (1, 17)): transform.Application(),
                transform.Spans((1, 18), (1, 21)): transform.Application(),
                transform.Spans((1, 30), (1, 33)): transform.Application(),
                transform.Spans((1, 37), (1, 40)): transform.Application(),
            },
        )

    def test_custom_operator_example(self):
        self.expect_transformation(
            "f <$> x <*> y <*> z ?? d",
            "f << x << y << z  and  d",
            {
                transform.Spans((1, 1), (1, 5)): transform.Custom("<$>"),
                transform.Spans((1, 6), (1, 10)): transform.Custom("<*>"),
                transform.Spans((1, 11), (1, 15)): transform.Custom("<*>"),
                transform.Spans((1, 16), (1, 23)): transform.Custom("??"),
            },
        )

    def test_invalid_keyword_application(self):
        # not exhaustive
        disallowed = (
            "x in",
            "in x",
            "import x",
            "is not",
            "not x",
            "async for",
            "for x",
            "await x",
            "lambda x",
            "def x",
            "async def",
            "not not" "not in",
            "not True",
            "None if",
            "return NotImplemented",
        )
        for bad in disallowed:
            self.expect_transformation(bad, bad, {})

    def test_valid_keyword_application(self):
        allowed = (
            ("x None", "x * None", 1),  # left end, right end
            ("x True", "x * True", 1),
            ("True False", "True * False", 4),
            ("NotImplemented x", "NotImplemented * x", 14),
        )
        for a, b, c in allowed:
            self.expect_transformation(
                a, b, {transform.Spans((1, c), (1, c + 3)): transform.Application()}
            )

    # TODO: more infix token tests
