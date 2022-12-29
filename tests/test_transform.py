import tokenize
import importlib
import inspect
import pytest
from hoopy import tokens, transform, utils


class TestMangleOperatorObjectsInplace:
    def expect_transformation(self, src, exp):
        toks = list(tokens.lex(src))
        transform.mangle_operator_objects_inplace(toks, "nonce")
        utils.print_token(toks)
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

    def test_simple_application(self):
        self.expect_transformation(
            "a b", "a * b", {transform.Spans((1, 2), (1, 3)): transform.Application()}
        )

    def test_strange_backticks(self):
        self.expect_transformation(
            "a cd  `efg` `123` 1`b`c`2 `  `",
            "a * cd  * `123` 1*c`2 `  `",
            {
                transform.Spans((1, 2), (1, 3)): transform.Application(),
                transform.Spans((1, 8), (1, 9)): transform.Identifier("efg"),
                transform.Spans((1, 17), (1, 18)): transform.Identifier("b"),
            },
        )

    def test_function_application_example(self):
        self.expect_transformation(
            "(fs[0]) True x {1, 2, 3} None 'hi'",
            "(fs[0]) * True * x * {1, 2, 3} * None * 'hi'",
            {
                transform.Spans((1, 8), (1, 9)): transform.Application(),
                transform.Spans((1, 15), (1, 16)): transform.Application(),
                transform.Spans((1, 19), (1, 20)): transform.Application(),
                transform.Spans((1, 31), (1, 32)): transform.Application(),
                transform.Spans((1, 38), (1, 39)): transform.Application(),
            },
        )

    def test_custom_operator_example(self):
        self.expect_transformation(
            "f <$> x <*> y <*> z ?? d",
            "f << x << y << z  and  d",
            {
                transform.Spans((1, 2), (1, 4)): transform.Custom("<$>"),
                transform.Spans((1, 7), (1, 9)): transform.Custom("<*>"),
                transform.Spans((1, 12), (1, 14)): transform.Custom("<*>"),
                transform.Spans((1, 18), (1, 21)): transform.Custom("??"),
            },
        )

    def test_monad(self):
        self.expect_transformation(
            "f >>= pure x",
            "f << pure * x",
            {
                transform.Spans(start=(1, 2), end=(1, 4)): transform.Inplace(op=">>="),
                transform.Spans(start=(1, 10), end=(1, 11)): transform.Application(),
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
            ("x None", "x * None", 2),  # left end, right end
            ("x True", "x * True", 2),
            ("True False", "True * False", 5),
            ("NotImplemented x", "NotImplemented * x", 15),
        )
        for a, b, c in allowed:
            self.expect_transformation(
                a, b, {transform.Spans((1, c), (1, c + 1)): transform.Application()}
            )

    # TODO: more infix token tests


class TestStandardLibraryBreakage:
    def expect_transformation(self, src, exp, spans):
        toks = list(tokens.lex(src))
        _, exp_spans = transform.collect_operator_tokens_inplace(toks)
        out = tokens.unlex(toks)
        assert out == exp
        assert spans == exp_spans

    def expect_no_operators_caught(self, src: str):
        toks = list(tokens.lex(src))
        _, spans = transform.collect_operator_tokens_inplace(toks)
        # The output tokens will not be identical, but they should have the same semantic meaning
        # print(*spans.items(), sep="\n")
        assert all(isinstance(v, transform.Inplace) for v in spans.values())

    def test_important_modules(self):
        modules = [
            "abc",
            "argparse",
            "ast",
            "asyncio",
            "codecs",
            "collections",
            "cProfile",
            "ctypes",
            "dataclasses",
            "decimal",
            "distutils",
            "encodings",
            "functools",
            "gzip",
            "hashlib",
            "html",
            "http",
            "importlib",
            "inspect",
            "io",
            "json",
            "keyword",
            "logging",
            "opcode",
            "os",
            "pathlib",
            "pickle",
            "pprint",
            "random",
            "secrets",
            "socket",
            "sqlite3",
            "subprocess",
            "threading",
            "token",
            "types",
            "typing",
            "venv",
            "weakref",
            "webbrowser",
            "zipfile",
        ]
        for module in (importlib.import_module(module) for module in modules):
            try:
                src = inspect.getsource(module)
            except TypeError:
                # The module is implemented in C and therefore doesn't have a source
                pass
            else:
                self.expect_no_operators_caught(src)

    # these test cases were caught thanks to above :)
    def test_slice_transformation(self):
        self.expect_transformation("x[:-1]", "x[:-1]", {})
        self.expect_transformation("x[:~1]", "x[:~1]", {})
        self.expect_transformation(
            "x ::: 1",
            "x + 1",
            {transform.Spans((1, 2), (1, 3)): transform.Custom(":::")},
        )

    def test_string_concatenation(self):
        self.expect_transformation("'a' 'b'", "'a' 'b'", {})
        self.expect_transformation("('a'\\\n 'b')", "('a'\\\n 'b')", {})

    def test_ellipsis_blocks(self):
        self.expect_transformation("if x is ...: y", "if x is ...: y", {})
