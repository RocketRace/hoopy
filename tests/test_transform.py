import ast
import tokenize
import importlib
import inspect
from typing import Mapping
import pytest
from hoopy import tokens, transform, utils


class TestMangleOperatorObjectsInplace:
    def expect_transformation(self, src: str, exp: str):
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
    def expect_transformation(
        self, src: str, exp: str, spans: Mapping[utils.Span, transform.OperatorBase]
    ):
        toks = list(tokens.lex(src))
        _, exp_spans = transform.collect_operator_tokens_inplace(toks)
        out = tokens.unlex(toks)
        assert out == exp
        assert spans == exp_spans

    def test_simple_application(self):
        self.expect_transformation(
            "a b", "a * b", {transform.Span((1, 1), (1, 4)): transform.Application()}
        )

    def test_strange_backticks(self):
        self.expect_transformation(
            "a cd  `efg` `123` 1`b`c`2 `  `",
            "a * cd  * `123` 1*c`2 `  `",
            {
                transform.Span((1, 1), (1, 4)): transform.Application(),
                transform.Span((1, 6), (1, 10)): transform.Identifier("efg"),
                transform.Span((1, 17), (1, 18)): transform.Identifier("b"),
            },
        )

    def test_function_application_example(self):
        self.expect_transformation(
            "(fs[0]) True x {1, 2, 3} None 'hi'",
            "(fs[0]) * True * x * {1, 2, 3} * None * 'hi'",
            {
                transform.Span((1, 7), (1, 10)): transform.Application(),
                transform.Span((1, 14), (1, 17)): transform.Application(),
                transform.Span((1, 18), (1, 21)): transform.Application(),
                transform.Span((1, 30), (1, 33)): transform.Application(),
                transform.Span((1, 37), (1, 40)): transform.Application(),
            },
        )

    def test_custom_operator_example(self):
        self.expect_transformation(
            "f <$> x <*> y <*> z ?? d",
            "f << x << y << z  and  d",
            {
                transform.Span((1, 1), (1, 5)): transform.Custom("<$>"),
                transform.Span((1, 6), (1, 10)): transform.Custom("<*>"),
                transform.Span((1, 11), (1, 15)): transform.Custom("<*>"),
                transform.Span((1, 16), (1, 23)): transform.Custom("??"),
            },
        )

    def test_monad(self):
        self.expect_transformation(
            "f >>= pure x",
            "f == pure * x",
            {
                transform.Span(start=(1, 1), end=(1, 5)): transform.Inplace(op=">>="),
                transform.Span(start=(1, 9), end=(1, 12)): transform.Application(),
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
                a, b, {transform.Span((1, c), (1, c + 3)): transform.Application()}
            )


class TestStandardLibraryBreakage:
    def expect_no_transformation(self, src: str):
        return self.expect_transformation(src, src, {})

    def expect_transformation(
        self, src: str, exp: str, spans: Mapping[utils.Span, transform.OperatorBase]
    ):
        toks = list(tokens.lex(src))
        _, exp_spans = transform.collect_operator_tokens_inplace(toks)
        out = tokens.unlex(toks)
        assert out == exp
        assert spans == exp_spans

    def expect_only_inplace_ops(self, src: str):
        toks = list(tokens.lex(src))
        _, spans = transform.collect_operator_tokens_inplace(toks)
        # The output tokens may not be identical, but they should have the same semantic meaning
        # print(*spans.items(), sep="\n")
        assert all(isinstance(v, transform.Inplace) for v in spans.values())

    def expect_round_trip_equivalance(self, src: str):
        # The header is always present in transformed code, ignore it
        out = transform.transform(src).replace("from hoopy.magic import *", "")
        assert ast.unparse(ast.parse(out)) == ast.unparse(ast.parse(src))

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
                self.expect_only_inplace_ops(src)
                self.expect_round_trip_equivalance(src)

    # these test cases were caught thanks to above :)
    def test_slice_transformation(self):
        self.expect_no_transformation("x[:-1]")
        self.expect_no_transformation("x[:~1]")
        self.expect_transformation(
            "x ::: 1",
            "x + 1",
            {transform.Span((1, 1), (1, 4)): transform.Custom(":::")},
        )

    def test_string_concatenation(self):
        self.expect_no_transformation("'a' 'b'")
        self.expect_no_transformation("('a'\\\n 'b')")

    def test_ellipsis_blocks(self):
        self.expect_no_transformation("if x is ...: y")

    def test_short_circuit(self):
        # short circuiting chains should maintain their original semantics
        self.expect_no_transformation("x and y and z")

    def test_nested_short_circuit(self):
        # ensure boolops maintain span information
        self.expect_no_transformation("(x and y) and (z and w)")


class TestFullTransformation:
    def test_simple_program(self):
        source = "a = f <$> x <*> y ?? z"
        expected = "from hoopy.magic import *\na = __operator__(__name__, '??')(__operator__(__name__, '<*>')(__operator__(__name__, '<$>')(f, x), y), z)"
        assert transform.transform(source) == expected

    def test_parenthesized_spans(self):
        source = "(f) x"
        expected = "from hoopy.magic import *\n__partial_apply__(f, x)"
        assert transform.transform(source) == expected

    def test_inline_inplace(self):
        source = "a = x += y"
        expected = "from hoopy.magic import *\na = __operator__(__name__, '+=')(x, y)"
        assert transform.transform(source) == expected

    def test_normal_inplace(self):
        source = "x += 2"
        expected = "from hoopy.magic import *\nx += 2"
        assert transform.transform(source) == expected

    def test_nested_normal_inplace(self):
        source = "if x:\n    x += 1"
        expected = "from hoopy.magic import *\nif x:\n    x += 1"
        assert transform.transform(source) == expected

    def test_plain(self):
        # Don't wrap builtin operators with an __operator__ call
        source = "1 + 2 + 3"
        expected = "from hoopy.magic import *\n1 + 2 + 3"
        assert transform.transform(source) == expected

    def test_operator_imports(self):
        source = "from x import (+)"
        expected = (
            "from hoopy.magic import *\n__import_operator__(__name__, 'x', 0, '+')"
        )
        assert transform.transform(source) == expected

    def test_inner_opeartor_imports(self):
        source = "from x import a, (+), b"
        expected = "from hoopy.magic import *\nfrom x import a\n__import_operator__(__name__, 'x', 0, '+')\nfrom x import b"
        assert transform.transform(source) == expected

    def test_magic_import_docstring(self):
        source = '"""docstring"""'
        expected = '"""docstring"""\nfrom hoopy.magic import *'
        assert transform.transform(source) == expected

    def test_magic_import_future(self):
        source = "from __future__ import annotations"
        expected = "from __future__ import annotations\nfrom hoopy.magic import *"
        assert transform.transform(source) == expected

    def test_magic_import_future_docstring(self):
        source = '"""docstring"""\nfrom __future__ import annotations'
        expected = '"""docstring"""\nfrom __future__ import annotations\nfrom hoopy.magic import *'
        assert transform.transform(source) == expected

    def test_bad_magic_import_future_docstring(self):
        source = 'from __future__ import annotations\n"""not docstring"""'
        expected = "from __future__ import annotations\nfrom hoopy.magic import *\n'not docstring'"
        assert transform.transform(source) == expected

    def test_operator_definition(self):
        source = "def (??)(a, b): pass"
        expected = "from hoopy.magic import *\n\n@__define_operator__('??', False)\ndef __operator_nonce_3f3f(a, b):\n    pass"
        assert transform.transform(source, "nonce") == expected

    def test_operator_definition_flipped(self):
        source = "def (??)(a, self): pass"
        expected = "from hoopy.magic import *\n\n@__define_operator__('??', True)\ndef __operator_nonce_3f3f(a, self):\n    pass"
        assert transform.transform(source, "nonce") == expected

    def test_class_operator_definition(self):
        source = "class C:\n    def (/%)(self, other): pass"
        expected = "from hoopy.magic import *\n\nclass C:\n\n    @__define_operator__('/%', False)\n    def __operator_nonce_2f25(self, other):\n        pass"
        assert transform.transform(source, "nonce") == expected

    def test_class_operator_definition_flipped(self):
        source = "class C:\n    def (/%)(other, self): pass"
        expected = "from hoopy.magic import *\n\nclass C:\n\n    @__define_operator__('/%', True)\n    def __operator_nonce_2f25(other, self):\n        pass"
        assert transform.transform(source, "nonce") == expected
