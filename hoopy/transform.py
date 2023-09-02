"""
This module is responsible for all operator-related token & AST transformations.
"""

from __future__ import annotations
import abc

import ast
import copy
import itertools
import keyword
import random
import token
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import tokens
from .runtime import (
    is_disallowed_operator,
    OperatorKind,
    is_builtin_with_kind,
    operator_proxy_for,
)
from .utils import T, U, Pipe, Span, SpanTree, dbg


def generate_import(orig: ast.ImportFrom, op: str, verbatim: bool = False) -> ast.stmt:
    """Generates a *statement* of the form
    ```
    __import_operator__(__name__, module, level, op)
    ```
    given values of `module`, `level`, and `op`.
    """
    return ast.Expr(
        value=ast.Call(
            func=ast.Name("__import_operator__", ctx=ast.Load()),
            args=[
                ast.Name("__name__", ctx=ast.Load()),
                ast.Constant(orig.module),
                ast.Constant(orig.level),
                ast.Constant(op),
            ],
            keywords=[],
        )
    )


def generate_operator_object(op: str, verbatim: bool = False) -> ast.expr:
    """Generate an *expression* of the form
    ```
    __operator__(__name__, op, verbatim=verbatim)
    ```
    """
    return ast.Call(
        func=ast.Name("__operator__", ctx=ast.Load()),
        args=[ast.Name("__name__", ctx=ast.Load()), ast.Constant(op)],
        keywords=[ast.keyword("verbatim", ast.Constant(verbatim))],
    )


def generate_operator_definition(
    op: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    asynchronous: bool,
    verbatim: bool = False,
) -> ast.stmt:
    """Generates a *statement* of the form
    ```
    @__define_operator__(op, flipped=<flipped>, verbatim=<verbatim>)
    <async?> def <...>
    ```
    where `flipped` is automatically inferred from the position of `self` in the args of `func`,
    as well as the context in which the function was defined.
    """
    args = func.args
    try:
        self_position = next(i for i, arg in enumerate(args.args) if arg.arg == "self")
        # (other, self) -> ...
        flipped = self_position == 1
    except StopIteration:
        # if no explicit self is given, make no assumptions
        flipped = False

    new_func = copy.deepcopy(func)
    new_func.decorator_list.insert(
        0,
        ast.Call(
            func=ast.Name("__define_operator__", ctx=ast.Load()),
            args=[ast.Constant(op)],
            keywords=[
                ast.keyword("flipped", ast.Constant(flipped)),
                ast.keyword("verbatim", ast.Constant(verbatim)),
            ],
        ),
    )
    return new_func


def generate_operator_class(
    op: str, cls: ast.ClassDef, verbatim: bool = False
) -> ast.stmt:
    """Generates a *statement* of the form
    ```
    @__define_operator__(op, flipped=False, verbatim=<verbatim>))
    class <...>
    ```
    which effectively makes the class callable as an operator,
    akin to cons list syntax.
    """
    new_cls = copy.deepcopy(cls)
    new_cls.decorator_list.insert(
        0,
        ast.Call(
            func=ast.Name("__define_operator__", ctx=ast.Load()),
            args=[ast.Constant(op)],
            keywords=[
                ast.keyword("flipped", ast.Constant(False)),
                ast.keyword("verbatim", ast.Constant(verbatim)),
            ],
        ),
    )
    return new_cls


MAGIC_IMPORTS = ast.ImportFrom(
    module="hoopy.magic", names=[ast.alias(name="*")], level=0
)


def mangle_operator_string(toks: Sequence[TokenInfo], nonce: str) -> str | None:
    """Returns the mangled representation of a token, or None if the tokens
    don't constitute a valid operator.

    Suffixing an operator with a backtick `` ` `` will cause some invalid tokens
    to become valid and treated as "verbatim" operators. In that case, the mangled
    representation includes this bit of metadata.

    Reasons for failure include:

    * Disallowed token types (`$2$` is invalid, `$.$` is valid)
    * Spaces between non-keyword tokens (`+ +` is invalid, `is not` is valid)
    * Specifically blacklisted tokens (`:=` is invalid, `:=:` is valid)
    """
    if not toks:
        return None

    is_verbatim = False

    if 1 <= len(toks) <= 2 and all(
        tokens.is_keyword_operator_token(tok) for tok in toks
    ):
        op_string = " ".join(tok.string for tok in toks)
        if not is_builtin_with_kind(op_string, OperatorKind.Keyword):
            return None

    else:
        prev = None
        operator: list[str] = []
        is_verbatim = toks[-1].string == "`"
        for tok in toks[:-1] if is_verbatim else toks:
            if not tokens.is_custom_operator_token(tok) or (
                prev is not None and not tokens.is_adjacent(prev, tok)
            ):
                return None
            operator.append(tok.string)
            prev = tok
        if is_verbatim:
            if prev is not None and not tokens.is_adjacent(prev, toks[-1]):
                return None

        op_string = "".join(operator)
        if not is_verbatim and is_disallowed_operator(op_string):
            return None

    suffix = f"_{nonce}_{''.join(f'{ord(c):x}' for c in op_string)}"
    if is_verbatim:
        return f"__operator_verbatim{suffix}"
    else:
        return f"__operator{suffix}"


def demangle_operator_string(s: str, nonce: str) -> tuple[str, bool] | None:
    """Returns the operator string which generated `s`, or None if invalid.

    Note: This does no validation on hex pairs, so may yield characters outside
    the printable ASCII range, with e.g. `operator_{nonce}_ffff` as input.
    This is considered an implementation detail and is subject to change in the future.
    """
    if s.startswith(f"__operator_{nonce}_"):
        hex = s.removeprefix(f"__operator_{nonce}_")
        is_verbatim = False
    else:
        if s.startswith(f"__operator_verbatim_{nonce}_"):
            hex = s.removeprefix(f"__operator_verbatim_{nonce}_")
            is_verbatim = True
        else:
            return

    try:
        return (
            "".join(
                # pairwise hex digits
                chr(int(hex[i : i + 2], base=16))
                for i in range(0, len(hex), 2)
            ),
            is_verbatim,
        )
    except ValueError:  # invalid hex
        return


def mangle_operator_objects_inplace(
    toks: list[TokenInfo], nonce: str
) -> list[TokenInfo]:
    """Substitutes all instances of objectified operators (e.g. `(+%+)`)
    with their hex representations, such as ` __operator_0f198f1a_2b252b `
    given the nonce `"0f198f1a"`. Notice the extra spaces around the identifiers.
    This is kept to ensure no tokens are accidentally joined together after this
    transformation.
    """
    tokens.remove_error_whitespace_inplace(toks)
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
        tokens.remove_inplace(toks, start - 1, end + 1)
        tokens.insert_inplace(
            toks, start - 1, token.NAME, mangled, left_offset=1, right_offset=1
        )

    return toks


class OperatorBase(abc.ABC):
    @abc.abstractmethod
    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an expression from these two buddies"""

    def generate_and_copy_spans(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates and finalizes an operation from the two arguments"""
        output = self.generate(left, right)
        output.lineno = left.lineno
        output.col_offset = left.col_offset
        output.end_lineno = right.end_lineno
        output.end_col_offset = right.end_col_offset
        return output


@dataclass
class Application(OperatorBase):
    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an *expression* of the form
        ```
        __partial_apply__(left, right)
        ```
        given values of `left` and `right`
        """
        return ast.Call(
            func=ast.Name("__partial_apply__"),
            args=[left, right],
            keywords=[],
            lineno=left.lineno,
            col_offset=left.col_offset,
            end_lineno=right.end_lineno,
            end_col_offset=right.end_col_offset,
        )


@dataclass
class Identifier(OperatorBase):
    name: str

    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an *expression* of the form
        ```
        name(left, right)
        ```
        given values of `name`, `left` and `right`.
        """
        return ast.Call(
            func=ast.Name(self.name, ctx=ast.Load()),
            args=[left, right],
            keywords=[],
            lineno=left.lineno,
            col_offset=left.col_offset,
            end_lineno=right.end_lineno,
            end_col_offset=right.end_col_offset,
        )


@dataclass
class Custom(OperatorBase):
    op: str

    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an *expression* of the form
        ```
        __operator__(__name__, op)(left, right)
        ```
        given values of `op`, `left` and `right`.
        """
        return ast.Call(
            func=generate_operator_object(self.op),
            args=[left, right],
            keywords=[],
            lineno=left.lineno,
            col_offset=left.col_offset,
            end_lineno=right.end_lineno,
            end_col_offset=right.end_col_offset,
        )


@dataclass
class Verbatim(Custom):
    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an *expression* of the form
        ```
        __operator__(__name__, op)(left, right)
        ```
        given values of `op`, `left` and `right`.
        """
        return ast.Call(
            func=generate_operator_object(self.op, verbatim=True),
            args=[left, right],
            keywords=[],
            lineno=left.lineno,
            col_offset=left.col_offset,
            end_lineno=right.end_lineno,
            end_col_offset=right.end_col_offset,
        )


@dataclass
class Inplace(OperatorBase):
    op: str

    def generate(self, left: ast.expr, right: ast.expr) -> ast.expr:
        """Generates an expression of the form
        ```
        __operator__(__name__, op)(left, right)
        ```
        """
        return Custom(self.op).generate(left, right)


Operator = Inplace | Custom | Identifier | Application

DEFAULT_OPERATOR = "*"


def compute_new_spans(left: TokenInfo, right: TokenInfo) -> Span:
    """Fetches the start & end points for a newly created token based on its leftmost neighbor"""
    return Span(left.end, right.start)


def collect_operator_tokens_inplace(
    toks: list[TokenInfo],
) -> tuple[list[TokenInfo], Mapping[Span, Operator]]:
    """Substitutes instances of infix identifiers (e.g. \\`foo`),
    implicit function application (e.g. `f x`), custom operators
    (e.g. `<$>`) with generic operators, and in-place operators
    (e.g. `>>=`) with their non-inplace variants, returning
    information about the transformed regions.
    """
    tokens.remove_error_whitespace_inplace(toks)

    # TODO: this is an imperative implementation of a parser for a regular
    # language. If only there were some form of regex for token streams...
    collected_spans: dict[Span, Operator] = {}
    i = 0
    while i < len(toks):
        # Function application
        # TODO: handle newline-delimited application by keeping a stack of [](){}
        match toks[i : i + 2]:
            case [
                TokenInfo() as first,
                TokenInfo() as second,
            ] if tokens.valid_application_boundary(first, second):
                tokens.insert_inplace(
                    toks, i + 1, token.OP, DEFAULT_OPERATOR, left_offset=1
                )
                collected_spans[compute_new_spans(toks[i], toks[i + 2])] = Application()
                # 2 since we added 1 extra token we don't need to process anymore
                i += 2
                continue
            case _:
                pass

        # Infixified identifiers:
        # a 3-token pattern replaced with a 1-token pattern,
        # checking 1 token on each end for lookaround
        match toks[i - 1 : i + 4]:
            case [
                TokenInfo(end=(_, left_col)) as first,
                TokenInfo(string="`", start=(_, start_col)),
                TokenInfo(type=token.NAME, string=string),
                TokenInfo(string="`"),
                TokenInfo(),
            ] if not keyword.iskeyword(string):
                tokens.remove_inplace(toks, i, i + 3)
                # after mutation
                offset = start_col - left_col
                tokens.insert_inplace(
                    toks,
                    i,
                    token.OP,
                    DEFAULT_OPERATOR,
                    left_offset=offset,
                    right_offset=-offset,
                )
                collected_spans[
                    compute_new_spans(toks[i - 1], toks[i + 1])
                ] = Identifier(string)
                i += 1
                continue
            case _:
                pass

        # Custom infix operators can be of variable width
        # This also catches in-place operators
        current = toks[i]
        operator = [current]
        is_verbatim = False
        if 0 < i < len(toks) - 1 and tokens.is_custom_operator_token(current):
            # slurp the next token while we can
            while i + len(operator) + 1 < len(toks):
                next = toks[i + len(operator)]
                if tokens.is_custom_operator_token(next) and tokens.is_adjacent(
                    current, next
                ):
                    operator.append(next)
                    current = next
                elif next.string == "`" and tokens.is_adjacent(current, next):
                    operator.append(next)
                    is_verbatim = True
                    break
                else:
                    break

            op_string = "".join(
                tok.string for tok in (operator[:-1] if is_verbatim else operator)
            )
            if (
                is_verbatim or not is_disallowed_operator(op_string)
            ) and not is_builtin_with_kind(op_string, OperatorKind.Symbolic):

                tokens.remove_inplace(toks, i, i + len(operator))
                left_col = toks[i - 1].end[1]
                offset = operator[0].start[1] - left_col

                proxy = operator_proxy_for(op_string)
                bonus = proxy.isalpha()

                tokens.insert_inplace(
                    # We need an extra space of offset for token hygeine,
                    # since the proxy may return `and` and `or`
                    toks,
                    i,
                    token.OP,
                    proxy,
                    left_offset=offset + bonus,
                    right_offset=-offset + bonus,
                )

                kind = (
                    Verbatim
                    if is_verbatim
                    else Inplace
                    if is_builtin_with_kind(op_string, OperatorKind.Inplace)
                    else Custom
                )

                collected_spans[compute_new_spans(toks[i - 1], toks[i + 1])] = kind(
                    op_string
                )

                i += 1
                continue

        # no patterns succeeded
        i += 1

    return toks, collected_spans


def builtin_to_ast_op(op: str) -> ast.operator:
    """Converts from a string to the appropriate operator node"""
    match op:
        case "+" | "+=":
            return ast.Add()
        case "-" | "-=":
            return ast.Sub()
        case "*" | "*=":
            return ast.Mult()
        case "/" | "/=":
            return ast.Div()
        case "//" | "//=":
            return ast.FloorDiv()
        case "%" | "%=":
            return ast.Mod()
        case "**" | "**=":
            return ast.Pow()
        case "<<" | "<<=":
            return ast.LShift()
        case ">>" | ">>=":
            return ast.RShift()
        case "|" | "|=":
            return ast.BitOr()
        case "^" | "^=":
            return ast.BitXor()
        case "@" | "@=":
            return ast.MatMult()
        case _:
            raise ValueError(f"Invalid operator '{op}'")


class HoopyTransformer(ast.NodeTransformer):
    """Class responsible for most of the Hoopy codegen."""

    def __init__(
        self,
        operator_nonce: str,
        custom_spans: Mapping[Span, Operator],
        custom_span_intervals: SpanTree,
    ) -> None:
        super().__init__()
        self.operator_nonce = operator_nonce
        self.custom_spans = custom_spans
        self.span_tree = custom_span_intervals

    def visit_Module(self, node: ast.Module) -> Any:
        # visit child nodes first
        node = self.generic_visit(node)

        # The magic import `from hoopy.magic import *` must go after
        # the module docstring and any __future__ imports
        stmts = node.body
        # this evaluates to 0 or 1 depending on if there's a module docstring
        offset = 0
        if stmts:
            match stmts[0]:
                case ast.Expr(value=ast.Constant(value=str())):
                    offset += 1
                case _:
                    pass

        # skip past every __future__ import
        while offset < len(stmts):
            match stmts[offset]:
                case ast.ImportFrom(module="__future__"):
                    offset += 1
                case _:
                    break

        stmts.insert(offset, MAGIC_IMPORTS)
        return ast.Module(body=stmts, type_ignores=node.type_ignores)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        node = self.generic_visit(node)
        prev = 0
        nodes: list[ast.stmt] = []
        for i, alias in enumerate(node.names):
            demangle_result = demangle_operator_string(alias.name, self.operator_nonce)
            if demangle_result is not None:
                demangled, verbatim = demangle_result
                if prev < i:
                    # TODO: Make the spans tidier for better reporting
                    clone = copy.deepcopy(node)
                    clone.names = clone.names[prev:i]
                    nodes.append(clone)
                nodes.append(generate_import(node, demangled, verbatim=verbatim))
                prev = i + 1
        if prev != len(node.names):
            clone = copy.deepcopy(node)
            clone.names = clone.names[prev:]
            nodes.append(clone)
        return nodes

    def matches_spans(self, left: ast.expr, right: ast.expr) -> Span | None:
        """
        Returns the spans of the operator, if it has been registered. Else, None.
        """
        span = Span(
            (
                getattr(left, "end_lineno", 0) or 0,
                getattr(left, "end_col_offset", 0) or 0,
            ),
            (getattr(right, "lineno", 0) or 0, getattr(right, "col_offset", 0) or 0),
        )
        if span in self.custom_spans:
            return span
        # The span is generated from the left and right endpoints of the binary
        # operator operands. Because of this, it's guaranteed that there's no other
        # AST node inside the span other than the operator token itself. If the input
        # span fully encompasses any entry within the span tree, the entry must be
        # an "expanded" form of the input span.
        return self.span_tree.encompassing_span(span)

    def transform_pair(self, left: ast.expr, right: ast.expr) -> ast.expr | None:
        """
        Transforms a pair of nodes as an Infix
        """
        span = self.matches_spans(left, right)
        # an ?? operator would be nice here
        return (
            None
            if span is None
            else self.custom_spans[span].generate_and_copy_spans(left, right)
        )

    def transform_pair_or(
        self, otherwise: Callable[[ast.expr, ast.expr], ast.expr]
    ) -> Callable[[ast.expr, ast.expr], ast.expr]:
        """
        it is ,,, functional
        """

        def inner(left: ast.expr, right: ast.expr) -> ast.expr:
            return self.transform_pair(left, right) or otherwise(left, right)

        return inner

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # Operators are visited depth-first!
        node = self.generic_visit(node)
        orig_left = node.left
        orig_right = node.right

        def otherwise(left: ast.expr, right: ast.expr):
            clone = copy.deepcopy(node)
            clone.left = left
            clone.right = right
            return clone

        return self.transform_pair_or(otherwise)(orig_left, orig_right)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # Boolean operands are stored as a list, for
        # the purpose of short circuiting
        node = self.generic_visit(node)

        values = node.values.copy()
        i = 0
        while i < len(values) - 1:
            left = values[i]
            right = values[i + 1]
            result = self.transform_pair(left, right)
            if result is None:
                i += 1
            else:
                values[i : i + 2] = [result]

        if len(values) == 1:
            return values[0]
        else:
            return ast.BoolOp(
                node.op,
                values,
                lineno=values[0].lineno,
                col_offset=values[0].col_offset,
                end_lineno=values[-1].end_lineno,
                end_col_offset=values[-1].end_col_offset,
            )

    def visit_Compare(self, node: ast.Compare) -> Any:
        node = self.generic_visit(node)
        # Custom operators don't get comparison chaining behavior
        # Instead, they're combined pairwise into single comparators
        # within a comparison chain
        elems = [node.left, *node.comparators]
        ops = node.ops[:]
        i = 0
        while i + 1 < len(elems):
            left = elems[i]
            right = elems[i + 1]
            transformed = self.transform_pair(left, right)
            if transformed is not None:
                elems[i] = transformed
                del elems[i + 1]
                del ops[i]
            else:
                i += 1

        # a chain of all custom operators
        if len(elems) == 1:
            return elems[0]
        else:
            left, *cmps = elems
            clone = copy.deepcopy(node)
            clone.left = left
            clone.ops = ops
            clone.comparators = cmps
            return clone

    # Single-expression statements
    def visit_Expr(self, node: ast.Expr) -> Any:
        node = self.generic_visit(node)
        # Check if we converted an in-place binary operator from statement to expression form
        # if we did, turn it back into a statement to reduce the overhead
        match node.value:
            case ast.Call(
                func=ast.Call(
                    func=ast.Name("__operator__"),
                    args=[ast.Name("__name__"), ast.Constant(value=op)],
                ),
                args=[target, x],  # binary operator
                keywords=[],
            ) if is_builtin_with_kind(op, OperatorKind.Inplace):
                return ast.AugAssign(target=target, op=builtin_to_ast_op(op), value=x)
            case _:
                return node

    # # for better diagnostics
    # # these are all the AST nodes that refer to identifiers
    # # TODO: perhaps directly go through each exposed node in `ast`,
    # # check which of their attribute types reference `_Identifier`,
    # # and override only those?

    # We don't actually care about the sanctity of python syntax
    # so enjoy this alternative syntax!
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node = self.generic_visit(node)
        match demangle_operator_string(node.name, self.operator_nonce):
            case None:
                return node
            case (op, verbatim):
                return generate_operator_definition(
                    op, node, asynchronous=False, verbatim=verbatim
                )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        node = self.generic_visit(node)
        match demangle_operator_string(node.name, self.operator_nonce):
            case None:
                return node
            case (op, verbatim):
                return generate_operator_definition(
                    op, node, asynchronous=True, verbatim=verbatim
                )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        node = self.generic_visit(node)
        match demangle_operator_string(node.name, self.operator_nonce):
            case None:
                return node
            case (op, verbatim):
                return generate_operator_class(op, node, verbatim=verbatim)

    def visit_Name(self, node: ast.Name) -> Any:
        match demangle_operator_string(node.id, self.operator_nonce):
            case None:
                return node
            case (op, verbatim):
                match node.ctx:
                    case ast.Load():
                        return generate_operator_object(op, verbatim=verbatim)
                    case _:
                        return node

    # def visit_Attribute(self, node: ast.Attribute) -> Any:
    #     return super().visit_Attribute(node)

    # def visit_alias(self, node: ast.alias) -> Any:
    #     return super().visit_alias(node)

    # # Seems like a problem
    # def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
    #     return super().visit_ExceptHandler(node)

    # def visit_arg(self, node: ast.arg) -> Any:
    #     return super().visit_arg(node)

    # def visit_keyword(self, node: ast.keyword) -> Any:
    #     return super().visit_keyword(node)

    # # Nops, I think? Since operators are always module scoped?
    # def visit_Global(self, node: ast.Global) -> Any:
    #     return super().visit_Global(node)

    # def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
    #     return super().visit_Nonlocal(node)


def remove_accidental_indents(
    toks: Iterable[TokenInfo], nonce: str
) -> Sequence[TokenInfo]:
    """Strips accidental indentation caused by the
    `(op) -> __operator_nonce_op ` transformation.

    This is kind of a hack, because I really don't want to write a Python parser.
    """
    out: list[TokenInfo] = []
    # I really don't like this imperative style
    last = None
    indents = 0
    just_indented = False
    for tok, next in itertools.pairwise(toks):
        last = next
        if (
            tok.type == token.INDENT
            and tok.string == " "
            and demangle_operator_string(next.string, nonce) is not None
        ):
            indents += 1
            just_indented = True
        elif tok.type == token.DEDENT and indents > 0:
            indents -= 1
        elif just_indented:
            out.append(tokens.offset(tok, by=-1))
            just_indented = False
        else:
            out.append(tok)
    if last is not None:
        out.append(last)
    return out


def transform(source: str, nonce: str | None = None) -> str:
    """Performs the Hoopy transformations on an input program."""

    if not nonce:
        nonce = str(random.randint(0, 2**32 - 1))

    def with_nonce(fn: Callable[[T, str], U]) -> Callable[[T], U]:
        def inner(x: T) -> U:
            return fn(x, nonce)

        return inner

    # Pipes are very unpythonic. Let's use them here
    toks, spans = (
        Pipe(source)
        | tokens.lex
        | list
        | with_nonce(mangle_operator_objects_inplace)
        | collect_operator_tokens_inplace
    )()

    tree = SpanTree(spans)

    # This additional newline is crucial. Without it, the code runs into a bug in cpython that causes
    # codec behavior to completely explode. Let that be a warning for the rest of you.
    newline = lambda s: s + "\n"

    return (
        Pipe(toks)
        | tokens.unlex
        | tokens.lex
        | with_nonce(remove_accidental_indents)
        | tokens.unlex
        | ast.parse
        | HoopyTransformer(nonce, spans, tree).visit
        | ast.fix_missing_locations
        | ast.unparse
        | newline
    )()
