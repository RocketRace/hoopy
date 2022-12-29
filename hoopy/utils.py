"""
Generic utilities used by the library, with no operator-specific functionality
"""
from __future__ import annotations

import ast
from bisect import bisect
import enum
import token
from dataclasses import dataclass
from tokenize import TokenInfo
from typing import Any, Callable, Iterable, NamedTuple, ParamSpec, TypeAlias, TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

P = ParamSpec("P")

Decorator: TypeAlias = Callable[[Callable[P, T]], U]


def singleton(cls: type[T]) -> T:
    return cls()


class Sentinel:
    pass


SENTINEL = Sentinel()


def dump(source: str, mode: str = "exec"):
    print(ast.dump(ast.parse(source, mode=mode), indent=4))


class FunctionContext(enum.IntEnum):
    Global = 0
    Class = 1
    Local = 2


def context(fn: Callable[..., Any]) -> FunctionContext:
    # cpython doesn't expose an attribute in  function objects
    # showing its context (whether it was defined in a class or
    # not), but it does use that information to construct __qualname__

    # Not a dotted name => in global scope
    # Path in <locals> => in local scope
    # Otherwise => in class scope

    if fn.__qualname__ == fn.__name__:
        return FunctionContext.Global
    (remaining, _, _) = fn.__qualname__.rpartition(".")
    if remaining.endswith("<locals>"):
        return FunctionContext.Local
    else:
        return FunctionContext.Class


def print_token(toks: Iterable[TokenInfo]) -> None:
    for tok in toks:
        _, string, start, end, _ = tok
        type_num = tok.exact_type
        type = token.tok_name[type_num]
        print(f"TokenInfo({type=!s}, {string=}, {start=}, {end=})")


def dbg(x: T) -> T:
    match x:
        case [TokenInfo(), *_]:
            print_token(x)  # type: ignore
        case ast.AST():
            print(ast.dump(x, indent=4, include_attributes=True))
        case _:
            print(x)
    return x  # type: ignore


# poorly typed, but static typing in python is weak in this case either way
@dataclass
class Pipe:
    __slots__ = ("value",)
    value: Any

    def __or__(self, fn: Callable[[Any], Any]) -> Pipe:
        return Pipe(fn(self.value))

    def __call__(self) -> Any:
        return self.value


class Span(NamedTuple):
    """The bounds of an operator"""

    start: tuple[int, int]
    end: tuple[int, int]


class SpanTree:
    """Not actually a tree but can be used for binary search so the name is evocative"""

    def __init__(self, spans: Iterable[Span]) -> None:
        # even indices are starting points, odd indices are endpoints
        self.spans = [endpoint for span in spans for endpoint in [span.start, span.end]]

    def encompassing_span(self, span: Span) -> Span | None:
        """Returns the span in this tree containing the input, or None if not found"""
        start_row, start_col = span.start
        # a fractional padding at the start means that an "equal" span is guaranteed to be contained within
        start = (start_row, start_col - 0.5)

        start_index = bisect(self.spans, start)
        end_index = bisect(self.spans, span.end)

        # This assumes that the input span is always "minimal", i.e. contains nothing else in the AST
        if start_index < end_index:
            return Span(self.spans[start_index], self.spans[end_index - 1])
