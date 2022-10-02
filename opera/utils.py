"""
Generic utilities used by the library, with no operator-specific functionality
"""
from __future__ import annotations

import ast
import enum
from typing import Any, Callable, ParamSpec, TypeAlias, TypeVar

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
