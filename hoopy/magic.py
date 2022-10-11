"""
This module exposes the magic `__operator__` dunder function.

This is responsible for accessing and resolving custom operator functions, as well
as accessing builtin operators in objectified form (e.g. `(+)`).
"""
from __future__ import annotations
import sys

from typing import Any, Callable

from .runtime import BUILTIN_OPERATORS, PartialFunction

# The transformer inserts `from hoopy.magic import *` to transformed programs
__all__ = "__operator__", "__import_operator__", "__partial_apply__"


def __operator__(module: str, key: str) -> Callable[[Any, Any], Any]:
    if key in BUILTIN_OPERATORS:
        return BUILTIN_OPERATORS[key]

    def get(left: object, right: object) -> object:
        # TODO: handle right-side operators
        # TODO: handle module-level operators
        if hasattr(type(left), "__infix_operators__"):
            sentinel = object()
            operator = type(left).__infix_operators__.get(key, sentinel)
            if operator is not sentinel:
                return operator(left, right)
        if hasattr(type(right), "__infix_operators_flipped__"):
            sentinel = object()
            operator = type(left).__infix_operators_flipped__.get(key, sentinel)
            if operator is not sentinel:
                return operator(left, right)

        mod = sys.modules[module]
        if hasattr(mod, "__infix_operators__"):
            pass

        raise TypeError(
            f"Unsupported operand type(s) for {key}: '{type(left).__name__}' and '{type(right).__name__}'"
        )

    return get


def __import_operator__(module: str, key: str, from_module: str) -> None:
    operator = __operator__(from_module, key)


def __partial_apply__(function: Any, argument: Any) -> Any:
    if isinstance(function, PartialFunction):
        return function(argument)
    else:
        return PartialFunction(function)(argument)
