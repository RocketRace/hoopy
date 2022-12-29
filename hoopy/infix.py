"""
User-facing API for infix operators
"""

from __future__ import annotations

import sys
from typing import Callable

from .runtime import (
    is_disallowed_operator,
    BUILTIN_OPERATORS,
    InfixOperator,
    bind_op_to,
)
from .utils import Decorator, FunctionContext, T, U, V, context


def infix(
    op: str, flipped: bool = False
) -> Decorator[[T, U], V, InfixOperator[T, U, V]]:
    """
    This decorator registers a custom infix operator.

    This can be used in a class, to make an operator overload.
    It can also be used outside a class, to register the operator
    globally for any values in the current module.

    # Examples:

    Top-level:
    ```python
    @infix("??")
    def nullish(self, other):
        return None if self is None else other

    print(42 ?? "meaning of life") # output: meaning of life
    print(None ?? "meaning of life") # output: None
    ```

    Using classes:
    ```python
    class Array(list):
        @infix("<$>", flipped=True)
        def fmap(self, function):
            # flipped=True means that the function arguments are
            # reversed, so `x <$> y` calls `y.fmap(x)`
            return Array(map(function, self))

    array = Array([1.5, 0.1, 1.9, 2.0])
    squared = int <$> (round <$> array) # the parentheses are required
    print(*squared) # output: 2 0 2 2
    ```

    # Parameters

    `op`: str - The custom operator to register. The operator may be any string
    containing only the special characters `+-*/%@&|^~!?<>.:$`, i.e. all the
    characters used in builtin operators, as well as the extra characters `?.:$`.

    Certain operators are blacklisted and will raise a `ValueError`. Currently,
    the following operators are disallowed:
    * `.`
    * `...`
    * `=`
    * `:=`
    * `:`
    * `::`
    * `~`

    If the operator is a built-in operator (such as `+`), and the declaration is
    inside a class context, then the decorator will add the appropriate special
    method (such as `__add__`) to the class. Otherwise, this raises `TypeError`.

    `flipped`: bool (default: `False`) - When `flipped=True`, the first
    function argument (e.g. `self`) will be the right value in the operator
    expression. This is similar to the difference between the `__radd__` and `__add__`
    special methods for regular operators. For example:
    ```python
    class Foo:
        @infix("^-^")
        def foo(self, other):
            return f"hi, {other}!"

    class Bar:
        @infix("^-^", flipped=True)
        def bar(self, other):
            returnf "hello there, {other}!"

    print(Foo() ^-^ "foo") # output: hi, foo!
    print("bar" ^-^ Bar()) # output: hello there, bar!
    print("foo" ^-^ Foo()) # raises TypeError
    print(Bar() ^-^ "bar") # raises TypeError
    print("foo" ^-^ "bar") # raises TypeError
    ```

    """
    if is_disallowed_operator(op):
        raise ValueError(f"The operator '{op}' is not allowed")
        # TODO

    def inner(fn: Callable[[T, U], V]) -> InfixOperator[T, U, V]:
        ctx = context(fn)
        if ctx == FunctionContext.Local:
            raise TypeError(
                f"Custom operator '{op}' for '{fn.__qualname__}' cannot be defined in a local scope"
            )
        operator = InfixOperator(op, flipped, fn)
        if ctx == FunctionContext.Global:
            if op not in BUILTIN_OPERATORS:
                raise TypeError(
                    f"Builtin operator '{op}' for '{fn.__qualname__}' cannot be defined in a global scope\n"
                    "Module-scoped operators must not share a name with built-in ones"
                )
            bind_op_to(sys.modules[fn.__module__], operator)
        return operator

    return inner
