"""
User-facing API for infix operators
"""

from __future__ import annotations

import abc
import inspect
import operator
import sys
from enum import IntEnum
from types import ModuleType
from typing import Any, Callable, Generic, Mapping, overload

from .magic import __operator__
from .utils import Decorator, FunctionContext, T, U, V, context


def _op_attr(flipped: bool, target: type[Any] | ModuleType) -> str:
    # Module-level custom operators are always stored in __infix_operators__
    # Overloaded custom operators are stored in either of the two dunders
    if flipped and isinstance(target, type):
        return "__infix_operators_flipped__"
    else:
        return "__infix_operators__"


def _bind_to(
    target: type[Any] | ModuleType, operator: InfixOperator[Any, Any, Any]
) -> None:
    # binds the operator to the appropriate dunder attribute of the object
    # the attribute (dict mapping op -> function) is created if missing
    attr = _op_attr(operator.flipped, target)
    if not hasattr(target, attr):
        setattr(target, attr, {})
    getattr(target, attr)[operator.op] = operator.function


def _unbind_from(
    target: type[Any] | ModuleType, operator: InfixOperator[Any, Any, Any]
) -> None:
    attr = _op_attr(operator.flipped, target)
    del getattr(target, attr)[operator.op]


# The signature of types.MethodType is not guaranteed to be consistent across
# python versions, so we make our own instance method.
class BoundInfixOperator(Generic[T, U, V], abc.ABC):
    """A bound instance method"""

    @abc.abstractmethod
    def __init__(self, function: Callable[[Any, Any], V], instance: T):
        # The documented instance method attributes
        self.__self__ = instance
        self.__doc__ = function.__doc__
        self.__name__ = function.__name__
        self.__module__ = function.__module__

    @abc.abstractmethod
    def __call__(self, other: U) -> V:
        raise NotImplementedError

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.__func__, attr)


class LeftBoundInfixOperator(BoundInfixOperator[T, U, V]):
    """An infix operator with its left argument bound."""

    def __init__(self, function: Callable[[T, U], V], instance: T):
        super().__init__(function, instance)
        self.__func__ = function

    def __call__(self, other: U) -> V:
        return self.__func__(self.__self__, other)


class RightBoundInfixOperator(BoundInfixOperator[T, U, V]):
    """An infix operator with its left argument bound."""

    def __init__(self, function: Callable[[U, T], V], instance: T):
        super().__init__(function, instance)
        self.__func__ = function

    def __call__(self, other: U) -> V:
        return self.__func__(other, self.__self__)


class InfixOperator(Generic[T, U, V]):
    """
    This type facilitates runtime overloads of custom operator
    on a class level. Custom operators bound using this descriptor
    are accessible as regular instance and class methods, as well as
    using the special infix operator syntax.
    """

    def __init__(self, op: str, flipped: bool, function: Callable[[T, U], V]):
        self.op = op
        self.flipped = flipped
        self.function = function
        self.__is_bound = False
        # Attributes for methods
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.__module__ = function.__module__

    def __set_name__(self, owner: type[T], __instance: T):
        # This is only called if the infix operator was not previously
        # bound to a module instance. The binding should also only
        # ever occur at the initial function definition site, which
        # is enforced using the `__is_bound` attribute. It should be
        # impossible for an operator to be bound to both a module and
        # a class instance.
        if not self.__is_bound:
            _bind_to(owner, self)
        self.__is_bound = True

    # Unbound method
    @overload
    def __get__(self, instance: None, owner: type[T]) -> Callable[[T, U], V]:
        ...

    # Bound method
    @overload
    def __get__(self, instance: T, owner: type[T] | None = None) -> Callable[[U], V]:
        ...

    def __get__(
        self, instance: T | None, __owner: type[T] | None = None
    ) -> Callable[[T, U], V] | Callable[[U], V]:
        if instance is None:
            return self
        return LeftBoundInfixOperator(self, instance)

    def __call__(self, left: T, right: U, /) -> V:
        return self.function(left, right)


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
    if op in BLACKLISTED_OPERATOR_STRINGS:
        raise ValueError(f"The operator '{op}' is not allowed")

        # TODO
        pass

    def inner(fn: Callable[[T, U], V]) -> InfixOperator[T, U, V]:
        ctx = context(fn)
        if ctx == FunctionContext.Local:
            raise TypeError(
                f"Custom operator '{op}' for '{fn.__qualname__}' cannot be defined in a local scope"
            )
        operator = InfixOperator(op, flipped, fn)
        if ctx == FunctionContext.Global:
            if BUILTIN_OPERATORS.get(op) is not None:
                raise TypeError(
                    f"Builtin operator '{op}' for '{fn.__qualname__}' cannot be defined in a global scope\n"
                    "Module-scoped operators must not share a name with built-in ones"
                )
            _bind_to(sys.modules[fn.__module__], operator)
        return operator

    return inner


multiplicative = "*/%@.:"
additive = "+-"
bitwise = "|&^~"
comparison = "=<>!?"
other = "$"

AnyOperator = Callable[[Any, Any], Any] | Callable[[Any], Any]


class OperatorKind(IntEnum):
    Symbolic = 0
    Inplace = 1
    Keyword = 2


class BuiltinInfixOperator(InfixOperator):
    def __init__(
        self,
        op: str,
        flipped: bool,
        function: Callable[[Any, Any], Any],
        kind: OperatorKind,
    ):
        super().__init__(op, flipped, function)
        self.kind = kind


# Convenience constructors
def keyword(op: str, fn: Callable[[Any, Any], Any]) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Keyword)


def symbolic(
    op: str, fn: Callable[[Any, Any], Any]
) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Symbolic)


def inplace(op: str, fn: Callable[[Any, Any], Any]) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Inplace)


# These functions made to mimic the `operator` module


def in_(a, b):
    """Same as a in b."""
    return a in b


def not_in(a, b):
    """Same as a not in b."""
    return a in b


def bool_and(a, b):
    """Same as a and b."""
    return a and b


def bool_or(a, b):
    """Same as a or b."""
    return a or b


# The python-builtin operators (excluding exotic ones like `await`, `:=`, and `if-else`)
BUILTIN_OPERATORS: Mapping[str, BuiltinInfixOperator] = dict(
    [
        # The "normal" operators
        symbolic("+", operator.add),
        symbolic("-", operator.sub),
        symbolic("*", operator.mul),
        symbolic("/", operator.truediv),
        symbolic("//", operator.floordiv),
        symbolic("%", operator.mod),
        symbolic("**", operator.pow),
        symbolic("&", operator.and_),
        symbolic("|", operator.or_),
        symbolic("^", operator.xor),
        symbolic("<<", operator.lshift),
        symbolic(">>", operator.rshift),
        symbolic("@", operator.matmul),
        symbolic("<", operator.lt),
        symbolic("<=", operator.le),
        symbolic("==", operator.eq),
        symbolic(">=", operator.ge),
        symbolic(">", operator.gt),
        symbolic("!=", operator.ne),
        # In-place operators
        inplace("+=", operator.iadd),
        inplace("-=", operator.isub),
        inplace("*=", operator.imul),
        inplace("/=", operator.itruediv),
        inplace("//=", operator.ifloordiv),
        inplace("%=", operator.imod),
        inplace("&=", operator.iand),
        inplace("|=", operator.ior),
        inplace("^=", operator.ixor),
        inplace("<<=", operator.ilshift),
        inplace(">>=", operator.irshift),
        inplace("@=", operator.imatmul),
        # Keyword-based operators
        keyword("is", operator.is_),
        keyword("is not", operator.is_not),
        # The missing `operator` functions
        keyword("in", in_),
        keyword("not in", not_in),
        keyword("and", bool_and),
        keyword("or", bool_or),
    ]
)


class PartialFunction:
    """An implementation of a chainable partial function.

    This only supports calling with positional arguments.

    It is used internally to implement Haskell-style curried
    function application, allowing e.g. `f x y z` to be
    functionally equivalent to `f(x, y, z)`.
    """

    __slots__ = ("func", "args", "n")

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.args = []
        spec = inspect.getfullargspec(func)
        self.n = len(spec.args) - len(spec.defaults or ())

    def __call__(self, arg: Any) -> Any:
        if len(self.args) >= self.n - 1:
            return self.func(*self.args, arg)
        else:
            self.args.append(arg)
            return self
