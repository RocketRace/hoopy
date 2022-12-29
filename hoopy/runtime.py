from __future__ import annotations

import abc
import inspect
import operator
import re
from enum import IntEnum
from types import ModuleType
from typing import Any, Callable, Generic, Mapping, overload

from .utils import T, U, V


def op_attr(flipped: bool, target: type[Any] | ModuleType) -> str:
    # Module-level custom operators are always stored in __infix_operators__
    # Overloaded custom operators are stored in either of the two dunders
    if flipped and isinstance(target, type):
        return "__infix_operators_flipped__"
    else:
        return "__infix_operators__"


def bind_op_to(
    target: type[Any] | ModuleType, operator: InfixOperator[Any, Any, Any]
) -> None:
    # binds the operator to the appropriate dunder attribute of the object
    # the attribute (dict mapping op -> function) is created if missing
    attr = op_attr(operator.flipped, target)
    if not hasattr(target, attr):
        setattr(target, attr, {})
    getattr(target, attr)[operator.op] = operator.function


def unbind_op_from(
    target: type[Any] | ModuleType, operator: InfixOperator[Any, Any, Any]
) -> None:
    attr = op_attr(operator.flipped, target)
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
            bind_op_to(owner, self)
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


def keyword(op: str, fn: Callable[[Any, Any], Any]) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Keyword)


def symbolic(
    op: str, fn: Callable[[Any, Any], Any]
) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Symbolic)


def inplace(op: str, fn: Callable[[Any, Any], Any]) -> tuple[str, BuiltinInfixOperator]:
    return op, BuiltinInfixOperator(op, False, fn, OperatorKind.Inplace)


# These functions mimic the `operator` module


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


def is_builtin_with_kind(op: str, kind: OperatorKind) -> bool:
    """Returns True if `op` is a builtin operator with kind `kind`, False otherwise."""
    return op in BUILTIN_OPERATORS and BUILTIN_OPERATORS[op].kind is kind


DISALLOWED_OPERATOR_PATTERNS = {
    r"\.\.\.",
    r"\.\.\.:",
    r"\.",
    r":",
    r"::",
    r":\+",
    r":-",
    r":~",
    r"=\+",
    r"=-",
    r"=~",
    r":=",
    r"=",
    r"~",
    r"->",
}


def is_disallowed_operator(op: str) -> bool:
    """Checks against the patterns above"""
    return any(
        re.fullmatch(pattern, op) is not None
        for pattern in DISALLOWED_OPERATOR_PATTERNS
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
        # Not ideal but there is no real alternative
        spec = inspect.getfullargspec(func)
        self.n = len(spec.args) - len(spec.defaults or ())

    def __call__(self, arg: Any) -> Any:
        if len(self.args) >= self.n - 1:
            return self.func(*self.args, arg)
        else:
            self.args.append(arg)
            return self


# Haskell:
# 9 . !!
# 8 ** ^ ^^
# 7 * /
# 6 + - <>
# 5 ++ :
# 4 == < > <= >= /= <$ <$> $> <* <*> *>
# 3 &&
# 2 ||
# 1 >>= >> =<<
# 0 $ $!

# Python:
# 9 ** (right-associative)
# 8 * / // % @
# 7 + -
# 6 << >>
# 5 &
# 4 ^
# 3 |
# 2 == != > < >= <= in is (comparison-associative)
# 1 and
# 0 or

# Hoopy:
# 8 * / @ % . !
# 7 + - :
# 6 < >
# 5 &
# 4 ^
# 3 |
# 2 = ~ (left-associative)
# 1 ?
# 0 $

OPERATOR_PROXIES = {
    "*": "*",
    "/": "*",
    "@": "*",
    "%": "*",
    ".": "*",
    "!": "*",
    "+": "+",
    "-": "+",
    ":": "+",
    "<": "<<",
    ">": "<<",
    "&": "&",
    "^": "^",
    "|": "|",
    "=": "==",
    "~": "==",
    "?": "and",
    "$": "or",
}


def operator_proxy_for(string: str) -> str:
    """Returns the simple operator string associated with the given
    more complex operator string."""
    return OPERATOR_PROXIES[string[0]]
