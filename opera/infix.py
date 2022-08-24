'''
User-facing API for infix operators
'''

from __future__ import annotations

import abc
import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Protocol, overload

from utils import T, U, V

if TYPE_CHECKING:
    from typing_extensions import Self

def _op_attr(flipped: bool) -> str:
    # Done to allow defining these special dunder names in exactly one place in code
    def special_dunder(ops: type):
        return list(ops.__annotations__)[0]
    
    return special_dunder(RightOperators if flipped else LeftOperators)

def _bind_to(target: object, op: str, flipped: bool, fn: Callable[[Any, Any], Any]) -> None:
    attr = _op_attr(flipped)
    if not hasattr(target, attr):
        setattr(target, attr, {})
    getattr(target, attr)[op] = fn

def _unbind_from(target: object, op: str, flipped: bool) -> None:
    attr = _op_attr(flipped)
    del getattr(target, attr)[op]

class LeftOperators(Protocol):
    __infix_operators__: ClassVar[dict[str, Callable[[Self, Any], Any]]]

class RightOperators(Protocol):
    __infix_operators_flipped__: ClassVar[dict[str, Callable[[Self, Any], Any]]]

class Operators(LeftOperators, RightOperators, Protocol):
    pass

multiplicative = "*/%@.:"
additive = "+-"
bitwise = "|&^~"
comparison = "=<>!?"
other = "$"

# The signature of types.MethodType is not guaranteed to be consistent across 
# python versions, so we make our own instance method. 
class BoundInfixOperator(Generic[T, U, V], abc.ABC):
    '''A bound instance method'''
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
    '''An infix operator with its left argument bound.'''
    def __init__(self, function: Callable[[T, U], V], instance: T):
        super().__init__(function, instance)
        self.__func__ = function

    def __call__(self, other: U) -> V:
        return self.__func__(self.__self__, other)

class RightBoundInfixOperator(BoundInfixOperator[T, U, V]):
    '''An infix operator with its left argument bound.'''
    def __init__(self, function: Callable[[U, T], V], instance: T):
        super().__init__(function, instance)
        self.__func__ = function

    def __call__(self, other: U) -> V:
        return self.__func__(other, self.__self__)

class InfixOperator(Generic[T, U, V]):
    '''
    This type facilitates runtime overloads of custom operator
    on a class level. Custom operators bound using this descriptor
    are accessible as regular instance and class methods, as well as 
    using the special infix operator syntax.
    '''
    def __init__(self, op: str, flipped: bool, function: Callable[[T, U], V]):
        self.op = op
        self.flipped = flipped
        self.function = function
        # Attributes for methods
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.__module__ = function.__module__
    
    def __set_name__(self, owner: type[T], __instance: T):
        # Race condition: 
        # There's a period of time between getting bound & unbound that 
        # class-level custom operators are accessible on the module level.
        # In a single thread, this doesn't matter as the descriptor is bound
        # to the class immediately after initialization.
        _unbind_from(sys.modules[self.function.__module__], self.op, self.flipped)
        _bind_to(owner, self.op, self.flipped, self.function)
    
    # Unbound method
    @overload 
    def __get__(self, instance: None, owner: type[T]) -> Callable[[T, U], V]:
        ...

    # Bound method
    @overload 
    def __get__(self, instance: T, owner: type[T] | None = None) -> Callable[[U], V]:
        ...

    def __get__(self, instance: T | None, __owner: type[T] | None = None) -> Callable[[T, U], V] | Callable[[U], V]:
        if instance is None:
            return self
        return LeftBoundInfixOperator(self, instance)
    
    def __call__(self, left: T, right: U, /) -> V:
        return self.function(left, right)
    
def infix(op: str, flipped: bool = False) -> Callable[[Callable[[T, U], V]], InfixOperator[T, U, V]]:
    def inner(fn: Callable[[T, U], V]) -> InfixOperator[T, U, V]:
        _bind_to(sys.modules[fn.__module__], op, flipped, fn)
        return InfixOperator(op, flipped, fn)
    return inner

