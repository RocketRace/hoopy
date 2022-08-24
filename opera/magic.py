from __future__ import annotations

from typing import Any, Callable, Literal, overload

from builtin import BUILTIN_OPERATORS, AnyOperator, Arity
from infix import LeftBoundInfixOperator, RightBoundInfixOperator
from utils import T, singleton

__all__ = "__operators__",

@singleton
class __operators__:
    @overload
    def get(self, module: str, key: str, *, binary_only: Literal[True]) -> Callable[[Any, Any], Any]:
        ...
    
    @overload
    def get(self, module: str, key: str, *, binary_only: Literal[False] = False) -> AnyOperator:
        ...
    
    def get(self, module: str, key: str, *, binary_only: bool = False) -> AnyOperator:
        if key in BUILTIN_OPERATORS:
            builtin = BUILTIN_OPERATORS[key]
            if binary_only and builtin.arity != Arity.Binary:
                raise TypeError(f"Unary operator '{key}' can't be used as a binary operator")
            return builtin.op
        
        def op(left: Any, right: Any) -> Any:
            # TODO: handle right-side operators
            # TODO: handle module-level operators
            if hasattr(left, "__infix_operators__"):
                sentinel = object()
                op = left.__operators__.get(key, sentinel)
                if op is not sentinel:
                    return op(left, right)
            raise TypeError(f"Unsupported operand type(s) for {key}: '{type(left).__name__}' and '{type(right).__name__}'")
        return op

    def left_bound(self, module: str, key: str, value: T) -> LeftBoundInfixOperator[T, Any, Any]:
        op = self.get(module, key, binary_only=True)
        return LeftBoundInfixOperator(op, value)
    
    def right_bound(self, module: str, key: str, value: T) -> RightBoundInfixOperator[Any, T, Any]:
        op = self.get(module, key, binary_only=True)
        return RightBoundInfixOperator(op, value)
