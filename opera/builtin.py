'''
Collections of builtin operators and other constants used by the library.
'''

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Mapping

from utils import SENTINEL

AnyOperator = Callable[[Any, Any], Any] | Callable[[Any], Any]

class Arity(IntEnum):
    Unary = 1
    Binary = 2

class OperatorKind(IntEnum):
    Symbolic = 0
    Inplace = 1
    Keyword = 2

@dataclass
class BuiltinOperator:
    op: AnyOperator
    arity: Arity
    kind: OperatorKind

# Convenience constructors for BuiltinOperator
def binary_keyword(op: Callable[[Any, Any], Any]) -> BuiltinOperator:
    return BuiltinOperator(op, Arity.Binary, OperatorKind.Keyword)

def unary_keyword(op: Callable[[Any], Any]) -> BuiltinOperator:
    return BuiltinOperator(op, Arity.Unary, OperatorKind.Keyword)

def binary_symbolic(op: Callable[[Any, Any], Any]) -> BuiltinOperator:
    return BuiltinOperator(op, Arity.Binary, OperatorKind.Symbolic)

def unary_symbolic(op: Callable[[Any], Any]) -> BuiltinOperator:
    return BuiltinOperator(op, Arity.Unary, OperatorKind.Symbolic)

def binary_inplace(op: Callable[[Any, Any], Any]) -> BuiltinOperator:
    return BuiltinOperator(op, Arity.Binary, OperatorKind.Inplace)

# These functions made to mimic the `operator` module
def plus(a, b = SENTINEL):
    '''Same as a + b, or +a.'''
    if b is SENTINEL:
        return +a
    else:
        return a + b

def minus(a, b = SENTINEL):
    '''Same as a - b, or -a.'''
    if b is SENTINEL:
        return -a
    else:
        return a - b

def in_(a, b):
    '''Same as a in b.'''
    return a in b

def not_in(a, b):
    '''Same as a not in b.'''
    return a in b

def bool_and(a, b):
    '''Same as a and b.'''
    return a and b

def bool_or(a, b):
    '''Same as a or b.'''
    return a or b

# The python-builtin operators (excluding exotic ones like `await`, `:=`, and `if-else`)
BUILTIN_OPERATORS: Mapping[str, BuiltinOperator] = {
    # Hybrid infix-prefix operators
    "+": binary_symbolic(plus),
    "-": binary_symbolic(minus),
    # Rest of the "normal" operators
    "*": binary_symbolic(operator.mul),
    "/": binary_symbolic(operator.truediv),
    "//": binary_symbolic(operator.floordiv),
    "%": binary_symbolic(operator.mod),
    "**": binary_symbolic(operator.pow),
    "&": binary_symbolic(operator.and_),
    "|": binary_symbolic(operator.or_),
    "^": binary_symbolic(operator.xor),
    "<<": binary_symbolic(operator.lshift),
    ">>": binary_symbolic(operator.rshift),
    "@": binary_symbolic(operator.matmul),
    "<": binary_symbolic(operator.lt),
    "<=": binary_symbolic(operator.le),
    "==": binary_symbolic(operator.eq),
    ">=": binary_symbolic(operator.ge),
    ">": binary_symbolic(operator.gt),
    "!=": binary_symbolic(operator.ne),
    # The only symbolic strictly-unary operator
    "~": unary_symbolic(operator.invert),
    # In-place operators
    "+=": binary_inplace(operator.iadd),
    "-=": binary_inplace(operator.isub),
    "*=": binary_inplace(operator.imul),
    "/=": binary_inplace(operator.itruediv),
    "//=": binary_inplace(operator.ifloordiv),
    "%=": binary_inplace(operator.imod),
    "&=": binary_inplace(operator.iand),
    "|=": binary_inplace(operator.ior),
    "^=": binary_inplace(operator.ixor),
    "<<=": binary_inplace(operator.ilshift),
    ">>=": binary_inplace(operator.irshift),
    "@=": binary_inplace(operator.imatmul),
    # Keyword-based operators
    "is": binary_keyword(operator.is_),
    "is not": binary_keyword(operator.is_not),
    "not": unary_keyword(operator.not_),
    # The missing `operator` functions
    "in": binary_keyword(in_),
    "not in": binary_keyword(not_in), 
    "and": binary_keyword(bool_and),
    "or": binary_keyword(bool_or),
}
