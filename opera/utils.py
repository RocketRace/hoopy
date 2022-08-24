'''
Generic utilities used by the library, with no operator-specific functionality
'''
from __future__ import annotations

import ast
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

def singleton(cls: type[T]) -> T:
    return cls()

class Sentinel: pass
SENTINEL = Sentinel()

def dump(source: str, mode: str = 'exec'):
    print(ast.dump(ast.parse(source, mode=mode), indent=4))
