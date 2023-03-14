"""
The hoopy module implements a subset of Haskell as an extension of the Python grammar.

For more information, see `README.md`.
"""
from __future__ import annotations

from .infix import infix as infix, InfixOperator as InfixOperator
from .codec import register as register, unregister as unregister
