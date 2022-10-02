"""
AST transformations
"""

from __future__ import annotations

import ast


def desugar_infix(op: str, left: ast.expr, right: ast.expr) -> ast.Call:
    """
    Generates an AST node of the form
    ```
    __operator__(__name__, op)(left, right)
    ```
    given values of `op`, `left` and `right`.
    """
    return ast.Call(
        ast.Call(
            ast.Name("__operator__", ctx=ast.Load()),
            args=[ast.Name("__name__", ctx=ast.Load()), ast.Constant(op)],
            keywords=[],
        ),
        args=[left, right],
        keywords=[],
    )
