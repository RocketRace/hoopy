from __future__ import annotations

"""
1. Token transformations
    a) Transform objectified operators
        - Find sequence of tokens in between parentheses that is a valid operator
            - Most sequences of gapless OPERATOR tokens with some exceptions
            - Including in-place operators such as +=
            - Including operator keywords such as "in", "is not" and "await"
        - Keep track of the in-place operators used
        - Replace (op) with (__builtin_operators__["op"])
        - Make sure to update existing token spans!
    b) Transform custom operators
        - Find consecutive gapless tokens that are formed of valid operator tokens
        - Store the spans of their arguments
            - Left end, right start
        - Replace the tokens with a single operator token
            - Note that trailing prefix operators may be swallowed, a ++ b is not a + + b
            - Use some static rule to bucket results into 1 ~ 6 precedence levels
            - Possibly: multiplicative, additive, bitwise, arrows
    c) Transform infixified functions
        - Find identifiers surrounded gaplessly by backticks
        - Store spans of the arguments
            - Left end, right start
        - Replace the tokens with a multiplicative operator such as '*'

2. AST transformations
    a) Transform custom and infixified operators
        - NodeTransformer with visit_BinOp (and possibly visit_Compare + visit_BoolOp)
        - Check whether the argument spans were marked by the token pass
        - Replace custom x op y with __infix_operators__["op"](x, y)
        - Replace infixified x `op` y with op(x, y)
    b) Prefix module with `from infix import __builtin_operators__, __infix_operators__`
        - Traverse module body to find the right index to insert, starting at 0:
            - If the child is Expr Constant with a string value, skip one (doc comment)
            - If the child is ImportFrom with module == "__future__", skip one (future import)

Example input:

    # coding: infix
    times = (*)
    five = 2 `times` 2 + 1
    five !? five

Example output:

    from infix import __builtin_operators__, __custom_operators__
    times = __builtin_operators__["*"]
    five = times(1, 1) + 1
    __custom_operators__["!?"](five, five)


Custom operator semantics:
    - Infix objects (via @infix declarations)
        - In a top-level scope, the resulting operator will be bound to the module object and accessible from its globals
        - In a class scope, the resulting operator will be bound to the class object and accessible from its dict
        - In a function scope, raises an exception (locally scoped operators are a bad idea)
        - Can be imported from other modules
            - Therefore must hold a reference to the module they were defined in?

"""

# from .infix import infix

# class Wrapper:
#     def __init__(self, wrapped):
#         self.wrapped = wrapped

#     @infix("??")
#     def maybe(self, other: int) -> int | None:
#         'weee'
#         return None if self.wrapped is None else other

# print(Wrapper(1).maybe)

# print(__infix_operators__["??"](Wrapper(None), 123))
# print(__infix_operators__["??"](None, 123))

# # coding: infix
# import functools
# sum = lambda xs: functools.reduce((+), xs, 0)
# product = lambda xs: functools.reduce((*), xs, 1)
# all = lambda xs: functools.reduce((and), xs, True)
# any = lambda xs: functools.reduce((or), xs, False)

# source = '''
# import\\
# sys
# '''

# from codec import Preprocessor
# import ast

# print(*Preprocessor.tokenize(source), sep="\n")
# print(ast.dump(ast.parse(source), indent=4))
# exec(source)
from .infix import infix as infix
