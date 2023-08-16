# coding: hoopy

def ($)(f, x):
    return f x

from nullish_hoo import (??)

print $ None ?? 42

add = (+)

# (+) 2 2

# note the parentheses, $ has looser precedence than ==
assert 2 + 2 == 2 `add` 2 == add 2 2 == (add $ 2 $ 2)
