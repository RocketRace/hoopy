# hoopy

Hoopy extends the Python language, letting you write a subset of Haskell in your scripts!

- `pip install hoopy` (not yet published)

## Example code

```py
# coding: hoopy
def ($)(f, x):
    return f x

from module import (??)

print $ None ?? 42 # prints 42

add = (+)

# note the parentheses, $ has looser precedence than ==
assert 2 + 2 == 2 `add` 2 == add 2 2 == (add $ 2 $ 2)
```

## Setup

The Hoopy library hooks into the Python import system using a custom codec. In a file separate from the
rest of your code, include the following:

```py
import hoopy
hoopy.register()
import your_entry_point_module
```

This ensures that Hoopy is properly initialized before your code is able to be parsed.

Your python files must use a special `# coding: hoopy` declaration at the start of the file to be processed by Hoopy. Otherwise, it will be parsed normally by the Python interpreter and probably immediately raise a dozen syntax errors.

## Custom operators

Hoopy aims to provide a pythonic API to custom operators. This means that they can be defined
both as operator overloads on an object, as well as global functions inside a module:

```py
# coding: hoopy
def (^-^)(x, y):
    return x ** 2 - y ** 2

class Box:
    def __init__(self, value):
        self.value = value
    def (^-^)(self, other):
        return self.value ^-^ other

print(Box(3) ^-^ 2) # prints 5
```

Note that all operators defined in this way are left-associative, and their precedence is determined
statically based on the first character of the operator, following the table below. They may differ slightly from what you would expect from Haskell, which is unavoidable as Haskell operators use all sorts of custom precedence levels.

| First character | Precedence level | Corresponding Python operators |
|-----------------|------------------|--------------------------------|
| `$` | 0 | `or` |
| `?` | 1 | `and` |
| `=`, `~` | 2 | `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `is`, `not in`, `is not` (The python operators have special "comparison" associativity) |
| `\|` | 3 | `\|` |
| `^` | 4 | `^` |
| `&` | 5 | `&` |
| `<`, `>` | 6 | `<<`, `>>` |
| `+`, `-`, `:` | 7 | `+`, `-` |
| `*`, `/`, `%`, `@`, `.`, `!`, as well as partial application and infixified functions | 8 | `*`, `/`, `//`, `%`, `@` |
| (none) | 9 | `**` (Right-associative) |

There are a number of custom operators that will introduce ambiguity in the grammar if defined, such as `:`, `+=`, etc. This includes most operators already used by python, as well as those that may be confused with a binary operator followed by a unary operator. To fix this issue, use a backtick character (`` ` ``) in your custom operators to mark it as a "verbatim" operator. This means that instead of `x >>= y`, you should write ``x >>=` y``.

Custom operators may also be imported from other modules:
```py
from data.functor import (<$>)
```

## Extensions to function syntax

Hoopy allows you to do a number of extra things with functions:
* Turn operators into objects: `(+)` which is treated as a callable object semantically equivalent to `(lambda x, y: x + y)`
* Infixify named functions: ``x `div` y`` is semantically equivalent to `div(x, y)`
* Perform partial application on function objects: `f x` partially applies `x` to the argument list of `f`. If `f` takes N positional arguments, then `f a1 a2 ... an` is equivalent to `f(a1, a2, ..., an)`. If `f` is variadic, or has optional arguments, then the function will be called once the minimum number of necessary arguments have been partially applied. This means that e.g. `print a b` raises an exception, since it's interpreted as `print(a)(b)`, causing a type error on the second function call. A partially applied function can also be called normally with arguments, in which case the wrapped function is invoked directly.

## Caveats

Currently, custom operators are not supported in local scopes. This means that you cannot define a custom
operator inside a function definition: If you attempt to do so, the library will raise an exception. This
is a technically demanding task because local scopes in python are far less dynamic than global scopes.
(This is the same reason why you can't use `from module import *` inside a function definition.)

Hoopy is implemented entirely using Python (making maximal use of the builtin parser and tokenizer).
Given that static typing in Python is optional, Hoopy implements custom operators are using pure-Python
dynamic dispatch. This adds some overhead to custom operators that wouldn't be present if they were a
first-class language feature.

Hoopy's error reporting could be improved. Currently, the AST transformation does not attempt to match the transformed code's spans
to the relevant positions in source code.

Hoopy does its best to ensure that code preprocessed by the library that's *not* using any of its features
remains semantically equivalent to its unprocessed form. This is currently tracked via unit tests running Hoopy
against selected modules within the Python standard library. Future plans include incorporating the builtin
(`python3 -m test`) tests directly into a *fully* preprocessed copy of the standard library.

Oh, and of course: Please don't use this in production code.

## Development

To begin, install the dependencies and set up the development environment. Please run all the unit tests
before submitting any code-related pull requests!

```bash
$ poetry install
$ poetry run pre-commit install
```

## Special thanks

Thanks to @Niki4tap for reviving this project from the ashes and showing that there is a way to sidestep the [`cpython` bug](https://github.com/python/cpython/issues/102353) that I had previously thought made this library impossible! Hoopy lives again 🕊️
