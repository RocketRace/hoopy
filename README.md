# hoopy

Hoopy extends the Python language, letting you write a subset of Haskell in your scripts!

- `pip install hoopy` (not yet published)

## Usage

```py

```

## Caveats

Currently, custom operators are not supported in local scopes. This means that you cannot define a custom
operator inside a function definition: If you attempt to do so, the library will raise an exception. This
is a technically demanding task because local scopes in python are far less dynamic than global scopes.
(This is the same reason why you can't use `from module import *` inside a function definition.)

Hoopy is implemented entirely using Python (making maximal use of the builtin parser and tokenizer).
Given that static typing in Python is optional, Hoopy implements custom operators are using pure-Python
dynamic dispatch. This adds some overhead to custom operators that wouldn't be present if they were a
first-class language feature.

Hoopy does its best to ensure that code preprocessed by the library that's *not* using any of its features
remains semantically equivalent to its unprocessed form. This is currently tracked via unit tests running Hoopy
against selected modules within the Python standard library. Future plans include incorporating the builtin
(`python3 -m test`) tests directly into a *fully* preprocessed copy of the standard library.

## Development

To begin, install the dependencies and set up the development environment. Please run all the unit tests
before submitting any code-related pull requests!

```bash
$ poetry install
$ poetry run pre-commit install
```

## Special thanks

Thanks to @Niki4tap for reviving this project from the ashes and showing that there is a way to sidestep the [`cpython` bug](https://github.com/python/cpython/issues/102353) that I had previously thought made this library impossible! Hoopy lives again 🕊️
