# hoopy

# This library is a stub. A working release is expected soon.

Hoopy extends the Python language, letting you write a subset of Haskell in your scripts!

- `pip install hoopy` (not yet published)

## Usage

## Caveats

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

(once implemented) Run the regression test suite on any significant code changes.

```bash
$ ./stdlib_test.sh
```
