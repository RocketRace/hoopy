Files using the hoopy codec in this directory should be suffixed with "_hoopy.py". This is to exclude them from linters and formatters (which understandably freak out over the files).

At the moment, using encoding comments is broken due to a regression in cpython. Versions 3.9+ are affected, making
this library unusable through them.
