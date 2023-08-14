"""
Module for enabling and disabling the codec-based transformer.

Interacts with the Python Codec API.
"""

from __future__ import annotations

from encodings import utf_8
import codecs


def encode(input: str, errors="strict"):
    raise NotImplementedError


def decode(input: bytes, errors: str = "strict") -> tuple[str, int]:
    """
    Transforms an utf-8 encoded source using the hoopy transformer
    """
    from .transform import transform

    source, read = utf_8.decode(input, errors=errors)
    return transform(source), read


def searcher(name: str) -> codecs.CodecInfo | None:
    """
    Returns the CodecInfo object associated with the codec name "hoopy", None otherwise
    """
    if name == "hoopy":
        return codecs.CodecInfo(
            name="hoopy",
            encode=encode,
            decode=decode,
            incrementalencoder=utf_8.IncrementalEncoder,
            incrementaldecoder=utf_8.IncrementalDecoder,
            streamreader=utf_8.StreamReader,
            streamwriter=utf_8.StreamWriter,
        )


def register():
    """Registers the Hoopy codec"""
    codecs.register(searcher)


def unregister():
    """Unregisters the Hoopy codec"""
    codecs.unregister(searcher)
