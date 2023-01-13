"""
Module for enabling and disabling the codec-based transformer.

Interacts with the Python Codec API.
"""

from __future__ import annotations

from encodings import utf_8
import codecs


def searcher(name: str) -> codecs.CodecInfo | None:
    """
    Returns the CodecInfo object associated with the codec name "hoopy", None otherwise
    """
    if name.lower() == "hoopy":
        return codecs.CodecInfo(
            # utf_8.encode arguments are pos-only but CodecInfo requires pos-or-kw
            lambda input, errors="strict": utf_8.encode(input, errors),
            decode_hoopy,
        )


def decode_hoopy(input: bytes, errors: str = "strict") -> tuple[str, int]:
    """
    Transforms an utf-8 encoded source using the hoopy transformer
    """
    from .transform import transform

    source, read = utf_8.decode(input, errors=errors)
    return transform(source), read


def register():
    """Registers the Hoopy codec"""
    codecs.register(searcher)


def unregister():
    """Unregisters the Hoopy codec"""
    codecs.unregister(searcher)
