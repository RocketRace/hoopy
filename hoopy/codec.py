"""
Codec operations and the preprocessor architecture
"""

from __future__ import annotations

import codecs
from encodings import utf_8
from io import StringIO
from tokenize import TokenInfo, generate_tokens, untokenize
from typing import Iterator, Sequence


class Preprocessor:
    def _codec(self) -> codecs.CodecInfo:
        return codecs.CodecInfo(
            # utf_8.encode arguments are pos-only but CodecInfo requires pos-or-kw
            lambda input, errors="strict": utf_8.encode(input, errors),
            self.decode,
        )

    @staticmethod
    def tokenize(source: str) -> Iterator[TokenInfo]:
        return generate_tokens(StringIO(source).readline)

    def decode(self, input: bytes, errors: str = "strict") -> tuple[str, int]:
        source, read = utf_8.decode(input, errors=errors)
        tokens = list(self.tokenize(source))
        return untokenize(self.preprocess_tokens(tokens)), read

    def preprocess_tokens(self, tokens: Sequence[TokenInfo]) -> Sequence[TokenInfo]:
        return tokens
