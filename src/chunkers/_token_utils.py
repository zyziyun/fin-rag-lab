"""Shared tokenizer utility. Falls back to char-bytes if tiktoken can't fetch."""
from __future__ import annotations
from functools import lru_cache


@lru_cache(maxsize=1)
def get_token_counter(model: str = "gpt-4o"):
    """Returns a callable that takes a string and returns its token count."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return lambda text: len(enc.encode(text))
    except Exception:
        # Offline / restricted env — fall back to a rough proxy
        return lambda text: len(text.encode("utf-8")) // 4


@lru_cache(maxsize=1)
def get_encoding(model: str = "gpt-4o"):
    """Returns a tiktoken-like object with .encode() and .decode()."""
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Char-level fallback (good enough for chunking math)
        class _Fallback:
            def encode(self, s: str) -> list[int]:
                return list(s.encode("utf-8"))
            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8", errors="replace")
        return _Fallback()
