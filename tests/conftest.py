"""Pytest config: patch tiktoken when network is unavailable."""
class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))
    def decode(self, tokens: list[int]) -> str:
        try:
            return bytes(tokens).decode("utf-8", errors="replace")
        except Exception:
            return ""

def _fake_encoding_for_model(_name: str):
    return _FakeEncoding()

def pytest_configure(config):
    try:
        import tiktoken
        try:
            tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            tiktoken.encoding_for_model = _fake_encoding_for_model
            tiktoken.get_encoding = lambda _: _FakeEncoding()
    except ImportError:
        pass
