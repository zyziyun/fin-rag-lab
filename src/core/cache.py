"""
Three-layer cache system. Content-addressed (SHA256 of input), so:
  - Cache keys are always correct (input changes → key changes → no stale data)
  - Cache directories are shareable (zip and ship to students)
  - Each layer can be enabled/disabled independently

Layers:
  1. Document cache:  parsed Document (post-VLM) keyed by (file_hash, parser_config)
  2. VLM cache:       caption text keyed by (image_bytes_hash | table_text_hash, model)
  3. Embedding cache: vectors keyed by (chunk_text, model) — uses LangChain's CacheBackedEmbeddings

Disable any layer by passing `enabled=False`. Disable all by setting
`settings.cache_enabled_default = False`.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic

T = TypeVar("T")


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """SHA256 of file contents — used as document cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# =============================================================
# Generic key-value cache base
# =============================================================
@dataclass
class _BaseKVCache(Generic[T]):
    """Tiny on-disk KV store. Keys hashed; values pickled."""
    cache_dir: Path
    enabled: bool = True
    namespace: str = "default"
    
    # Stats
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    
    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        if self.enabled:
            (self.cache_dir / self.namespace).mkdir(parents=True, exist_ok=True)
    
    def _path_for(self, key: str) -> Path:
        return self.cache_dir / self.namespace / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[T]:
        if not self.enabled:
            self.misses += 1
            return None
        p = self._path_for(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with open(p, "rb") as f:
                value = pickle.load(f)
            self.hits += 1
            return value
        except (pickle.UnpicklingError, EOFError):
            # Corrupt cache file — treat as miss
            p.unlink(missing_ok=True)
            self.misses += 1
            return None
    
    def set(self, key: str, value: T) -> None:
        if not self.enabled:
            return
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def clear(self) -> int:
        """Wipe the namespace. Returns # files deleted."""
        if not (self.cache_dir / self.namespace).exists():
            return 0
        n = 0
        for p in (self.cache_dir / self.namespace).iterdir():
            if p.is_file():
                p.unlink()
                n += 1
        return n
    
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "namespace": self.namespace}


# =============================================================
# Layer 1: Document cache
# =============================================================
class DocumentCache(_BaseKVCache):
    """Caches a fully-processed Document (post-parsing, post-VLM-captioning)."""
    
    def make_key(
        self,
        source_path: str | Path,
        parser_name: str,
        max_pages: Optional[int],
        page_range: Optional[tuple[int, int]],
        captioner_model: str,
    ) -> str:
        config_str = f"{parser_name}|{max_pages}|{page_range}|{captioner_model}"
        return f"{file_sha256(source_path)[:16]}_{_sha256(config_str)[:8]}"


# =============================================================
# Layer 2: VLM caption cache
# =============================================================
class VLMCache(_BaseKVCache):
    """Caches a single VLM-generated caption keyed by content."""
    
    def make_key_for_text(self, text: str, model: str) -> str:
        return f"txt_{_sha256(f'{model}|{text}')[:24]}"
    
    def make_key_for_image(self, image_bytes: bytes, model: str) -> str:
        return f"img_{_sha256(model.encode() + image_bytes)[:24]}"


# =============================================================
# Layer 3: Embedding cache (uses LangChain's CacheBackedEmbeddings)
# =============================================================
def make_cached_embeddings(
    base_embeddings,
    cache_dir: Path | str,
    namespace: str,
    enabled: bool = True,
):
    """Wrap an embedding model with on-disk cache. Returns the wrapped model.
    
    LangChain handles the per-text caching internally. We just wire it up.
    """
    if not enabled:
        return base_embeddings
    
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    return CacheBackedEmbeddings.from_bytes_store(
        base_embeddings,
        LocalFileStore(str(cache_dir)),
        namespace=namespace,
    )


# =============================================================
# Convenience: bundle the 3 caches together for a pipeline
# =============================================================
@dataclass
class CacheBundle:
    """All three caches in one object. The IngestionPipeline takes this."""
    docs: DocumentCache
    vlm: VLMCache
    embeddings_dir: Path  # passed to make_cached_embeddings
    
    @classmethod
    def from_root(cls, root: Path | str = "cache", enabled: bool = True) -> "CacheBundle":
        root = Path(root)
        return cls(
            docs=DocumentCache(cache_dir=root, enabled=enabled, namespace="docs"),
            vlm=VLMCache(cache_dir=root, enabled=enabled, namespace="vlm"),
            embeddings_dir=root / "embeddings",
        )
    
    def all_stats(self) -> dict:
        return {
            "docs": self.docs.stats(),
            "vlm": self.vlm.stats(),
        }
