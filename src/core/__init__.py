"""Core domain types and abstract interfaces.

This module is import-safe (no API clients instantiated at module load).
"""
from .models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    BlockType,
    BoundingBox,
    IngestionReport,
)
from .interfaces import (
    BaseLoader,
    BaseParser,
    BaseCaptioner,
    BaseChunker,
    BaseRetriever,
    BaseGenerator,
)
from .config import Settings, settings

__all__ = [
    "Document",
    "DocumentBlock",
    "DocumentChunk",
    "BlockType",
    "BoundingBox",
    "IngestionReport",
    "BaseLoader",
    "BaseParser",
    "BaseCaptioner",
    "BaseChunker",
    "BaseRetriever",
    "BaseGenerator",
    "Settings",
    "settings",
]
