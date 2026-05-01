"""Centralized config — all knobs live here, loaded from env once.

Model selection follows a **dual-model strategy** ("T-shaped role" architecture):

  - GENERATOR  → user-facing answer. Quality > cost. Defaults to gpt-5-mini,
                 the named successor to o4-mini per OpenAI's model directory.
                 "Near-frontier intelligence for cost sensitive workloads" —
                 exactly the kind of careful-with-numbers task that produced
                 the $482M-vs-$5.4B Q4 net income error in our 5-question
                 evaluation when the lab was still running on gpt-4o-mini.
  
  - JUDGE      → Ragas + HallucinationDetector. Must be smarter than the
                 generator — same model judging itself shares its blindspots.
                 Defaults to gpt-5.4-mini ("our strongest mini model yet").
  
  - VISION     → bulk image/table captioning during ingestion. Volume is high
                 but per-call stakes are low → use the cheap end of GPT-5
                 family that still supports vision. Defaults to gpt-5-mini.

Override any role via env var (see .env.example):
    GENERATOR_MODEL=gpt-5.4
    JUDGE_MODEL=gpt-5.5
    VISION_MODEL=gpt-5-nano
    EMBEDDING_MODEL=text-embedding-3-large

Note (2026-04): gpt-4o-mini, gpt-4o, and o4-mini are now DEPRECATED in OpenAI's
directory. Old defaults left in pricing table for cost-tracker backward compat
when reading old cache files.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable settings loaded from env. Use the `settings` singleton."""
    
    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    upload_dir: Path = field(default_factory=lambda: Path("data/uploads"))
    
    # Models — env-overridable per role, see module docstring
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = 1536
    # `llm_model` is the legacy name for `generator_model`. Kept as alias.
    llm_model: str = field(
        default_factory=lambda: os.getenv("GENERATOR_MODEL", "gpt-5-mini")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-5.4-mini")
    )
    vision_model: str = field(
        default_factory=lambda: os.getenv("VISION_MODEL", "gpt-5-mini")
    )
    
    # Pricing (per 1K tokens) — used by CostTracker
    # Source: https://openai.com/api/pricing as of 2026-04
    # NOTE: Pricing for 5.x models is approximate and may have changed since
    # this code was written. Set OPENAI_PRICING_OVERRIDE env var if you need
    # exact numbers; otherwise these are good for ballpark cost estimates.
    pricing_per_1k: dict = field(default_factory=lambda: {
        # current models (2026-04)
        "gpt-5-nano":      {"input": 0.00005,  "output": 0.00040},
        "gpt-5-mini":      {"input": 0.00025,  "output": 0.00200},
        "gpt-5":           {"input": 0.00125,  "output": 0.01000},
        "gpt-5.4-nano":    {"input": 0.00020,  "output": 0.00125},
        "gpt-5.4-mini":    {"input": 0.00075,  "output": 0.00450},
        "gpt-5.4":         {"input": 0.00250,  "output": 0.01500},
        "gpt-5.5":         {"input": 0.00250,  "output": 0.01500},  # approx
        "gpt-4.1":         {"input": 0.00200,  "output": 0.00800},
        # legacy (deprecated but kept for cache backward compat)
        "gpt-4o-mini":     {"input": 0.000150, "output": 0.000600},
        "gpt-4o":          {"input": 0.0025,   "output": 0.010},
        "o4-mini":         {"input": 0.00110,  "output": 0.00440},
        # embeddings
        "text-embedding-3-small": {"input": 0.000020, "output": 0.0},
        "text-embedding-3-large": {"input": 0.000130, "output": 0.0},
    })
    # Rough cost per VLM image call (gpt-5-mini, ~detail=low, ~85 tokens/img)
    vlm_cost_per_image: float = 0.000425
    
    # Cache
    cache_enabled_default: bool = True
    
    # Limits
    max_pages_default: int | None = None  # None = all pages
    
    @property
    def has_openai_key(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))
    
    @property
    def has_langsmith_key(self) -> bool:
        return bool(os.getenv("LANGSMITH_API_KEY"))


settings = Settings()


# Models that DON'T support custom temperature (must use default of 1.0).
# This applies to GPT-5 family and o-series reasoning models. Older models
# (gpt-4o, gpt-4.1) accept temperature normally.
_NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def supports_temperature(model: str) -> bool:
    """Return True if the given model accepts a custom temperature parameter.
    
    GPT-5 family and o-series reasoning models reject any temperature value
    other than the default (1.0). Older models (gpt-4o, gpt-4.1) accept
    custom values.
    """
    m = (model or "").lower()
    return not any(m.startswith(p) for p in _NO_TEMPERATURE_PREFIXES)


def make_chat_llm(model: str, temperature: float = 0.0, **kwargs):
    """Construct a ChatOpenAI instance, handling the GPT-5/o-series constraint
    that those models only accept temperature=1.

    Note: langchain_openai.ChatOpenAI's class default is temperature=0.7, which
    it WILL send in the API body even when caller omits the kwarg. So for
    GPT-5/o-series we must explicitly pass temperature=1 (the only value the
    API accepts), not just omit it.
    """
    from langchain_openai import ChatOpenAI
    if supports_temperature(model):
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    # GPT-5 / o-series: API requires temperature=1, anything else (including
    # LangChain's silent default of 0.7) is rejected with a 400.
    return ChatOpenAI(model=model, temperature=1, **kwargs)


def configure_langsmith():
    """Idempotent LangSmith setup. Safe to call without an API key (it just no-ops)."""
    if settings.has_langsmith_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "fin-rag-lab")