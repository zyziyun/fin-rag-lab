"""
Global cost tracker. Singleton-ish: instantiate one per pipeline run.

Tracks LLM/embedding/VLM costs in USD using prices from settings.pricing_per_1k.
Used by every stage that calls an API. Reports flow into IngestionReport.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from src.core.config import settings


@dataclass
class CostTracker:
    """Records API costs per stage. Reset by creating a new instance.

    Cost accuracy notes:
      - For GPT-5 / o-series reasoning models, hidden reasoning tokens are
        counted as output tokens (the OpenAI API bills them at output rates).
        We read these from `response_metadata.token_usage.completion_tokens_details.reasoning_tokens`.
      - Pricing for newer (5.x) models is approximate; check
        https://openai.com/api/pricing for current values and override
        settings.pricing_per_1k as needed.
      - The flat `vlm_cost_per_image` is a fallback when token_usage is
        unavailable; when LangChain returns real token counts we use those.
    """

    by_stage: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    n_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Track reasoning tokens separately so we can show them in reports
    reasoning_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_llm(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> float:
        """Record a chat/completion call. Returns its USD cost.

        reasoning_tokens are billed at the OUTPUT rate (it's how OpenAI charges
        for hidden thinking on reasoning models). We add them to output_tokens
        for cost calculation.
        """
        prices = settings.pricing_per_1k.get(model)
        if not prices:
            # Unknown model — skip cost calc but still count
            cost = 0.0
        else:
            total_output = output_tokens + reasoning_tokens
            cost = (
                input_tokens / 1000.0 * prices["input"]
                + total_output / 1000.0 * prices["output"]
            )
        self.by_stage[stage] += cost
        self.by_model[model] += cost
        self.n_calls[stage] += 1
        if reasoning_tokens:
            self.reasoning_tokens[stage] += reasoning_tokens
        return cost

    @staticmethod
    def extract_token_usage(result) -> dict:
        """Pull (input, output, reasoning) token counts from a LangChain result.
        Returns dict with keys: prompt_tokens, completion_tokens, reasoning_tokens.
        Missing values default to 0.
        """
        md = getattr(result, "response_metadata", None) or {}
        usage = md.get("token_usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
        }

    def record_embedding(
        self, stage: str, model: str, input_tokens: int
    ) -> float:
        """Record an embedding call (no output tokens)."""
        return self.record_llm(stage, model, input_tokens, 0)

    def record_vlm_image(self, stage: str = "vlm_caption") -> float:
        """Record a VLM image call using flat per-image price."""
        cost = settings.vlm_cost_per_image
        self.by_stage[stage] += cost
        self.by_model[settings.vision_model] += cost
        self.n_calls[stage] += 1
        return cost

    @property
    def total(self) -> float:
        return sum(self.by_stage.values())

    def report(self) -> dict:
        return {
            "total_usd": round(self.total, 6),
            "by_stage": {k: round(v, 6) for k, v in self.by_stage.items()},
            "by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "n_calls": dict(self.n_calls),
            "reasoning_tokens": dict(self.reasoning_tokens),
        }

    def summary_line(self) -> str:
        rt_total = sum(self.reasoning_tokens.values())
        rt_note = f", incl. {rt_total} reasoning tokens" if rt_total > 0 else ""
        return f"Total: ${self.total:.4f} ({sum(self.n_calls.values())} calls{rt_note})"