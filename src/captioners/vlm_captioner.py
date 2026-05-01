"""
VLM/LLM Captioners — Stage 3 of ingestion.

Implements S4 §3.3: turn non-textual blocks (tables, images, charts) into
embedding-friendly natural-language summaries. The summary goes into
block.semantic_content. Raw structured data stays in block.structured_data.

Production concerns built in:
  - Cache: every caption is keyed by content hash → never re-charged
  - Cost tracking: every API call recorded with stage breakdown
  - Retry: tenacity-style retry on transient errors
  - Eager VLM (decision 2 = A): all applicable blocks captioned at ingest time
"""
from __future__ import annotations
import base64
import time
from typing import Optional
from langsmith import traceable

from src.core.interfaces import BaseCaptioner
from src.core.models import DocumentBlock, Document
from src.core.cache import VLMCache
from src.core.config import settings
from src.observability import CostTracker


_TABLE_PROMPT = """Summarize this table in 1-3 sentences for semantic search retrieval.
Mention: (1) what entity/period the table is about, (2) the key metrics or columns,
(3) the most notable values or comparisons. Be specific with numbers when relevant.

Document context: {context}
Heading: {heading}

Table (markdown):
{table}

Summary:"""


_IMAGE_PROMPT = """You are looking at a figure/photo from a financial earnings document.
In 1-3 sentences, describe (1) what the image shows, (2) any visible numbers, labels,
or chart values, (3) what business context it likely supports. Be concrete.

Document context: {context}
Heading: {heading}

Description:"""


class NoOpCaptioner(BaseCaptioner):
    """Skip captioning entirely — useful for fast tests / when no API key."""
    
    name = "noop"
    
    def caption(self, block: DocumentBlock, doc_context: str = "") -> DocumentBlock:
        return block


class GPT4oCaptioner(BaseCaptioner):
    """
    Caption tables and images using a vision-capable model (default gpt-5-mini).
    
    Args:
        cache: VLMCache instance (or None to disable caching for this captioner).
        cost_tracker: CostTracker instance to record costs (or None).
        max_retries: how many times to retry on transient errors.
        model: vision-capable OpenAI model. Default from settings.
    """
    
    name = "gpt4o_captioner"
    
    def __init__(
        self,
        cache: Optional[VLMCache] = None,
        cost_tracker: Optional[CostTracker] = None,
        max_retries: int = 3,
        model: Optional[str] = None,
    ):
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries
        self.model = model or settings.vision_model
        self._llm = None  # lazy
    
    def _get_llm(self):
        if self._llm is None:
            from src.core.config import make_chat_llm
            self._llm = make_chat_llm(self.model, temperature=0)
        return self._llm
    
    @traceable(name="vlm_caption_block")
    def caption(self, block: DocumentBlock, doc_context: str = "") -> DocumentBlock:
        """Caption a single block. Mutates and returns it."""
        if block.block_type == "table":
            block.semantic_content = self._caption_table(block, doc_context)
        elif block.block_type in ("image", "chart", "figure"):
            block.semantic_content = self._caption_image(block, doc_context)
        return block
    
    # ---- private ----
    def _caption_table(self, block: DocumentBlock, doc_context: str) -> str:
        heading = " > ".join(block.heading_path) or "(root)"
        prompt = _TABLE_PROMPT.format(
            context=doc_context,
            heading=heading,
            table=block.text,
        )
        
        # Cache lookup
        if self.cache:
            key = self.cache.make_key_for_text(prompt, self.model)
            cached = self.cache.get(key)
            if cached is not None:
                return cached
        
        # Fresh call
        from langchain_core.messages import HumanMessage
        result = self._invoke_with_retry([HumanMessage(content=prompt)])
        text = result.content.strip()
        
        if self.cache:
            self.cache.set(key, text)
        if self.cost_tracker:
            from src.observability import CostTracker
            usage = CostTracker.extract_token_usage(result)
            in_tok = usage["prompt_tokens"] or len(prompt) // 4
            out_tok = usage["completion_tokens"] or len(text) // 4
            reasoning = usage["reasoning_tokens"]
            self.cost_tracker.record_llm("vlm_caption", self.model, in_tok, out_tok, reasoning)
        return text

    def _caption_image(self, block: DocumentBlock, doc_context: str) -> str:
        if not block.structured_data or "image_bytes" not in block.structured_data:
            return "[Image with no extractable bytes]"

        image_bytes: bytes = block.structured_data["image_bytes"]
        heading = " > ".join(block.heading_path) or "(root)"
        prompt_text = _IMAGE_PROMPT.format(context=doc_context, heading=heading)

        # Cache lookup keyed on image bytes
        if self.cache:
            key = self.cache.make_key_for_image(image_bytes, self.model)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        # Build vision message
        from langchain_core.messages import HumanMessage
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
        ])

        result = self._invoke_with_retry([message])
        text = result.content.strip()

        if self.cache:
            self.cache.set(key, text)
        if self.cost_tracker:
            # Prefer real token usage from API response. Fall back to flat
            # per-image rate only if response_metadata is missing.
            from src.observability import CostTracker
            usage = CostTracker.extract_token_usage(result)
            if usage["prompt_tokens"] or usage["completion_tokens"]:
                self.cost_tracker.record_llm(
                    "vlm_caption", self.model,
                    usage["prompt_tokens"], usage["completion_tokens"],
                    usage["reasoning_tokens"],
                )
            else:
                self.cost_tracker.record_vlm_image("vlm_caption")
        return text

    def _invoke_with_retry(self, messages):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return self._get_llm().invoke(messages)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        # All retries exhausted
        raise RuntimeError(f"VLM caption failed after {self.max_retries} retries: {last_exc}")