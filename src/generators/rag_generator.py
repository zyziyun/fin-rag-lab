"""
RAGGenerator — Stage 6: query + retrieved chunks → grounded answer with citations.

Design choices worth knowing in interviews:

  1. Citation format: we tag each chunk with [^1], [^2], ... in the prompt and
     ask the model to include those tags. Then we resolve them back to chunk_ids
     in the response. This is the simplest scheme that lets users click back to
     source. Production systems use structured outputs (JSON tool-calling) for
     stronger guarantees — see 04_generation notebook for the upgrade path.

  2. Refusal handling: if no chunks retrieved, we DON'T call the LLM. We return
     a fixed refusal answer. That saves cost AND prevents the LLM from hallucinating
     to fill the void.

  3. Lazy LLM init: ChatOpenAI is constructed on first call, not at __init__.
     This makes the module import-safe without an API key.
"""
from __future__ import annotations
import re
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from src.core.interfaces import BaseGenerator
from src.core.models import DocumentChunk
from src.core.config import settings
from src.observability import CostTracker


_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst assistant. Your task is to answer questions \
based ONLY on the provided context. Never use outside knowledge.

Rules:
1. **Verify the question matches the context** before answering. If the user asks about \
"Q4 2025 net income" but the context only shows "Q4 2024" or only mentions a specific \
segment's net income (e.g., Consumer Banking), DO NOT use that number for the bank-wide \
answer. Numbers from the wrong period or wrong scope are worse than refusing.

2. **Refusal protocol**: If you cannot answer with confidence, do NOT just say "I don't know". \
Instead, output a structured response with two parts:
   - "What I found in the sources: <briefly state what relevant info IS there, with citations>."
   - "What's missing: <state what would be needed to actually answer>."
   This lets the user follow up productively. Use this protocol whenever the context is \
related but does not contain the specific number / fact requested.

3. **Citations**: Cite every factual claim using [^1], [^2], etc. matching the source \
numbers below. When quoting a number, ALWAYS cite where it came from.

4. **Use specific numbers and direct quotes when available**. Do not paraphrase numbers. \
If a value appears as "$5.4 billion" in context, write "$5.4 billion", not "5.4B" or \
"about 5 billion".

5. Keep the answer concise — 1-3 sentences for fact lookups, up to 5 sentences \
for analytical questions."""),
    ("human", """Context (numbered sources):
{context}

Question: {question}

Answer (with citations):"""),
])


_NO_RESULT_ANSWER = (
    "I could not find any relevant information in the provided sources to answer that question. "
    "No documents were retrieved — please check whether the relevant document is indexed."
)


def _build_context(chunks: list[DocumentChunk]) -> tuple[str, dict[int, str]]:
    """
    Build the numbered context block. Returns (context_string, num_to_chunk_id).
    """
    parts = []
    num_to_chunk_id: dict[int, str] = {}
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        page = f" (p. {chunk.page_number})" if chunk.page_number else ""
        header = f"[Source {i}]" + (f" {heading}" if heading else "") + page
        parts.append(f"{header}\n{chunk.text}")
        num_to_chunk_id[i] = chunk.chunk_id
    return "\n\n".join(parts), num_to_chunk_id


def _extract_citations(answer: str, num_to_chunk_id: dict[int, str]) -> list[str]:
    """Find [^N] citations in the answer and resolve to chunk_ids."""
    nums = [int(n) for n in re.findall(r"\[\^(\d+)\]", answer)]
    seen: set[str] = set()
    out: list[str] = []
    for n in nums:
        cid = num_to_chunk_id.get(n)
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


class RAGGenerator(BaseGenerator):
    """
    Citation-aware answer generation.
    
    Args:
        model: LLM model name (default: settings.llm_model = gpt-5-mini)
        temperature: 0 for factual answers (default)
        cost_tracker: pass a tracker to record API costs
        refuse_on_empty: if True, returns a fixed refusal when no chunks (default: True)
    """
    
    name = "rag_generator"
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        cost_tracker: Optional[CostTracker] = None,
        refuse_on_empty: bool = True,
    ):
        self.model = model or settings.llm_model
        self.temperature = temperature
        self.cost_tracker = cost_tracker
        self.refuse_on_empty = refuse_on_empty
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            from src.core.config import make_chat_llm
            self._llm = make_chat_llm(self.model, temperature=self.temperature)
        return self._llm
    
    @traceable(name="rag_generate")
    def generate(self, query: str, chunks: list[DocumentChunk]) -> dict[str, Any]:
        if not chunks and self.refuse_on_empty:
            return {
                "answer": _NO_RESULT_ANSWER,
                "citations": [],
                "n_sources_used": 0,
                "refused": True,
            }
        
        context, num_to_chunk_id = _build_context(chunks)
        prompt_messages = _RAG_PROMPT.format_messages(context=context, question=query)
        
        result = self._get_llm().invoke(prompt_messages)
        answer = result.content.strip()
        citations = _extract_citations(answer, num_to_chunk_id)
        
        # Cost tracking - read from response_metadata if present, else estimate.
        # extract_token_usage handles the reasoning_tokens field for GPT-5/o-series.
        if self.cost_tracker:
            usage = CostTracker.extract_token_usage(result)
            in_tok = usage["prompt_tokens"] or sum(len(m.content) // 4 for m in prompt_messages)
            out_tok = usage["completion_tokens"] or len(answer) // 4
            reasoning = usage["reasoning_tokens"]
            self.cost_tracker.record_llm("rag_generate", self.model, in_tok, out_tok, reasoning)

        return {
            "answer": answer,
            "citations": citations,
            "n_sources_used": len(citations),
            "n_sources_retrieved": len(chunks),
            "refused": answer.startswith(_NO_RESULT_ANSWER[:20]),
        }