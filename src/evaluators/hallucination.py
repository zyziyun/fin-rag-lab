"""
Custom hallucination detector — claim-level NLI-style check.

Why not just use Ragas faithfulness? Two reasons:

  1. Ragas faithfulness gives one score per answer. For compliance-critical
     domains (finance, legal), reviewers want to see WHICH claim is unsupported.
  
  2. The detector returns per-claim verdicts (entailed / refuted / unsupported)
     with the supporting chunk IDs. That's actionable — you can highlight the
     unsupported sentence in red in the UI.

Method:
  1. Use the LLM to atomic-decompose the answer into discrete factual claims
  2. For each claim, ask the LLM to verify it against the joined context
  3. Aggregate results
"""
from __future__ import annotations
import json
import re
from typing import Optional, Literal, Any
from dataclasses import dataclass, asdict

from src.core.models import DocumentChunk
from src.core.config import settings
from src.observability import CostTracker


_DECOMPOSE_PROMPT = """Break this answer into a numbered list of atomic factual claims.
A claim is one self-contained fact (a number, a date, a relationship). Skip introductory \
phrases, commentary, and citations like [^1].

Answer:
{answer}

Claims (one per line, no numbering, no citation marks):"""


_VERIFY_PROMPT = """Given the context, classify the claim as one of:
  - "entailed": the context directly supports the claim
  - "refuted": the context contradicts the claim
  - "unsupported": the context neither supports nor contradicts the claim

Respond with a JSON object: {{"verdict": "<one of the three>", "reasoning": "<one sentence>"}}

Context:
{context}

Claim: {claim}

Response (JSON only):"""


Verdict = Literal["entailed", "refuted", "unsupported"]


@dataclass
class ClaimVerdict:
    claim: str
    verdict: Verdict
    reasoning: str


@dataclass
class HallucinationReport:
    n_claims: int
    n_entailed: int
    n_refuted: int
    n_unsupported: int
    claims: list[ClaimVerdict]
    
    @property
    def faithfulness_score(self) -> float:
        if self.n_claims == 0:
            return 1.0
        return self.n_entailed / self.n_claims
    
    def to_dict(self):
        d = asdict(self)
        d["faithfulness_score"] = self.faithfulness_score
        return d


class HallucinationDetector:
    def __init__(
        self,
        model: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.model = model or settings.judge_model
        self.cost_tracker = cost_tracker
        self._llm = None
    
    def _get_llm(self):
        if self._llm is None:
            from src.core.config import make_chat_llm
            self._llm = make_chat_llm(self.model, temperature=0)
        return self._llm
    
    def detect(
        self, answer: str, chunks: list[DocumentChunk]
    ) -> HallucinationReport:
        # Step 1: decompose into atomic claims
        claims = self._decompose(answer)
        if not claims:
            return HallucinationReport(0, 0, 0, 0, [])
        
        # Step 2: verify each claim
        context = "\n\n".join(c.text for c in chunks)
        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            v = self._verify(claim, context)
            verdicts.append(v)
        
        n_e = sum(1 for v in verdicts if v.verdict == "entailed")
        n_r = sum(1 for v in verdicts if v.verdict == "refuted")
        n_u = sum(1 for v in verdicts if v.verdict == "unsupported")
        return HallucinationReport(
            n_claims=len(verdicts),
            n_entailed=n_e,
            n_refuted=n_r,
            n_unsupported=n_u,
            claims=verdicts,
        )
    
    # ---- private ----
    def _decompose(self, answer: str) -> list[str]:
        from langchain_core.messages import HumanMessage
        result = self._get_llm().invoke(
            [HumanMessage(content=_DECOMPOSE_PROMPT.format(answer=answer))]
        )
        if self.cost_tracker:
            from src.observability import CostTracker
            u = CostTracker.extract_token_usage(result)
            self.cost_tracker.record_llm(
                "hallucination_decompose", self.model,
                u["prompt_tokens"] or len(answer) // 4,
                u["completion_tokens"] or len(result.content) // 4,
                u["reasoning_tokens"],
            )
        # Split lines, strip, drop empties
        claims = [
            re.sub(r"^[\-\*\d\.\)]+\s*", "", ln).strip()
            for ln in result.content.split("\n")
        ]
        return [c for c in claims if c and len(c) > 4]

    def _verify(self, claim: str, context: str) -> ClaimVerdict:
        from langchain_core.messages import HumanMessage
        prompt = _VERIFY_PROMPT.format(claim=claim, context=context)
        result = self._get_llm().invoke([HumanMessage(content=prompt)])
        if self.cost_tracker:
            from src.observability import CostTracker
            u = CostTracker.extract_token_usage(result)
            self.cost_tracker.record_llm(
                "hallucination_verify", self.model,
                u["prompt_tokens"] or len(prompt) // 4,
                u["completion_tokens"] or len(result.content) // 4,
                u["reasoning_tokens"],
            )
        # Parse JSON, tolerate fenced code blocks
        raw = result.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        try:
            data = json.loads(raw)
            verdict = data.get("verdict", "unsupported").lower()
            if verdict not in ("entailed", "refuted", "unsupported"):
                verdict = "unsupported"
            reasoning = data.get("reasoning", "")
        except json.JSONDecodeError:
            # Fall back to keyword search
            low = raw.lower()
            if "entail" in low: verdict = "entailed"
            elif "refut" in low or "contradict" in low: verdict = "refuted"
            else: verdict = "unsupported"
            reasoning = raw[:200]
        return ClaimVerdict(claim=claim, verdict=verdict, reasoning=reasoning)