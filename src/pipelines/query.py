"""
QueryPipeline — LangGraph state machine for the read side of RAG.

Why LangGraph instead of a linear LCEL chain?
  - We need branching (quick vs deep retrieval based on query type)
  - We need a refusal short-circuit when retrieval returns nothing
  - We want every node observable in LangSmith independently

The graph:

    [classify_query]
         │
         ├── (factual_lookup) ── [quick_retrieve] ──┐
         │                                          │
         └── (analytical) ────── [deep_retrieve] ───┤
                                                    │
                              [check_retrieval] ────┤
                                  │                 │
                          ┌───────┴───────┐         │
                          │               │         │
                  (no_results)        (has_results) │
                          │               │         │
                          ▼               ▼         │
                    [refuse]         [generate]     │
                          │               │         │
                          └───────┬───────┘         │
                                  │                 │
                                 END                │

Each node is a pure function of state → state. State is a TypedDict carrying
query, retrieved chunks, answer, citations, and a stage trace for debugging.
"""
from __future__ import annotations
from typing import TypedDict, Literal, Optional, Any
from langgraph.graph import StateGraph, END
from langsmith import traceable

from src.core.interfaces import BaseRetriever, BaseGenerator
from src.core.models import DocumentChunk
from src.observability import CostTracker


# =============================================================
# State
# =============================================================
class QueryState(TypedDict, total=False):
    query: str
    query_type: Literal["factual_lookup", "analytical"]
    chunks: list[DocumentChunk]
    answer: str
    citations: list[str]
    refused: bool
    stages: list[str]                 # debug trace
    metadata: dict[str, Any]


_FACTUAL_KEYWORDS = (
    "what", "how much", "how many", "what was", "what is", "when did", "ratio", "percent",
    "amount", "value", "number"
)


def _classify_query(query: str) -> Literal["factual_lookup", "analytical"]:
    """Cheap heuristic — no LLM call, no cost. Adequate for the lab.
    
    Production: replace with a small LLM call or fine-tuned classifier.
    """
    q = query.lower().strip()
    # Short queries that start with what/how/when → factual lookup
    if any(q.startswith(k) for k in _FACTUAL_KEYWORDS) and len(query.split()) < 12:
        return "factual_lookup"
    return "analytical"


# =============================================================
# Pipeline
# =============================================================
class QueryPipeline:
    """
    Orchestrates retriever + generator via a LangGraph state machine.
    
    Args:
        retriever: BaseRetriever (typically HybridRetriever)
        generator: BaseGenerator (typically RAGGenerator)
        quick_k: top-k for factual lookups (default 3)
        deep_k: top-k for analytical questions (default 8)
        cost_tracker: shared cost tracker
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        quick_k: int = 3,
        deep_k: int = 8,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.quick_k = quick_k
        self.deep_k = deep_k
        self.cost_tracker = cost_tracker
        self.graph = self._build_graph()
    
    # ---- Nodes ----
    def _node_classify(self, state: QueryState) -> QueryState:
        qt = _classify_query(state["query"])
        return {
            **state,
            "query_type": qt,
            "stages": [*state.get("stages", []), f"classify→{qt}"],
        }
    
    def _node_quick_retrieve(self, state: QueryState) -> QueryState:
        chunks = self.retriever.retrieve(state["query"], k=self.quick_k)
        return {
            **state,
            "chunks": chunks,
            "stages": [*state.get("stages", []), f"quick_retrieve→{len(chunks)}"],
        }
    
    def _node_deep_retrieve(self, state: QueryState) -> QueryState:
        chunks = self.retriever.retrieve(state["query"], k=self.deep_k)
        return {
            **state,
            "chunks": chunks,
            "stages": [*state.get("stages", []), f"deep_retrieve→{len(chunks)}"],
        }
    
    def _node_generate(self, state: QueryState) -> QueryState:
        result = self.generator.generate(state["query"], state.get("chunks", []))
        return {
            **state,
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "refused": result.get("refused", False),
            "stages": [*state.get("stages", []), "generate"],
            "metadata": {**state.get("metadata", {}), **{
                k: v for k, v in result.items() if k.startswith("n_")
            }},
        }
    
    def _node_refuse(self, state: QueryState) -> QueryState:
        return {
            **state,
            "answer": "I don't have enough information in the provided sources to answer that question.",
            "citations": [],
            "refused": True,
            "stages": [*state.get("stages", []), "refuse"],
        }
    
    # ---- Edges ----
    @staticmethod
    def _route_after_classify(state: QueryState) -> Literal["quick_retrieve", "deep_retrieve"]:
        return "quick_retrieve" if state["query_type"] == "factual_lookup" else "deep_retrieve"
    
    @staticmethod
    def _route_after_retrieve(state: QueryState) -> Literal["generate", "refuse"]:
        return "generate" if state.get("chunks") else "refuse"
    
    # ---- Build ----
    def _build_graph(self):
        g = StateGraph(QueryState)
        g.add_node("classify", self._node_classify)
        g.add_node("quick_retrieve", self._node_quick_retrieve)
        g.add_node("deep_retrieve", self._node_deep_retrieve)
        g.add_node("generate", self._node_generate)
        g.add_node("refuse", self._node_refuse)
        
        g.set_entry_point("classify")
        g.add_conditional_edges(
            "classify", self._route_after_classify,
            {"quick_retrieve": "quick_retrieve", "deep_retrieve": "deep_retrieve"},
        )
        g.add_conditional_edges(
            "quick_retrieve", self._route_after_retrieve,
            {"generate": "generate", "refuse": "refuse"},
        )
        g.add_conditional_edges(
            "deep_retrieve", self._route_after_retrieve,
            {"generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("generate", END)
        g.add_edge("refuse", END)
        return g.compile()
    
    # ---- Public API ----
    @traceable(name="query_pipeline")
    def query(self, question: str) -> dict[str, Any]:
        initial = QueryState(query=question, stages=[], metadata={})
        final = self.graph.invoke(initial)
        return {
            "query": question,
            "answer": final.get("answer", ""),
            "citations": final.get("citations", []),
            "chunks": final.get("chunks", []),
            "refused": final.get("refused", False),
            "stages": final.get("stages", []),
            "query_type": final.get("query_type"),
        }
    
    def draw_mermaid(self) -> str:
        """Return the graph as Mermaid markup — useful for notebook display."""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return "graph TD\n  A[classify] --> B[retrieve]\n  B --> C[generate]\n  C --> D[END]"
