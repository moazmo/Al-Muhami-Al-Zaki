"""
Edge logic for Al-Muhami Al-Zaki CRAG graph.

Defines conditional routing between nodes based on state.
"""

from typing import Literal

from loguru import logger

from src.graph.state import GraphState
from src.utils.config import get_settings


def route_after_grading(
    state: GraphState,
) -> Literal["generate", "rewrite", "no_answer"]:
    """
    Conditional edge: Decide next step after document grading.

    Decision Matrix:
    ┌─────────────────────────────┬──────────────────┬──────────────┐
    │ Condition                   │ Rewrite Count    │ Decision     │
    ├─────────────────────────────┼──────────────────┼──────────────┤
    │ ≥1 relevant document        │ Any              │ generate     │
    │ 0 relevant documents        │ < max_rewrites   │ rewrite      │
    │ 0 relevant documents        │ ≥ max_rewrites   │ no_answer    │
    └─────────────────────────────┴──────────────────┴──────────────┘

    Args:
        state: Current graph state with graded_documents and rewrite_count

    Returns:
        Next node name: "generate", "rewrite", or "no_answer"
    """
    settings = get_settings()

    has_relevant_docs = len(state["graded_documents"]) > 0
    can_rewrite = state["rewrite_count"] < settings.max_rewrite_attempts

    if has_relevant_docs:
        logger.info(
            f"Routing to GENERATE: {len(state['graded_documents'])} relevant docs"
        )
        return "generate"
    elif can_rewrite:
        logger.info(f"Routing to REWRITE: attempt {state['rewrite_count'] + 1}")
        return "rewrite"
    else:
        logger.warning(
            f"Routing to GENERATE (Fallback): max rewrites ({settings.max_rewrite_attempts}) reached. "
            "Forcing generation with retrieved docs despite low relevance score."
        )
        return "generate"


def should_continue_grading(state: GraphState) -> Literal["continue", "done"]:
    """
    Check if we should continue grading more documents.

    Used for early exit if we already have enough relevant documents.

    Args:
        state: Current graph state

    Returns:
        "continue" or "done"
    """
    # For MVP: grade all documents, no early exit
    # Future optimization: exit early if we have 3+ relevant docs
    return "done"
