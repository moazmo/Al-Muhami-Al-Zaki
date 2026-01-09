"""
GraphState definition for Al-Muhami Al-Zaki CRAG.

Defines the state object passed between all nodes in the
Corrective RAG state machine.
"""

from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document
from operator import add


class GraphState(TypedDict):
    """
    State object for the CRAG (Corrective RAG) graph.

    This state is passed between all nodes and accumulates
    information as the query is processed.

    Attributes:
        question: The user's original legal query
        generation: The final answer (populated by generate node)
        documents: Retrieved documents from Qdrant
        graded_documents: Documents that passed relevance grading
                         (uses Annotated[..., add] for accumulation)
        grade_decision: Routing decision after grading:
                       - "generate": proceed to answer generation
                       - "rewrite": reformulate query and retry
                       - "no_answer": give up after max retries
        rewrite_count: Number of query rewrites attempted (max 2)
        query_history: List of all query versions (for debugging/audit)
    """

    # User input
    question: str

    # Final output
    generation: str

    # Retrieval state
    documents: List[Document]
    graded_documents: Annotated[List[Document], add]

    # Routing state
    grade_decision: Literal["generate", "rewrite", "no_answer"]
    rewrite_count: int

    # Audit trail
    query_history: List[str]


def create_initial_state(question: str) -> GraphState:
    """
    Create an initial state for a new query.

    Args:
        question: The user's legal question

    Returns:
        Initialized GraphState ready for processing
    """
    return GraphState(
        question=question,
        generation="",
        documents=[],
        graded_documents=[],
        grade_decision="",
        rewrite_count=0,
        query_history=[question],
    )
