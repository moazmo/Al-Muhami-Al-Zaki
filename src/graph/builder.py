"""
LangGraph builder for Al-Muhami Al-Zaki CRAG.

Constructs and compiles the Corrective RAG state machine.
"""

from langgraph.graph import END, StateGraph
from loguru import logger

from src.graph.state import GraphState
from src.graph.nodes import (
    retrieve,
    grade_documents,
    generate,
    rewrite_query,
    no_answer,
)
from src.graph.edges import route_after_grading


def build_crag_graph() -> StateGraph:
    """
    Construct the Corrective RAG graph.

    Graph Structure:

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ retrieve│ ◄─────────────────┐
    └────┬────┘                   │
         │                        │
         ▼                        │
    ┌─────────┐                   │
    │  grade  │                   │
    └────┬────┘                   │
         │                        │
         ▼                        │
    ┌─────────────┐               │
    │   router    │               │
    └──┬────┬────┬┘               │
       │    │    │                │
       ▼    │    ▼                │
    ┌────┐  │  ┌────────┐         │
    │gen │  │  │no_answ │         │
    └──┬─┘  │  └───┬────┘         │
       │    │      │              │
       ▼    │      ▼              │
    ┌─────┐ │   ┌─────┐           │
    │ END │ │   │ END │           │
    └─────┘ │   └─────┘           │
            │                     │
            ▼                     │
       ┌─────────┐                │
       │ rewrite │────────────────┘
       └─────────┘

    Returns:
        Compiled LangGraph ready for invocation
    """
    logger.info("Building CRAG graph")

    # Create the state graph
    workflow = StateGraph(GraphState)

    # -------------------------------------------------------------------------
    # Add nodes
    # -------------------------------------------------------------------------
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("no_answer", no_answer)

    # -------------------------------------------------------------------------
    # Set entry point
    # -------------------------------------------------------------------------
    workflow.set_entry_point("retrieve")

    # -------------------------------------------------------------------------
    # Add edges
    # -------------------------------------------------------------------------

    # retrieve -> grade (always)
    workflow.add_edge("retrieve", "grade")

    # grade -> router (conditional)
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "no_answer": "no_answer",
        },
    )

    # rewrite -> retrieve (loop back)
    workflow.add_edge("rewrite", "retrieve")

    # Terminal nodes -> END
    workflow.add_edge("generate", END)
    workflow.add_edge("no_answer", END)

    # -------------------------------------------------------------------------
    # Compile
    # -------------------------------------------------------------------------
    graph = workflow.compile()

    logger.info("CRAG graph compiled successfully")

    return graph


# Pre-built graph instance for import
_graph = None


def get_crag_graph() -> StateGraph:
    """
    Get the compiled CRAG graph (singleton).

    Returns:
        Compiled LangGraph
    """
    global _graph
    if _graph is None:
        _graph = build_crag_graph()
    return _graph


async def run_query(question: str) -> dict:
    """
    Run a legal question through the CRAG pipeline.

    Convenience function for quick testing.

    Args:
        question: User's legal question in Arabic

    Returns:
        Final state with 'generation' answer
    """
    from src.graph.state import create_initial_state

    graph = get_crag_graph()
    initial_state = create_initial_state(question)

    logger.info(f"Running query: {question[:50]}...")

    final_state = await graph.ainvoke(initial_state)

    return final_state
