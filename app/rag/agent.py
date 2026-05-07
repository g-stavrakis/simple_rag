from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.rag.utils.nodes import (
    agent,
    detect_intent,
    generate,
    retrieve,
    route_question,
    rewrite_query_for_retrieval,
)
from app.rag.utils.state import RAGInputState, RAGOutputState, RAGState


def build_graph():
    builder = StateGraph(
        RAGState,
        input_schema=RAGInputState,
        output_schema=RAGOutputState,
    )

    # Register the core graph steps.
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("rewrite_query_for_retrieval", rewrite_query_for_retrieval)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("agent", agent)

    # Start by deciding whether this is a retrieval or complaint-form request.
    builder.set_entry_point("detect_intent")
    builder.add_conditional_edges(
        "detect_intent",
        route_question,
        {
            "agent": "agent",
            "retrieve": "rewrite_query_for_retrieval",
        },
    )
    # Retrieval requests are rewritten, searched, and then answered from context.
    builder.add_edge("rewrite_query_for_retrieval", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("agent", END)

    return builder.compile()
