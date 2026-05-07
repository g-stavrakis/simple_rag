from __future__ import annotations

import sys

from app.rag.agent import build_graph
from app.rag.utils.tools import close_cached_vectorstore


def main() -> None:
    # Accept a CLI question, or fall back to a simple demo prompt.
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the employee probation period?"
    )

    graph = build_graph()
    try:
        result = graph.invoke({"question": question})
    finally:
        # Release the cached Qdrant client after the request finishes.
        close_cached_vectorstore()

    # Print the final response and the retrieved source previews.
    print("\nQUESTION:\n")
    print(question)
    print("\nANSWER:\n")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:\n")
    for i, source in enumerate(result.get("sources", []), start=1):
        print(f"{i}. {source['source']} :: {source['preview']}...")


if __name__ == "__main__":
    main()
