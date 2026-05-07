from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.rag.utils.state import RAGState, SourcePreview
from app.rag.utils.tools import get_complaint_form, get_retriever, llm

ALLOWED_CATEGORIES = [
    "benefits",
    "complain_form",
    "finance",
    "handbook",
    "leave",
    "onboarding",
    "performance",
    "security",
    "training",
    "travel",
]

IntentType = Literal["agent", "retrieve"]
CategoryType = Literal[
    "benefits",
    "complain_form",
    "finance",
    "handbook",
    "leave",
    "onboarding",
    "performance",
    "security",
    "training",
    "travel",
    "none",
]


class DetectIntentOutput(BaseModel):
    intent: IntentType = Field(
        description="Whether the request should go to the complaint-form agent or retrieval."
    )
    category: CategoryType = Field(
        description="The best matching document category, or `none` if no category applies."
    )


def detect_intent(state: RAGState) -> dict[str, Any]:
    # Use structured output so intent/category validation happens via Pydantic.
    prompt = ChatPromptTemplate.from_template(
        """
You classify user questions for a company policy assistant.

First decide the intent:
- `agent` for complaint form or complaint submission requests
- `retrieve` for policy and handbook questions

If the intent is `retrieve`, also choose exactly one category from this list:
{categories}

If none is a strong fit, use `none` as the category.

Question: {question}
""".strip()
    )
    structured_llm = llm.with_structured_output(DetectIntentOutput)
    result = (prompt | structured_llm).invoke(
        {
            "categories": ", ".join(ALLOWED_CATEGORIES),
            "question": state["question"],
        }
    )

    # Normalize `none` into the empty state used by downstream retrieval logic.
    category = None if result.category == "none" else result.category
    return {"intent": result.intent, "category": category}


def rewrite_query_for_retrieval(state: RAGState) -> dict[str, Any]:
    # Rewrite the original user wording into a tighter retrieval query.
    prompt = ChatPromptTemplate.from_template(
        """
You rewrite user questions to improve retrieval for a company policy RAG system.
Preserve the original meaning.
Make the query specific and keyword-rich.
Use the detected category when it helps focus the query.
Return only the rewritten query.

Detected category: {category}
Original question: {question}
""".strip()
    )
    rewritten = (
        prompt | llm
    ).invoke(
        {
            "question": state["question"],
            "category": state.get("category") or "none",
        }
    ).content.strip()
    return {"rewritten_question": rewritten}


def retrieve(state: RAGState) -> dict[str, Any]:
    # Search with the rewritten query when available, scoped by category metadata.
    query = state.get("rewritten_question") or state["question"]
    retriever = get_retriever(category=state.get("category"))
    docs = retriever.invoke(query)
    sources: list[SourcePreview] = []
    for doc in docs:
        source = doc.metadata.get("source_path") or doc.metadata.get("source") or "unknown"
        preview = doc.page_content[:220].replace("\n", " ").strip()
        sources.append(
            {
                "source": source,
                "category": doc.metadata.get("category"),
                "preview": preview,
            }
        )
    return {"documents": docs, "sources": sources}


def generate(state: RAGState) -> dict[str, Any]:
    # Build a single context block from the retrieved chunks before answering.
    context = "\n\n".join(
        f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(state["documents"])
    )
    prompt = ChatPromptTemplate.from_template(
        """
You are a concise assistant answering questions from retrieved context.
Use only the context below.
If the answer is not supported by the context, say you do not know.
When possible, mention the source number(s) you used.

Context:
{context}

Question:
{question}
""".strip()
    )
    answer = (prompt | llm).invoke(
        {"context": context, "question": state["question"]}
    ).content.strip()
    return {"answer": answer}


def agent(state: RAGState) -> dict[str, Any]:
    """Agent node that serves complaint form requests."""
    # Bind the complaint-form tool for requests that should bypass retrieval.
    llm_with_tools = llm.bind_tools([get_complaint_form])

    messages = [
        SystemMessage(
            content=(
                "You are a helpful HR assistant. "
                "Use the complaint form tool whenever the user asks for a complaint form, "
                "wants to submit a complaint, or asks how to structure a complaint."
            )
        ),
        HumanMessage(content=state["question"])
    ]

    response = llm_with_tools.invoke(messages)

    # Execute the first tool call when the model chooses the complaint form tool.
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call['name'] == 'get_complaint_form':
            answer = get_complaint_form.invoke(tool_call['args'])
        else:
            answer = "Tool not recognized"
    else:
        answer = response.content

    return {"answer": answer}

def route_question(state: RAGState) -> str:
    # Complaint-form traffic goes to the agent; everything else continues to retrieval.
    if state.get("intent") == "agent" or state.get("category") == "complain_form":
        return "agent"

    return "retrieve"
