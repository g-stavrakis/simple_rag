from __future__ import annotations

from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class SourcePreview(TypedDict):
    source: str
    category: str | None
    preview: str


class RAGInputState(TypedDict):
    question: str


class RAGOutputState(TypedDict):
    answer: str | None


class RAGState(TypedDict):
    question: str
    intent: str | None
    category: str | None
    rewritten_question: str | None
    documents: list[Document]
    answer: str | None
    sources: list[SourcePreview]
    messages: list[BaseMessage]
