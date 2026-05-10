from __future__ import annotations

import math

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable, wrappers
from openai import OpenAI
from pydantic import BaseModel

from app.rag.agent import build_graph
from app.config import get_settings
from app.rag.utils.tools import close_cached_vectorstore


load_dotenv()
settings = get_settings()

# Wrap the OpenAI client so evaluator LLM calls can be traced in LangSmith
oai_client = wrappers.wrap_openai(OpenAI())
embedding_model = OpenAIEmbeddings(
    model=settings.openai_embedding_model,
    api_key=settings.openai_api_key.get_secret_value(),
)

# Llm-as-a-judge response schema
class HelpfulnessGrade(BaseModel):
    helpful: bool
    reasoning: str


# Define the evaluator function
def helpfulness_evaluator(inputs: dict, outputs: dict) -> dict:
    """Use an LLM judge to decide if the answer is helpful."""
    instructions = """
You are grading the helpfulness of an HR assistant answer.

Mark the answer as helpful if it directly addresses the user's question,
is clear, and gives useful information based on the question.

Mark the answer as not helpful if it is vague, avoids the question,
is clearly incomplete, or does not address what the user asked.
""".strip()

    message = (
        f"Question: {inputs.get('question', '')}\n"
        f"Answer: {outputs.get('answer', '')}"
    )
    # Invoking the evaluator
    response = oai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": message},
        ],
        response_format=HelpfulnessGrade,
    )

    grade = response.choices[0].message.parsed
    return {
        "key": "helpfulness",
        "score": grade.helpful,
        "comment": grade.reasoning,
    }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Define an embedding-based evaluator for retrieval relevance
def retrieval_similarity_evaluator(inputs: dict, outputs: dict) -> dict:
    """Score retrieval relevance with cosine similarity between query and sources."""
    query = inputs.get("query") or inputs.get("question") or ""
    sources = outputs.get("sources", [])

    source_texts = [
        source.get("preview", "")
        for source in sources
        if source.get("preview")
    ]

    if not query or not source_texts:
        return {
            "key": "retrieval_similarity",
            "score": 0.0,
            "comment": "Missing query or retrieved source previews.",
        }

    query_embedding = embedding_model.embed_query(query)
    source_embeddings = embedding_model.embed_documents(source_texts)
    similarities = [
        cosine_similarity(query_embedding, source_embedding)
        for source_embedding in source_embeddings
    ]

    avg_score = sum(similarities) / len(similarities)

    return {
        "key": "retrieval_similarity",
        "score": avg_score,   
        }


# Define the target function that will be evaluated
@traceable
def hr_assistant(inputs: dict) -> dict:
    graph = build_graph()
    try:
        return graph.invoke({"question": inputs["question"]})
    finally:
        close_cached_vectorstore()
