"""
FastAPI backend for the GraphRAG project.

Why this exists
---------------
- FAISS index and embedding metadata live in-memory in `vector_store.py`.
- If you run `ingest.py` and `agent.py` as separate processes, they do
  NOT share that in-memory state, so semantic search returns 0 results.

This FastAPI app keeps everything in **one long-running process** so:
- `/ingest` populates Neo4j + FAISS + metadata.
- `/query` runs the LangGraph-based agent using the same in-memory data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingest import ingest_vault
from agent import query as agent_query


app = FastAPI(
    title="GraphRAG Backend",
    description="FastAPI backend for semantic graph RAG over Obsidian notes.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    vault_path: str = Field(
        ...,
        description="Path to the root folder of your Obsidian vault (or notes).",
        example="/Users/imetomi/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsinotes/1 - Atomic Notes",
    )
    top_k_semantic_neighbors: int = Field(
        5,
        description="How many semantic neighbors to consider when linking notes.",
        ge=1,
        le=20,
    )


class IngestResponse(BaseModel):
    message: str
    vault_path: str
    top_k_semantic_neighbors: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question to ask over the graph.")


class QueryResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    """Simple root endpoint to avoid 404 on GET /."""

    return {
        "message": "GraphRAG backend is running. Use /docs for Swagger UI, /ingest to load your vault, and /query to ask questions."
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(payload: IngestRequest) -> IngestResponse:
    """Ingest notes from a vault directory into Neo4j + FAISS.

    This is intentionally a blocking, long-running call. For a large
    vault it might take a while; you can monitor progress via logs.
    """

    vault = Path(payload.vault_path).expanduser().resolve()
    if not vault.exists():
        raise HTTPException(
            status_code=400, detail=f"Vault path does not exist: {vault}"
        )

    ingest_vault(
        vault_dir=vault,
        top_k_semantic_neighbors=payload.top_k_semantic_neighbors,
    )

    return IngestResponse(
        message="Ingestion completed.",
        vault_path=str(vault),
        top_k_semantic_neighbors=payload.top_k_semantic_neighbors,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest) -> QueryResponse:
    """Ask a question over the ingested knowledge graph.

    NOTE: This assumes `/ingest` has been called at least once since
    the process started, so the FAISS index + metadata are populated.
    """

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must be non-empty.")

    try:
        answer = agent_query(question)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e

    return QueryResponse(answer=answer)


@app.get("/health")
def healthcheck() -> dict:
    """Simple health check endpoint."""

    return {"status": "ok"}


if __name__ == "__main__":
    # Run with:
    #   cd backend
    #   uvicorn main:app --reload
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
