"""
Vector store utilities for the GraphRAG project.

Step 2 from the README:
- Initialize a sentence-transformers model
- Create a FAISS index
- Provide helper functions:
  - add_embeddings()
  - search()

High-level idea
---------------
1. We use a sentence-transformers model to convert text into vectors
   (a.k.a. embeddings). Each text -> 1 vector (e.g. 384 or 768 dims).

2. We store those vectors in a FAISS index. FAISS is optimized for
   fast nearest-neighbor search in high-dimensional spaces.

3. We keep a Python-side list that maps FAISS row indices to metadata,
   e.g. (note_title, chunk_text), so that from a vector ID we know
   which note / chunk it came from.

4. At query time:
   - We embed the question with the SAME sentence-transformers model.
   - We search the FAISS index for the closest vectors.
   - We return their metadata and similarity scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import faiss  # type: ignore
import numpy as np
import ollama


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Name of the Ollama embedding model.
# From the README:
#   ollama pull toshk0/nomic-embed-text-v2-moe:Q6_K
#
# This is a *different* model from your chat/completion LLM, but it is
# still served by Ollama. We call the embeddings API to get vectors.
EMBEDDING_MODEL_NAME: str = "toshk0/nomic-embed-text-v2-moe:Q6_K"


# ---------------------------------------------------------------------------
# Dataclasses for clarity
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingMetadata:
    """Represents the metadata associated with one embedded chunk.

    Typical usage in this project:
    - note_title: title of the note the chunk came from
    - chunk_text: the text of the chunk that was embedded
    - (you can add other fields like chunk_id, file_path, etc.)
    """

    note_title: str
    chunk_text: str
    # You can add arbitrary extra info if needed, e.g.:
    # extra: Dict[str, Any] | None = None


@dataclass
class SearchResult:
    """Represents one vector search hit."""

    score: float  # lower distance means more similar for L2
    metadata: EmbeddingMetadata


# ---------------------------------------------------------------------------
# Singleton FAISS index
# ---------------------------------------------------------------------------

_index: Optional[faiss.IndexFlatL2] = None

# Python-side mapping from FAISS row index -> EmbeddingMetadata
_id_to_metadata: List[EmbeddingMetadata] = []


def _ensure_index(dimension: int) -> faiss.IndexFlatL2:
    """Create the FAISS index if it doesn't exist yet.

    We use IndexFlatL2, which is the simplest "exact" nearest-neighbor
    index using L2 (Euclidean) distance. For small / medium projects
    this is totally fine.
    """

    global _index

    if _index is None:
        # FAISS expects float32 vectors with a fixed dimension.
        _index = faiss.IndexFlatL2(dimension)

    return _index


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Convert a list of texts into a 2D NumPy array of embeddings using Ollama.

    Shape: (num_texts, embedding_dim)
    Dtype: float32 (FAISS expects float32)

    How it works
    ------------
    For each text we call:

        ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)

    The response looks like:

        {
            "embedding": [float, float, ...],
            "num_tokens": int,
            ...
        }

    We collect the "embedding" lists into a 2D NumPy array.
    """

    if not texts:
        # Return an empty 2D array with 0 rows and 0 columns. The caller
        # should normally check for empty input before calling this.
        return np.zeros((0, 0), dtype="float32")

    vectors: List[List[float]] = []

    # Hard safety limits: if some upstream code accidentally passes a
    # huge string (e.g. entire book), we truncate it before sending it
    # to Ollama to avoid "input length exceeds the context length".
    #
    # Model details (toshk0/nomic-embed-text-v2-moe:Q6_K):
    # - Max context ~512 tokens.
    # - We approximate tokens with:
    #     - a **word cap** to stay well under 512 tokens
    #     - a **character cap** to handle weird no-space languages or
    #       huge single tokens.
    MAX_WORDS_PER_EMBEDDING = 400
    MAX_CHARS_PER_EMBEDDING = 1600

    for text in texts:
        # Truncate by characters first (helps for long unbroken strings).
        if len(text) > MAX_CHARS_PER_EMBEDDING:
            text = text[:MAX_CHARS_PER_EMBEDDING]

        # Then truncate by word count (more directly maps to tokens).
        words = text.split()
        if len(words) > MAX_WORDS_PER_EMBEDDING:
            words = words[:MAX_WORDS_PER_EMBEDDING]
            text = " ".join(words)

        # Call the local Ollama server to get an embedding vector.
        # IMPORTANT:
        # - You must have Ollama running.
        # - You must have pulled the model:
        #       ollama pull toshk0/nomic-embed-text-v2-moe:Q6_K
        #
        # This function is synchronous and will block until the model
        # finishes computing the embedding.
        response = ollama.embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)

        # We expect "embedding" to be a list[float] of fixed size,
        # e.g. 768 or 1024 dimensions depending on the model.
        embedding = response["embedding"]
        vectors.append(embedding)

    embeddings = np.array(vectors, dtype="float32")

    return embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_embeddings(
    texts: Sequence[str],
    metadatas: Optional[Sequence[EmbeddingMetadata]] = None,
) -> List[int]:
    """Embed the given texts and add them to the FAISS index.

    Parameters
    ----------
    texts:
        A list of strings to embed (e.g. note chunks).

    metadatas:
        A list of EmbeddingMetadata objects, **same length** as `texts`.
        Each metadata item should describe one text (note_title, chunk_text, ...).
        If None, we create minimal metadata using the raw text as chunk_text
        and an empty note_title.

    Returns
    -------
    List[int]
        The FAISS IDs (row indices) assigned to these embeddings.
        These are simply [current_size, current_size+1, ...] as we append.
    """

    global _id_to_metadata

    if not texts:
        return []

    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must be the same length as texts (or None).")

    # 1. Compute embeddings
    embeddings = _embed_texts(texts)
    num_vectors, dim = embeddings.shape

    # 2. Ensure FAISS index exists with correct dimension
    index = _ensure_index(dimension=dim)

    # 3. Add vectors to FAISS
    #    FAISS stores them consecutively; it doesn't know about metadata.
    start_id = index.ntotal  # current number of vectors in the index
    index.add(embeddings)
    end_id = index.ntotal  # new total

    # 4. Update Python-side metadata mapping
    new_ids: List[int] = list(range(start_id, end_id))

    if metadatas is None:
        # If no metadata provided, create minimal defaults.
        for text in texts:
            _id_to_metadata.append(
                EmbeddingMetadata(
                    note_title="",
                    chunk_text=text,
                )
            )
    else:
        _id_to_metadata.extend(metadatas)

    return new_ids


def search(
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    """Embed the query and search the FAISS index for similar vectors.

    Parameters
    ----------
    query:
        The user's question or text we want to find similar chunks for.

    top_k:
        How many nearest neighbors to return.

    Returns
    -------
    List[SearchResult]
        A list of SearchResult objects, each containing:
        - score: L2 distance (lower is better)
        - metadata: EmbeddingMetadata (note_title, chunk_text, ...)

    Notes
    -----
    - This function assumes you have already called add_embeddings()
      at least once to populate the index.
    """

    if _index is None or _index.ntotal == 0:
        # No vectors have been added yet.
        return []

    # 1. Embed the query text -> (1, dim) vector
    query_embedding = _embed_texts([query])

    # 2. Run FAISS search
    #    - D: (1, top_k) distances (float32)
    #    - I: (1, top_k) indices (int64)
    distances, indices = _index.search(query_embedding, top_k)

    results: List[SearchResult] = []

    # 3. Map FAISS row indices back to metadata
    for distance, idx in zip(distances[0], indices[0]):
        # FAISS uses -1 for "no result" when the index has fewer vectors
        # than top_k; we skip those.
        if idx == -1:
            continue

        # Safety guard: index might be out of range if our Python-side
        # metadata list got desynced. That shouldn't happen if we only
        # use add_embeddings() to modify both.
        if idx < 0 or idx >= len(_id_to_metadata):
            continue

        metadata = _id_to_metadata[idx]
        results.append(SearchResult(score=float(distance), metadata=metadata))

    return results


# ---------------------------------------------------------------------------
# Simple manual test (run: python -m backend.vector_store)
# ---------------------------------------------------------------------------


def _test_vector_store() -> None:
    """Quick test to verify that embeddings + FAISS search work.

    This is deliberately small and easy to follow:
    - Create a few example sentences.
    - Add them to the index.
    - Search with a query.
    """

    print("Testing vector store...")

    texts = [
        "Neo4j is a graph database.",
        "FAISS is a library for efficient similarity search.",
        "Sentence transformers create embeddings for text.",
        "FastAPI is a web framework for building APIs.",
        "Streamlit is used for building data apps.",
    ]

    metadatas = [
        EmbeddingMetadata(note_title="Neo4j", chunk_text=texts[0]),
        EmbeddingMetadata(note_title="FAISS", chunk_text=texts[1]),
        EmbeddingMetadata(note_title="Embeddings", chunk_text=texts[2]),
        EmbeddingMetadata(note_title="FastAPI", chunk_text=texts[3]),
        EmbeddingMetadata(note_title="Streamlit", chunk_text=texts[4]),
    ]

    print("Adding embeddings...")
    add_embeddings(texts, metadatas)

    query = "How do I search similar vectors efficiently?"
    print(f"\nQuery: {query}\n")

    results = search(query, top_k=3)

    print("Top 3 results (lower distance = more similar):")
    for r in results:
        print(f"- score={r.score:.4f}, note_title={r.metadata.note_title}")
        print(f"  chunk_text={r.metadata.chunk_text}")


if __name__ == "__main__":
    _test_vector_store()
