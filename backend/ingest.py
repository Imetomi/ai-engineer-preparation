"""
Ingestion pipeline for the GraphRAG project – **semantic overlinking** version.

Goal
----
- Walk through a folder of Markdown notes (your Obsidian vault).
- For each note:
  - Store it as a `Note` node in Neo4j.
  - Chunk its text and create embeddings via the vector store (Ollama + FAISS).
  - **Automatically create graph links based on semantic similarity**, NOT wikilinks.

Key idea: *semantic overlinking*
--------------------------------
Instead of using explicit `[[wikilinks]]`, we:

1. Embed each chunk of a note using your Ollama embedding model
   (`toshk0/nomic-embed-text-v2-moe:Q6_K` via `vector_store.py`).

2. For each new chunk, we **search** the existing FAISS index for the
   top-k most similar chunks (semantic neighbors).

3. For every neighbor we find, we take its `note_title` and create a
   Neo4j `LINKS_TO` edge between the two notes:

   (CurrentNote)-[:LINKS_TO]->(NeighborNote)

This is how we get fully automatic, semantic connections between notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from neo4j_client import create_link, create_note
from vector_store import EmbeddingMetadata, add_embeddings, search


# ---------------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------------


@dataclass
class NoteFile:
    """Represents one Markdown file on disk."""

    title: str  # derived from filename (without extension)
    content: str  # full text of the file
    path: Path  # full path on disk (for debugging)


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------


def load_markdown_notes(root_dir: str | Path) -> List[NoteFile]:
    """Load all `.md` files under `root_dir` into memory.

    Rules:
    - We treat the **filename without extension** as the note title.
      Example: `/vault/GraphRAG.md` -> title `"GraphRAG"`.
    - We read the whole content of the file as a single string.
    """

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Vault directory does not exist: {root}")

    notes: List[NoteFile] = []

    for path in root.rglob("*.md"):
        # You can customize title extraction if you want folder-based context.
        title = path.stem
        content = path.read_text(encoding="utf-8")
        notes.append(NoteFile(title=title, content=content, path=path))

    return notes


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    max_tokens: int = 200,
    overlap_tokens: int = 30,
) -> List[str]:
    """Very simple text chunker based on word count.

    We don't have access to a real tokenizer here, so we approximate
    "tokens" with whitespace-separated words. This is good enough for
    a personal GraphRAG prototype.

    Strategy:
    - Split text into words.
    - Take sliding windows of size `max_tokens` with overlap
      `overlap_tokens`.
    - Join each window back into a string = one chunk.

    Example:
    - max_tokens = 200, overlap_tokens = 50
    - Chunk 1: words[0 : 200]
    - Chunk 2: words[150 : 350]
    - Chunk 3: words[300 : 500]
    - ...
    """

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []

    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_tokens, n)
        window_words = words[start:end]
        chunk = " ".join(window_words).strip()
        if chunk:
            chunks.append(chunk)

        # Move the window: keep an overlap so chunks share some context.
        if end == n:
            break
        start = max(0, end - overlap_tokens)

    return chunks


# ---------------------------------------------------------------------------
# Core ingestion with semantic overlinking
# ---------------------------------------------------------------------------


def ingest_vault(
    vault_dir: str | Path,
    top_k_semantic_neighbors: int = 5,
) -> None:
    """Ingest all notes from a vault directory with semantic overlinking.

    Steps (high level)
    ------------------
    1. Read all `.md` files -> `NoteFile` objects.
    2. For each note:
       a. Create / update the Neo4j `Note` node with full content.
       b. Chunk the text.
       c. For **each chunk**:
          - Use the current FAISS index to find `top_k_semantic_neighbors`
            most similar chunks (from previously ingested notes).
          - For each similar chunk, create a Neo4j `LINKS_TO` edge between
            the current note and the neighbor's note (semantic overlink).
          - Add the chunk embedding + metadata to the FAISS index so that
            future notes can link to it.

    Important detail: incremental semantic linking
    ---------------------------------------------
    - We process notes one by one.
    - When we process note N, only notes 0..N-1 are in the vector index.
    - That means:
      - Note N will link *to* earlier semantically similar notes.
      - Later notes may also link back to it.
    - This is good enough to get a rich, dense semantic graph.
    """

    # 1. Load all notes from disk
    notes = load_markdown_notes(vault_dir)
    if not notes:
        print(f"No Markdown files found in vault: {vault_dir}")
        return

    print(f"Found {len(notes)} notes in vault: {vault_dir}")

    # 2. First pass: create Neo4j Note nodes with full content.
    #    We do this up front so graph queries can always assume every
    #    title we reference exists as a node.
    for note in notes:
        print(f"[Neo4j] Upserting note node: {note.title}")
        create_note(title=note.title, content=note.content)

    # 3. Second pass: semantic chunking + linking + embeddings.
    for note in notes:
        print(f"\n[Ingest] Processing note: {note.title} ({note.path})")

        chunks = chunk_text(note.content)
        if not chunks:
            print("  - No text chunks produced (empty or whitespace-only file).")
            continue

        print(f"  - Created {len(chunks)} chunks from note content.")

        # We wrap the entire per-note semantic step in a try/except so that
        # if ONE file causes trouble (e.g. Ollama context issues), we just
        # skip that note instead of killing the whole ingestion run.
        try:
            for idx, chunk in enumerate(chunks):
                print(
                    f"    - Chunk {idx + 1}/{len(chunks)}: {len(chunk.split())} words"
                )

                # 3.a Semantic search: find neighbors for this chunk
                #     We only look at notes that already have embeddings
                #     in the FAISS index (earlier notes).
                search_results = search(chunk, top_k=top_k_semantic_neighbors)

                # 3.b For each neighbor, create a Neo4j link between notes.
                #     We ignore neighbors that come from the same note to
                #     avoid self-links.
                seen_neighbor_titles: set[str] = set()

                for result in search_results:
                    neighbor_title = result.metadata.note_title

                    # Skip if metadata has no title (should not happen if
                    # ingest is the only writer) or refers to the same note.
                    if not neighbor_title or neighbor_title == note.title:
                        continue

                    # Avoid creating duplicate edges to the same neighbor
                    # for multiple chunks of this note.
                    if neighbor_title in seen_neighbor_titles:
                        continue

                    seen_neighbor_titles.add(neighbor_title)

                    print(
                        f"      -> Semantic neighbor: {neighbor_title} "
                        f"(distance={result.score:.4f})"
                    )

                    # Create a directed semantic link. You could also choose
                    # to create a reverse link if you want a more undirected
                    # feel:
                    #   create_link(neighbor_title, note.title)
                    create_link(note.title, neighbor_title)

                # 3.c Finally, add this chunk to the vector index so that
                #     future chunks (from later notes) can find it.
                metadata = EmbeddingMetadata(
                    note_title=note.title,
                    chunk_text=chunk,
                )
                add_embeddings([chunk], [metadata])
        except Exception as e:  # noqa: BLE001 - we want to catch anything here
            print(
                f"  !! Skipping note '{note.title}' due to embedding/search error: {e}"
            )
            continue


# ---------------------------------------------------------------------------
# Simple manual test
# ---------------------------------------------------------------------------


def _test_ingest_small_folder() -> None:
    """Tiny test helper to run ingestion on a local folder.

    Usage:
        python -m backend.ingest

    This will:
    - Load Markdown files from the hard-coded path below.
    - Populate Neo4j with one `Note` node per file.
    - Populate FAISS with semantic chunks.
    - Create semantic `LINKS_TO` edges between similar notes.
    """

    # TODO: change this to your real Obsidian vault path or pass it
    # from environment variables.
    demo_vault = (
        "/Users/imetomi/Library/Mobile Documents/"
        "iCloud~md~obsidian/Documents/Obsinotes/1 - Atomic Notes"
    )

    ingest_vault(demo_vault, top_k_semantic_neighbors=5)


if __name__ == "__main__":
    _test_ingest_small_folder()
