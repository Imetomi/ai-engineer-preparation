"""
LangGraph-based query agent for the GraphRAG project.

High-level architecture
-----------------------
We build a tiny LangGraph with THREE "agents" (graph nodes):

1. OrchestratorAgent  (entry/exit point)
   - Receives the user's question.
   - Calls the SearchAgent to gather context.
   - Calls the SummarizerAgent to turn context into an answer.
   - Returns the final answer string.

2. SearchAgent
   - Uses the vector store (FAISS + Ollama embeddings) to find
     semantically similar chunks for the question.
   - Collects the note titles from those chunks.
   - Uses Neo4j to fetch graph neighbors for those notes.
   - Produces a big context string (semantic + graph).

3. SummarizerAgent
   - Calls Ollama (LLM) with a prompt that includes:
       - the user's original question
       - the context from SearchAgent
   - Produces a concise, helpful answer.

The public entrypoint is the `query(question: str) -> str` function
at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ollama
from langgraph.graph import END, StateGraph

from neo4j_client import get_neighbors
from vector_store import EmbeddingMetadata, SearchResult, search


# ---------------------------------------------------------------------------
# State definition for LangGraph
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """State that flows through the LangGraph.

    We keep it intentionally small and explicit.
    """

    question: str
    # Context string built by SearchAgent (semantic + graph).
    context: str = ""
    # Final answer text from SummarizerAgent.
    answer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """LangGraph works with dict-like state, so we provide helpers."""

        return {
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Convert dict state back into AgentState."""

        return cls(
            question=data.get("question", ""),
            context=data.get("context", ""),
            answer=data.get("answer", ""),
        )


# ---------------------------------------------------------------------------
# Config: models and prompt templates
# ---------------------------------------------------------------------------

# LLM to use via Ollama. This should already be pulled, see README:
#   ollama pull qwen3-vl:2b-instruct
LLM_MODEL_NAME: str = "qwen3-vl:2b-instruct"


def _build_system_prompt() -> str:
    """System prompt for the summarizer agent.

    This is where you define the "character" and behavior of the agent.
    """

    return (
        "You are a helpful assistant answering questions based on a personal "
        "knowledge graph built from Obsidian notes. Use ONLY the provided "
        "context to answer. If the answer is uncertain, say that explicitly.\n\n"
        "Be concise, structured, and clear. You can use bullet points and short "
        "paragraphs, but avoid unnecessary fluff."
    )


# ---------------------------------------------------------------------------
# Node 2: SearchAgent - semantic + graph retrieval
# ---------------------------------------------------------------------------


def search_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: find semantic + graph context for the question.

    Steps:
    1. Run vector search on the question to get top-k similar chunks.
    2. For each chunk, grab its note_title.
    3. For a small set of distinct note titles:
       - Use Neo4j to fetch the note and its neighbors.
    4. Build a big context string that includes:
       - High-similarity chunks.
       - Graph neighbors (titles + contents).
    5. Store that context in the state.
    """

    print("\n[Orchestrator -> SearchAgent] Starting semantic + graph retrieval...")
    agent_state = AgentState.from_dict(state)
    question = agent_state.question
    print(f"[SearchAgent] Question: {question!r}")

    # 1. Vector search: get top-k similar chunks.
    top_k_chunks = 8
    search_results: List[SearchResult] = search(question, top_k=top_k_chunks)
    print(
        f"[SearchAgent] Retrieved {len(search_results)} semantic chunks from vector store."
    )

    if not search_results:
        print("[SearchAgent] No semantic hits found. Context will be empty.")
        # No semantic hits at all -> context stays empty; summarizer will
        # have to fall back to generic reasoning / say "I don't know".
        agent_state.context = ""
        return agent_state.to_dict()

    # 2. Collect note titles referenced by these chunks.
    titles_in_order: List[str] = []
    for result in search_results:
        title = result.metadata.note_title
        if title and title not in titles_in_order:
            titles_in_order.append(title)

    print("[SearchAgent] Note titles from semantic search (in order):")
    for t in titles_in_order:
        print(f"  - {t}")

    # Optionally limit how many distinct notes we expand in the graph to
    # avoid huge prompts.
    max_notes_for_graph = 5
    selected_titles = titles_in_order[:max_notes_for_graph]
    print(
        f"[SearchAgent] Selecting up to {max_notes_for_graph} titles for graph expansion:"
    )
    for t in selected_titles:
        print(f"  * {t}")

    # 3. For each selected note title, fetch its neighbors from Neo4j.
    graph_snippets: List[str] = []

    for title in selected_titles:
        neighbors = get_neighbors(title, depth=1)
        if neighbors is None:
            print(f"[SearchAgent] No graph data found for note {title!r}.")
            continue

        print(f"[SearchAgent] Graph neighbors for note {title!r}:")

        # Center note
        center = neighbors.center
        center_block = [
            f"NOTE: {center.title}",
            center.content.strip(),
        ]

        # Neighbor notes (immediate neighbors)
        neighbor_blocks: List[str] = []
        for nb in neighbors.neighbors:
            print(f"    - neighbor: {nb.title}")
            neighbor_blocks.append(f"- NEIGHBOR NOTE: {nb.title}\n{nb.content.strip()}")

        snippet = "\n".join(center_block)
        if neighbor_blocks:
            snippet += "\nNEIGHBORS:\n" + "\n\n".join(neighbor_blocks)

        graph_snippets.append(snippet)

    # 4. Build semantic chunk snippets from FAISS results.
    semantic_snippets: List[str] = []
    for idx, result in enumerate(search_results, start=1):
        meta: EmbeddingMetadata = result.metadata
        semantic_snippets.append(
            f"[CHUNK {idx}] (note: {meta.note_title}, distance={result.score:.4f})\n"
            f"{meta.chunk_text.strip()}"
        )

    # Final context string fed to the summarizer.
    context_parts: List[str] = []
    if semantic_snippets:
        context_parts.append("SEMANTIC CHUNKS:\n" + "\n\n".join(semantic_snippets))
    if graph_snippets:
        context_parts.append("GRAPH CONTEXT:\n" + "\n\n".join(graph_snippets))

    agent_state.context = "\n\n---\n\n".join(context_parts)
    print("[SearchAgent] Finished building context, passing to SummarizerAgent.")

    return agent_state.to_dict()


# ---------------------------------------------------------------------------
# Node 3: SummarizerAgent - call Ollama LLM
# ---------------------------------------------------------------------------


def summarizer_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: summarize context + question into an answer.

    We call Ollama's chat API with a system + user message:
    - system: instructions for behavior
    - user: question + context
    """

    print("\n[SearchAgent -> SummarizerAgent] Starting summarization...")
    agent_state = AgentState.from_dict(state)

    system_prompt = _build_system_prompt()

    # Build the user message combining question + context.
    if agent_state.context:
        user_content = (
            f"Question:\n{agent_state.question}\n\n"
            f"Context (from semantic + graph search):\n"
            f"{agent_state.context}\n\n"
            "Answer the question using ONLY this context."
        )
    else:
        user_content = (
            f"Question:\n{agent_state.question}\n\n"
            "There is no additional context available from the knowledge base. "
            "If the answer is not obvious, say that you don't know based on the "
            "current data."
        )

    print("[SummarizerAgent] Calling Ollama LLM for final answer...")
    response = ollama.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    # The response format from Ollama's chat endpoint typically looks like:
    #   {"message": {"role": "assistant", "content": "..."}, ...}
    answer_text = response.get("message", {}).get("content", "").strip()

    print("\n[SummarizerAgent] Summary / answer:")
    print("-----------------------------------")
    print(answer_text)
    print("-----------------------------------\n")

    agent_state.answer = answer_text
    return agent_state.to_dict()


# ---------------------------------------------------------------------------
# Node 1: OrchestratorAgent - orchestrates the flow
# ---------------------------------------------------------------------------


def orchestrator_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: entry node, currently a no-op.

    For now, this node doesn't modify the state; it just exists as the
    starting point. In a more advanced setup, you could:
    - Classify the user intent.
    - Decide whether to call SearchAgent at all.
    - Route between different tools / subgraphs.
    """

    print("\n[OrchestratorAgent] Received new question, calling SearchAgent...")
    # We just echo the state through.
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    """Build a LangGraph with three nodes wired in sequence:

    OrchestratorAgent -> SearchAgent -> SummarizerAgent -> END
    """

    graph = StateGraph(dict)  # we pass around dict-based state

    # Register nodes
    graph.add_node("orchestrator", orchestrator_agent_node)
    graph.add_node("search", search_agent_node)
    graph.add_node("summarizer", summarizer_agent_node)

    # Set entry point
    graph.set_entry_point("orchestrator")

    # Wiring: orchestrator -> search -> summarizer -> END
    graph.add_edge("orchestrator", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", END)

    return graph


# Build the compiled graph lazily (only once).
_compiled_graph = None


def _get_compiled_graph():
    """Get or build the compiled LangGraph application."""

    global _compiled_graph
    if _compiled_graph is None:
        graph = _build_graph()
        # compile() returns a callable "app" we can run with an initial state.
        _compiled_graph = graph.compile()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def query(question: str) -> str:
    """High-level API for the rest of the backend.

    Usage:
        from backend.agent import query
        answer = query("What is GraphRAG and how does it use Neo4j?")

    Under the hood:
    - Builds an AgentState with the question.
    - Runs the LangGraph:
        OrchestratorAgent -> SearchAgent -> SummarizerAgent.
    - Returns the final answer string.
    """

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    initial_state = AgentState(question=question.strip())

    app = _get_compiled_graph()

    # Run the graph. The compiled app returns the final state as a dict.
    final_state_dict = app.invoke(initial_state.to_dict())
    final_state = AgentState.from_dict(final_state_dict)

    return final_state.answer


if __name__ == "__main__":
    # Small manual test:
    # NOTE: This assumes you have already:
    # - Ingested your vault (so FAISS + Neo4j are populated).
    # - Started Ollama and pulled the LLM model.
    test_q = "How shuold I handle my dreams about Rebi?"
    print("Question:", test_q)
    print("\nRunning agent...\n")
    print(query(test_q))
