"""
Neo4j client utilities for the GraphRAG project.

This module is responsible for:
- Creating a Neo4j driver (connection pool)
- Providing small, easy-to-understand helper functions for:
  - create_note()      -> create / upsert a Note node
  - create_link()      -> create a LINKS_TO relationship between two notes
  - get_neighbors()    -> fetch a note and its neighbors from the graph

You should start by:
1. Making sure Neo4j is running (see README).
2. Adjusting NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD if needed.
3. Running this file directly (`python -m backend.neo4j_client`) to test
   that the connection works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# By default we match the README docker command:
#   - URI: bolt://localhost:7687
#   - USER: neo4j
#   - PASSWORD: password
#
# If you change your Neo4j container credentials, update these values
# or load them from environment variables instead.

NEO4J_URI: str = "bolt://localhost:7687"
NEO4J_USER: str = "neo4j"
NEO4J_PASSWORD: str = "password"


# ---------------------------------------------------------------------------
# Data models (lightweight containers to make return types clearer)
# ---------------------------------------------------------------------------


@dataclass
class Note:
    """Represents a Note node in Neo4j.

    This mirrors the schema from the README:

    (Note {title: str, content: str})-[:LINKS_TO]->(Note)
    """

    title: str
    content: str


@dataclass
class NeighborResult:
    """Represents the main note and its neighbors returned by get_neighbors()."""

    center: Note
    neighbors: List[Note]


# ---------------------------------------------------------------------------
# Driver management
# ---------------------------------------------------------------------------

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """Get a singleton Neo4j Driver instance.

    The Driver is an object that manages a pool of connections to the
    Neo4j database. Creating it is relatively expensive, so we do it
    once and reuse it.
    """

    global _driver

    if _driver is None:
        # GraphDatabase.driver creates a connection pool but does not
        # actually open a connection until the first query is executed.
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    return _driver


def close_driver() -> None:
    """Close the global Neo4j driver if it exists.

    Call this when your application shuts down (FastAPI lifespan, etc.)
    to cleanly release network resources.
    """

    global _driver

    if _driver is not None:
        _driver.close()
        _driver = None


def _get_session() -> Session:
    """Internal helper: open a new session from the global driver.

    A session represents a logical "conversation" with the database.
    In this simple project we just:
    - open a session
    - run a query
    - close the session
    """

    return get_driver().session()


# ---------------------------------------------------------------------------
# Helper functions used by the rest of the backend
# ---------------------------------------------------------------------------


def create_note(title: str, content: str) -> Note:
    """Create or update a Note node.

    - If a Note with the same title already exists, we update its content.
    - If it does not exist, we create it.

    We use MERGE so that "title" acts like a natural key / unique identifier.
    """

    cypher = """
    MERGE (n:Note {title: $title})
    SET n.content = $content
    RETURN n.title AS title, n.content AS content
    """

    params: Dict[str, Any] = {"title": title, "content": content}

    # `with _get_session() as session` ensures the session is closed
    # even if something goes wrong.
    #
    # IMPORTANT:
    # - This query uses MERGE + SET, so it is a **write** operation.
    # - It MUST use execute_write, not execute_read, otherwise Neo4j
    #   will complain: "Writing in read access mode not allowed."
    with _get_session() as session:
        record = session.execute_write(lambda tx: tx.run(cypher, **params).single())

    if record is None:
        # This should basically never happen if the query is correct,
        # but we guard against it in case something weird happens.
        raise RuntimeError("Failed to create or retrieve Note from Neo4j.")

    return Note(title=record["title"], content=record["content"])


def create_link(source_title: str, target_title: str) -> None:
    """Create a LINKS_TO relationship between two notes.

    - If the notes do not exist yet, we create them with empty content.
      (This can happen if you parse a link before you've loaded the note.)
    - If the relationship already exists, MERGE ensures we don't create
      a duplicate relationship.
    """

    cypher = """
    MERGE (source:Note {title: $source_title})
      ON CREATE SET source.content = coalesce(source.content, "")
    MERGE (target:Note {title: $target_title})
      ON CREATE SET target.content = coalesce(target.content, "")
    MERGE (source)-[:LINKS_TO]->(target)
    """

    params: Dict[str, Any] = {
        "source_title": source_title,
        "target_title": target_title,
    }

    with _get_session() as session:
        # We don't care about the result, we just want the effect.
        session.execute_write(lambda tx: tx.run(cypher, **params))


def get_neighbors(title: str, depth: int = 1) -> Optional[NeighborResult]:
    """Return the note with the given title and its neighbors.

    Parameters
    ----------
    title:
        Title of the "center" note we want to expand around.
    depth:
        How far to traverse in the graph. For example:
        - depth = 1 -> immediate neighbors (1 hop)
        - depth = 2 -> neighbors of neighbors (2 hops), etc.

    Returns
    -------
    NeighborResult | None
        - NeighborResult(center=Note, neighbors=[Note, ...]) if the
          center note exists.
        - None if no note with the specified title exists.
    """

    # We use a variable length pattern `[:LINKS_TO*1..$depth]` to grab
    # neighbors up to the specified depth, both incoming and outgoing.
    cypher_center = """
    MATCH (n:Note {title: $title})
    RETURN n.title AS title, n.content AS content
    """

    # NOTE: Neo4j does not allow using a parameter directly in the variable
    # length pattern (e.g. [:LINKS_TO*1..$depth]), so we safely interpolate
    # the integer depth into the query string here.
    cypher_neighbors = f"""
    MATCH (n:Note {{title: $title}})
    MATCH (n)-[:LINKS_TO*1..{int(depth)}]-(neighbor:Note)
    RETURN DISTINCT neighbor.title AS title, neighbor.content AS content
    """

    params: Dict[str, Any] = {"title": title}

    with _get_session() as session:
        # First, fetch the center note.
        center_record = session.execute_read(
            lambda tx: tx.run(cypher_center, **params).single()
        )

        if center_record is None:
            # No note with this title.
            return None

        center = Note(
            title=center_record["title"],
            content=center_record["content"],
        )

        # Then, fetch neighbors up to the given depth.
        neighbor_records = session.execute_read(
            lambda tx: list(tx.run(cypher_neighbors, **params))
        )

    neighbors = [
        Note(title=rec["title"], content=rec["content"])
        for rec in neighbor_records
        # It's possible the center note appears in this result;
        # we filter it out so that `center` is not duplicated.
        if rec["title"] != center.title
    ]

    return NeighborResult(center=center, neighbors=neighbors)


# ---------------------------------------------------------------------------
# Simple manual test (run: python -m backend.neo4j_client)
# ---------------------------------------------------------------------------


def _test_connection() -> None:
    """Quick-and-dirty test to verify Neo4j connectivity.

    This will:
    - connect to Neo4j
    - create a couple of notes and a link
    - fetch neighbors
    - print results to the console
    """

    print("Testing Neo4j connection...")
    driver = get_driver()
    try:
        # Basic "ping" using a simple query.
        with driver.session() as session:
            result = session.run("RETURN 1 AS n").single()
            print("Ping result from Neo4j:", result["n"])

        print("\nCreating example notes and link...")
        note_a = create_note("Example A", "This is note A.")
        note_b = create_note("Example B", "This is note B.")
        create_link(note_a.title, note_b.title)

        print("Fetching neighbors for 'Example A'...")
        neighbors = get_neighbors("Example A", depth=1)
        if neighbors is None:
            print("No note found with title 'Example A'")
        else:
            print("Center:", neighbors.center)
            print("Neighbors:")
            for n in neighbors.neighbors:
                print(" -", n)
    finally:
        print("\nClosing driver...")
        close_driver()


if __name__ == "__main__":
    _test_connection()
