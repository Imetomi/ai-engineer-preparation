# GraphRAG Practice Project

## Tech Stack

**Core:**
- Neo4j (graph database)
- FAISS (vector search)
- Ollama (llama3.2:1b for LLM)
- sentence-transformers (embeddings)
- FastAPI (backend)
- Streamlit (frontend)

**Python Libraries:**
```
neo4j
faiss-cpu
sentence-transformers
fastapi
uvicorn
streamlit
python-multipart
```

## Project Structure

```
graphrag-project/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── agent.py             # Query agent
│   ├── ingest.py            # Obsidian parser + Neo4j + FAISS
│   ├── neo4j_client.py      # Neo4j connection
│   └── vector_store.py      # FAISS operations
├── frontend/
│   └── app.py               # Streamlit UI
├── data/
│   └── obsidian_vault/      # Your test notes
├── storage/
│   └── faiss.index          # Generated FAISS index
├── pytproject.toml
└── README.md
```

## Getting Started

### 1. Setup Neo4j
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```
Access: http://localhost:7474 (user: neo4j, pass: password)

### 2. Setup Ollama
```bash
ollama pull qwen3-vl:2b-instruct # we will use this model! already downloaded
ollama pull toshk0/nomic-embed-text-v2-moe:Q6_K  # we will use this model! already downloaded
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Order

**Step 1: `backend/neo4j_client.py`**
- Create Neo4j driver connection
- Write helper functions: `create_note()`, `create_link()`, `get_neighbors()`

**Step 2: `backend/vector_store.py`**
- Initialize sentence-transformers model
- Create FAISS index
- Functions: `add_embeddings()`, `search()`

**Step 3: `backend/ingest.py`**
- Parse `.md` files from folder
- Extract `[[wikilinks]]` with regex
- Insert into Neo4j (nodes + relationships)
- Chunk text, embed, add to FAISS
- Map FAISS indices to note titles

**Step 4: `backend/agent.py`**
- `query(question)` function:
  - Embed question → FAISS search → get top-k chunks
  - Get note titles from chunks → Neo4j Cypher for neighbors
  - Combine context → send to Ollama
  - Return answer

**Step 5: `backend/main.py`**
- FastAPI endpoints:
  - `POST /ingest` - accepts folder path, runs ingestion "/Users/imetomi/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsinotes/1 - Atomic Notes"
  - `POST /query` - accepts question, returns answer

**Step 6: `frontend/app.py`**
- Streamlit UI:
  - Text input for vault folder path
  - Button to trigger ingestion
  - Text input for questions
  - Display answers

### 5. Run

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
streamlit run app.py
```

## Key Implementation Details

**Neo4j Schema:**
```cypher
(Note {title: str, content: str})-[:LINKS_TO]->(Note)
```

**FAISS Mapping:**
Store separate list/dict mapping FAISS index → (note_title, chunk_text)

**Obsidian Parsing:**
- Regex: `\[\[([^\]]+)\]\]` for wikilinks
- Split on headers or fixed chunk size

**Agent Logic:**
1. Vector search for semantic similarity
2. Graph traversal for connected context
3. Merge both → LLM prompt

Start with `neo4j_client.py` and test the connection.