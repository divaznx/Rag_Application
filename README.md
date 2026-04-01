# RAG System — OpenAI + ChromaDB

Two-program local RAG system.  
`kb_manager.py` ingests folders → `rag_query.py` answers questions.

---

## Setup

```bash
pip install chromadb openai python-dotenv
```

Create a `.env` file in the project folder:
```
OPENAI_API_KEY=sk-...
```

---

## Folder Structure

```
your_project/
├── kb_manager.py       ← Program 1: ingest & manage knowledge base
├── rag_query.py        ← Program 2: query & get answers
├── embedder.py         ← Shared: OpenAI text-embedding-3-small
├── .env                ← Your OpenAI API key (never commit this)
│
└── knowledge_base/     ← Auto-created on first ingest
    ├── processed/      ← ChromaDB vector store
    └── registry.json   ← Tracks which folders are already indexed
```

---

## kb_manager.py — Commands

```bash
# Ingest a folder (sub-folders = separate document groups)
python kb_manager.py --ingest ./my_docs

# Re-ingest same folder → already-indexed ones are skipped automatically
python kb_manager.py --ingest ./my_docs

# Force re-index everything (e.g. after changing chunk size)
python kb_manager.py --ingest ./my_docs --force

# Check what's indexed
python kb_manager.py --status

# Remove a folder from the index
python kb_manager.py --delete folder_name
```

---

## rag_query.py — Commands

```bash
# Interactive shell
python rag_query.py

# Single question
python rag_query.py --query "What is gradient descent?"

# With options
python rag_query.py --query "Explain asyncio" --folder python_notes --show-sources
python rag_query.py --query "Summarise the docs" --model gpt-4o --top-k 8
```

### Interactive shell commands

| Command            | What it does                        |
|--------------------|-------------------------------------|
| `/folder ml_notes` | Search only that folder             |
| `/folder`          | Clear filter, search everything     |
| `/sources on`      | Show retrieved chunks with answers  |
| `/model gpt-4o`    | Switch to a different OpenAI model  |
| `/topk 8`          | Retrieve more chunks per query      |
| `/status`          | List all indexed folders            |
| `/quit`            | Exit                                |

---

## Models used

| Purpose    | Model                    | Notes                        |
|------------|--------------------------|------------------------------|
| Embeddings | text-embedding-3-small   | Fast, cheap, 1536 dims       |
| Generation | gpt-4o-mini (default)    | Change with --model flag     |
| Generation | gpt-4o                   | Better quality, higher cost  |

---

## Configuration (top of each file)

**kb_manager.py**
```python
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks
```

**rag_query.py**
```python
DEFAULT_TOP_K     = 5       # chunks retrieved per query
DEFAULT_MODEL     = "gpt-4o-mini"
SIMILARITY_CUTOFF = 1.4     # cosine distance filter (lower = stricter)
```

**embedder.py**
```python
EMBEDDING_MODEL = "text-embedding-3-small"   # swap to text-embedding-3-large if needed
```