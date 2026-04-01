"""
rag_query.py  —  RAG Query Engine (OpenAI)
============================================
Query your knowledge base using ChromaDB for retrieval and
OpenAI (gpt-4o-mini) for generation.

Set your API key in a .env file or environment variable:
    OPENAI_API_KEY=sk-...

Usage (interactive mode):
    python rag_query.py

Usage (single question):
    python rag_query.py --query "What is the refund policy?"

Options:
    --top-k N        Number of chunks to retrieve (default: 5)
    --model NAME     OpenAI model to use (default: gpt-4o-mini)
    --show-sources   Print source chunks alongside the answer
    --folder NAME    Restrict retrieval to a specific ingested folder
"""

import os
import sys
import json
import argparse
import textwrap
from pathlib import Path

import chromadb
from openai import OpenAI
from embedder import get_embedding_function

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
KB_DIR          = Path("knowledge_base")
PROCESSED_DIR   = KB_DIR / "processed"
REGISTRY_FILE   = KB_DIR / "registry.json"
COLLECTION_NAME = "rag_collection"

DEFAULT_TOP_K     = 5
DEFAULT_MODEL     = "gpt-4o-mini"
SIMILARITY_CUTOFF = 1.4   # cosine distance (lower = more similar)


# ─────────────────────────────────────────────
#  COLOUR HELPERS
# ─────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"

def banner(text):
    line = "─" * 60
    print(f"\n{C.BOLD}{C.CYAN}{line}\n  {text}\n{line}{C.RESET}\n")

def ok(msg):   print(f"  {C.GREEN}✔  {C.RESET}{msg}")
def warn(msg): print(f"  {C.YELLOW}⚠  {C.RESET}{msg}")
def err(msg):  print(f"  {C.RED}✘  {C.RESET}{msg}")
def info(msg): print(f"  {C.BLUE}ℹ  {C.RESET}{msg}")


# ─────────────────────────────────────────────
#  OPENAI CLIENT
# ─────────────────────────────────────────────
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        err("OPENAI_API_KEY not set.")
        print("  Add it to a .env file:  OPENAI_API_KEY=sk-...")
        sys.exit(1)
    return OpenAI(api_key=api_key)


# ─────────────────────────────────────────────
#  CHROMADB
# ─────────────────────────────────────────────
def get_collection():
    if not PROCESSED_DIR.exists():
        err("Knowledge base not found. Run:  python kb_manager.py --ingest <folder>")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(PROCESSED_DIR))
    ef = get_embedding_function()
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        err("Collection not found. Ingest documents first with kb_manager.py.")
        sys.exit(1)


# ─────────────────────────────────────────────
#  REGISTRY
# ─────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
#  RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(query: str, collection, top_k: int, folder_filter: str | None) -> list[dict]:
    where  = {"folder": folder_filter} if folder_filter else None
    kwargs = dict(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results   = collection.query(**kwargs)
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text":     doc,
            "folder":   meta.get("folder", "?"),
            "file":     meta.get("file", "?"),
            "chunk_id": meta.get("chunk_id", 0),
            "distance": round(dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, distances)
        if dist <= SIMILARITY_CUTOFF
    ]


# ─────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────
def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    system = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context. If the context doesn't contain enough information, "
        "say \"I don't have enough information in the knowledge base to answer this.\" "
        "Do not make things up. Be concise and precise."
    )

    if not chunks:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Question: {query}"},
        ]

    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = f"{c['folder']}/{c['file']}"
        context_parts.append(f"[Chunk {i} | Source: {source}]\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)

    user_msg = f"""Use the context below to answer the question.

=== CONTEXT ===
{context}
=== END CONTEXT ===

Question: {query}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]


# ─────────────────────────────────────────────
#  OPENAI GENERATION
# ─────────────────────────────────────────────
def call_openai(messages: list[dict], model: str, openai_client: OpenAI) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error: {e}]"


# ─────────────────────────────────────────────
#  PRINT SOURCES
# ─────────────────────────────────────────────
def print_sources(chunks: list[dict]):
    print(f"\n{C.BOLD}{C.BLUE}{'─'*60}\n  Retrieved Sources\n{'─'*60}{C.RESET}")
    for i, c in enumerate(chunks, 1):
        print(f"\n  {C.CYAN}[{i}] {c['folder']}/{c['file']}  "
              f"{C.DIM}(chunk {c['chunk_id']}, distance={c['distance']}){C.RESET}")
        wrapped = textwrap.fill(
            c['text'][:400], width=72,
            initial_indent="      ", subsequent_indent="      "
        )
        print(f"{C.DIM}{wrapped}{'…' if len(c['text']) > 400 else ''}{C.RESET}")


# ─────────────────────────────────────────────
#  SINGLE QUERY HANDLER
# ─────────────────────────────────────────────
def answer_query(query: str, collection, top_k: int, model: str,
                 show_sources: bool, folder_filter: str | None,
                 openai_client: OpenAI):

    print(f"\n{C.BOLD}Query:{C.RESET} {query}")
    print(f"{C.DIM}Searching knowledge base…{C.RESET}")

    chunks = retrieve(query, collection, top_k, folder_filter)

    if not chunks:
        warn("No sufficiently relevant chunks found in the knowledge base.")
        print(f"\n  {C.YELLOW}Tip:{C.RESET} Try rephrasing, or check "
              f"'python kb_manager.py --status' to see what's indexed.\n")
        return

    info(f"Found {len(chunks)} relevant chunk(s). Generating answer with {model}…")

    if show_sources:
        print_sources(chunks)

    messages = build_messages(query, chunks)
    response = call_openai(messages, model, openai_client)

    print(f"\n{C.BOLD}{C.GREEN}{'─'*60}\n  Answer\n{'─'*60}{C.RESET}\n")
    wrapped = textwrap.fill(response, width=72,
                            initial_indent="  ", subsequent_indent="  ")
    print(f"{C.BOLD}{wrapped}{C.RESET}\n")

    if not show_sources:
        sources = list({f"{c['folder']}/{c['file']}" for c in chunks})
        print(f"{C.DIM}  Sources: {', '.join(sources)}{C.RESET}\n")


# ─────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────
HELP_TEXT = """
Commands:
  <question>          — Ask anything
  /status             — Show indexed folders
  /sources on|off     — Toggle showing source chunks
  /folder <name>      — Restrict search to a folder
  /folder             — Clear folder filter (search all)
  /model <name>       — Switch OpenAI model  (e.g. /model gpt-4o)
  /topk <n>           — Change number of chunks retrieved
  /help               — Show this menu
  /quit               — Exit
"""

def interactive_loop(collection, registry: dict, top_k: int,
                     model: str, show_sources: bool, openai_client: OpenAI):
    folder_filter = None

    banner(f"RAG Query Engine  |  model: {model}  |  top-k: {top_k}")
    info(f"Knowledge base: {collection.count()} chunk(s) across {len(registry)} folder(s).")
    print(f"  Type {C.CYAN}/help{C.RESET} for commands or just ask a question.\n")

    while True:
        try:
            label = f"{C.BOLD}{C.CYAN}[{folder_filter or 'all'}]> {C.RESET}"
            line  = input(label).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Bye!{C.RESET}\n")
            break

        if not line:
            continue

        if line in ("/quit", "/exit", "quit", "exit"):
            print(f"\n{C.DIM}Bye!{C.RESET}\n")
            break

        elif line == "/help":
            print(HELP_TEXT)

        elif line == "/status":
            if not registry:
                warn("Nothing indexed yet.")
            else:
                print(f"\n  {'Folder':<30} {'Files':>6} {'Chunks':>8}  Indexed At")
                print(f"  {'─'*30} {'─'*6} {'─'*8}  {'─'*19}")
                for name, meta in registry.items():
                    print(f"  {C.CYAN}{name:<30}{C.RESET} "
                          f"{meta['files']:>6} {meta['chunks']:>8}  "
                          f"{C.DIM}{meta['indexed_at']}{C.RESET}")
                print()

        elif line.startswith("/sources"):
            parts = line.split()
            if len(parts) == 2:
                show_sources = parts[1] == "on"
                ok(f"Source display {'ON' if show_sources else 'OFF'}.")
            else:
                print(f"  Current: {'ON' if show_sources else 'OFF'}. Use /sources on|off")

        elif line.startswith("/folder"):
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                folder_filter = None
                info("Folder filter cleared — searching all folders.")
            else:
                folder_filter = parts[1].strip()
                info(f"Now searching only: '{folder_filter}'")

        elif line.startswith("/model"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                model = parts[1].strip()
                ok(f"Model switched to: {model}")
            else:
                print(f"  Current model: {model}")

        elif line.startswith("/topk"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                ok(f"top-k set to {top_k}")
            else:
                print(f"  Current top-k: {top_k}. Use /topk <number>")

        else:
            answer_query(line, collection, top_k, model,
                         show_sources, folder_filter, openai_client)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG Query Engine — ask questions from your knowledge base"
    )
    parser.add_argument("--query",        metavar="QUESTION",
                        help="Single question (non-interactive)")
    parser.add_argument("--top-k",        type=int, default=DEFAULT_TOP_K,
                        help=f"Chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--model",        default=DEFAULT_MODEL,
                        help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--show-sources", action="store_true",
                        help="Print retrieved chunks with the answer")
    parser.add_argument("--folder",       metavar="NAME",
                        help="Restrict retrieval to a specific folder")
    args = parser.parse_args()

    openai_client = get_openai_client()
    collection    = get_collection()
    registry      = load_registry()

    if collection.count() == 0:
        warn("Knowledge base is empty. Ingest documents first:")
        print("  python kb_manager.py --ingest <folder>\n")

    if args.query:
        answer_query(args.query, collection, args.top_k, args.model,
                     args.show_sources, args.folder, openai_client)
    else:
        interactive_loop(collection, registry, args.top_k,
                         args.model, args.show_sources, openai_client)


if __name__ == "__main__":
    main()