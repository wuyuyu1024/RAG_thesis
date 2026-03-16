# RAG Thesis System

> A conversational interface for a long thesis and the kind friends who would rather ask questions than read the whole PDF.

This project is a retrieval-augmented generation (RAG) app built around my PhD thesis at Utrecht University. It turns the thesis into something you can query through a browser UI or a simple command-line interface, using an in-memory ChromaDB collection, hybrid retrieval, optional cross-encoder reranking, and Gemini-generated answers.

The source material is my Utrecht University PhD thesis: [thesis PDF](https://research-portal.uu.nl/ws/portalfiles/portal/266197286/thesis%20-%20684ee48bb78b1.pdf).

This started after some friends basically told me, "I support you, but I am not reading all of that." Fair enough. So I gave the thesis a chat interface.

## Why This Exists

- The thesis is real.
- The PDF is long.
- `Ctrl+F` is helpful, but it has no patience for follow-up questions.

## Features

- FastAPI backend with a browser-based chat interface
- Hybrid retrieval that blends embedding search and keyword scoring
- Optional cross-encoder reranking for cleaner result ordering
- Citation lookup from bibliography entries when the query calls for references
- Conversation memory that keeps the last 10 turns
- CLI entrypoint for local interactive use, if the terminal is your preferred habitat

## Requirements

- Python 3.13 or newer
- `uv`
- A `GEMINI_API_KEY`
- Internet access for Gemini API calls and, on first use, any model downloads triggered by dependencies

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Create an environment file:

```bash
cp .env.example .env
```

3. Edit `.env` and set your Gemini key:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

4. Start the web app:

```bash
uv run python run_web.py
```

5. Open `http://localhost:8000`.

## Run Modes

### Web app

Recommended:

```bash
uv run python run_web.py
```

Development with auto-reload:

```bash
uv run uvicorn app:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

### CLI

```bash
uv run python src/main.py
```

The CLI prompts for:

- query text
- number of results
- embedding weight
- whether reranking is enabled
- which cross-encoder model to use

## API

### Endpoints

- `GET /`: serves the web interface
- `POST /query`: runs retrieval and answer generation
- `GET /health`: reports whether the app and database are initialized
- `GET /memory/status`: returns conversation-memory metadata
- `GET /memory/history`: returns the stored conversation turns
- `POST /memory/clear`: clears the current conversation memory

### `POST /query` body

```json
{
  "query": "What are decision maps?",
  "n_results": 5,
  "embedding_weight": 0.7,
  "use_reranking": true,
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### Example request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are decision maps?",
    "n_results": 5,
    "embedding_weight": 0.7,
    "use_reranking": true,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
  }'
```

### Response shape

```json
{
  "answer": "Generated answer text",
  "query": "What are decision maps?",
  "memory_turns": 1
}
```

## How It Works

1. On startup, the app reads [`data/chunks.txt`](./data/chunks.txt) and loads the text chunks into a ChromaDB collection.
2. For each query, it retrieves a candidate set using embedding similarity and BM25-style keyword scoring.
3. If enabled, it reranks the retrieved chunks with a cross-encoder.
4. It asks Gemini to produce the final answer from the retrieved chunks and current conversation context.
5. If the query appears to need references, it extracts citation keys from retrieved LaTeX snippets and resolves them through [`data/bib_entries.json`](./data/bib_entries.json).
6. The resulting turn is stored in a rolling 10-turn conversation memory.

## Project Structure

```text
.
|- data/
|  |- bib_entries.json
|  `- chunks.txt
|- src/
|  |- app.py
|  |- main.py
|  |- generation/
|  |  |- answer_generator.py
|  |  `- citation_handler.py
|  |- models/
|  |  `- memory.py
|  |- search/
|  |  |- database.py
|  |  |- hybrid_search.py
|  |  `- reranking.py
|  `- utils/
|     `- config.py
|- static/
|  `- index.html
|- pyproject.toml
|- run_web.py
`- uv.lock
```

## Key Files

- [`run_web.py`](./run_web.py): small startup wrapper for the FastAPI app
- [`src/app.py`](./src/app.py): API routes, startup lifecycle, and frontend serving
- [`src/main.py`](./src/main.py): CLI entrypoint
- [`src/search/database.py`](./src/search/database.py): ChromaDB collection creation from thesis chunks
- [`src/search/hybrid_search.py`](./src/search/hybrid_search.py): hybrid retrieval and reranking orchestration
- [`src/generation/answer_generator.py`](./src/generation/answer_generator.py): Gemini answer generation
- [`src/generation/citation_handler.py`](./src/generation/citation_handler.py): citation detection and bibliography lookup
- [`static/index.html`](./static/index.html): browser chat UI

## Operational Notes

- The Chroma collection is rebuilt on every startup and is not persisted between runs.
- The app raises an error during import if `GEMINI_API_KEY` is missing.
- `embedding_weight` must stay between `0.0` and `1.0`.
- Debug logging is currently enabled in [`src/utils/config.py`](./src/utils/config.py).
- The current citation extraction logic looks for LaTeX `\citep{...}` and `\citeyear{...}` patterns in retrieved chunks.

## Known Gaps

- There is no automated test suite in the repo.
- There is no `.env` committed by design, so the app is not runnable until the environment variable is set.
- The web UI and API are functional, but this is still a research-style codebase rather than a production-hardened service.
