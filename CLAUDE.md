# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system for a PhD thesis about decision maps and machine learning classification models. The system uses ChromaDB for vector storage, Google Gemini AI for answer generation, and cross-encoder re-ranking for improved retrieval quality.

## Architecture

### Core Components

- **Main Application** (`src/main.py`): Single-file application containing the entire RAG pipeline
- **Data Storage**: 
  - `data/chunks.txt`: LaTeX document chunks from the PhD thesis
  - `data/bib_entries.json`: Bibliography entries in JSON format for citation lookup
- **Vector Database**: ChromaDB collection for semantic search over thesis content

### RAG Pipeline Flow

1. **Database Construction** (`build_db`): Loads thesis chunks into ChromaDB collection
2. **Query Processing** (`query_db`): Retrieves relevant chunks using semantic search + cross-encoder re-ranking
3. **Citation Detection** (`check_reference`): Uses Gemini AI to determine if citations are needed
4. **Citation Extraction** (`find_citation`): Extracts citation keys from retrieved content
5. **Answer Generation** (`generate_answer_single` / `generate_answer_with_citation`): Generates responses with or without citations

## Dependencies

- `chromadb>=1.0.15`: Vector database for semantic search
- `google-genai>=1.28.0`: Google Gemini AI integration
- `sentence-transformers>=5.1.0`: Cross-encoder for result re-ranking
- `dotenv>=0.9.9`: Environment variable management

## Environment Setup

Required environment variable:
- `GEMINI_API_KEY`: Google Gemini API key (stored in `.env`)

## Running the Application

The application runs interactively from the command line:

```bash
python src/main.py
```

The system will:
1. Build the ChromaDB database from `data/chunks.txt`
2. Enter interactive mode for query processing
3. For each query, ask for number of results to return
4. Generate answers with citations when appropriate

## Development Notes

- The system uses cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve retrieval quality
- Citation detection uses a binary classification approach with Gemini AI
- Bibliography lookup uses exact key matching against `data/bib_entries.json`
- Debug mode is enabled by default (`DEBUG = True`)