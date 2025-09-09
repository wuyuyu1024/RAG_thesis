"""
FastAPI backend for RAG system.

Provides REST API endpoints for querying PhD thesis content
using hybrid search with optional cross-encoder reranking.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os

from models import DialogMemory
from search import build_db
from generation import generate_answer_with_citation


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    embedding_weight: float = 0.7
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class QueryResponse(BaseModel):
    answer: str
    query: str
    memory_turns: int


class ConversationTurn(BaseModel):
    query: str
    answer: str
    timestamp: str


class ConversationHistory(BaseModel):
    turns: list[ConversationTurn]
    total_turns: int


# Global variables for database and memory
collection = None
memory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and memory on startup."""
    global collection, memory
    collection = build_db()
    memory = DialogMemory(max_turns=10)
    print("Database build completed.")
    print("Dialog memory initialized. The system will remember the last 10 conversation turns.")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="RAG Thesis API",
    description="RESTful API for RAG system querying PhD thesis content",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files (for frontend)
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the HTML frontend."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # Simple fallback HTML if static file doesn't exist
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Thesis Query System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select, button { width: 100%; padding: 8px; box-sizing: border-box; }
            textarea { height: 100px; resize: vertical; }
            button { background-color: #007bff; color: white; border: none; padding: 12px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .loading { color: #6c757d; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>RAG Thesis Query System</h1>
        <form id="queryForm">
            <div class="form-group">
                <label for="query">Query:</label>
                <textarea id="query" name="query" placeholder="Enter your query here..." required></textarea>
            </div>
            <div class="form-group">
                <label for="n_results">Number of Results:</label>
                <input type="number" id="n_results" name="n_results" value="5" min="1" max="20">
            </div>
            <div class="form-group">
                <label for="embedding_weight">Embedding Weight (0.0-1.0):</label>
                <input type="number" id="embedding_weight" name="embedding_weight" value="0.7" min="0" max="1" step="0.1">
            </div>
            <div class="form-group">
                <label for="use_reranking">Enable Cross-encoder Reranking:</label>
                <select id="use_reranking" name="use_reranking">
                    <option value="true" selected>Yes</option>
                    <option value="false">No</option>
                </select>
            </div>
            <div class="form-group">
                <label for="rerank_model">Rerank Model:</label>
                <input type="text" id="rerank_model" name="rerank_model" value="cross-encoder/ms-marco-MiniLM-L-6-v2">
            </div>
            <button type="submit">Query</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('queryForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div class="loading">Processing your query...</div>';
                
                const formData = new FormData(e.target);
                const data = {
                    query: formData.get('query'),
                    n_results: parseInt(formData.get('n_results')),
                    embedding_weight: parseFloat(formData.get('embedding_weight')),
                    use_reranking: formData.get('use_reranking') === 'true',
                    rerank_model: formData.get('rerank_model')
                };
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    resultDiv.innerHTML = `
                        <h3>Answer:</h3>
                        <p>${result.answer}</p>
                        <small>Memory turns: ${result.memory_turns}</small>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query and return the generated answer."""
    global collection, memory
    
    if collection is None or memory is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Validate embedding weight
        if not 0.0 <= request.embedding_weight <= 1.0:
            raise HTTPException(status_code=400, detail="Embedding weight must be between 0.0 and 1.0")
        
        # Generate answer using the modular pipeline
        answer = generate_answer_with_citation(
            request.query,
            collection,
            request.n_results,
            memory,
            request.embedding_weight,
            request.use_reranking,
            request.rerank_model
        )
        
        return QueryResponse(
            answer=answer,
            query=request.query,
            memory_turns=len(memory.turns) if memory.has_history() else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "database_initialized": collection is not None}


@app.get("/memory/status")
async def memory_status():
    """Get memory status information."""
    global memory
    if memory is None:
        raise HTTPException(status_code=500, detail="Memory not initialized")
    
    return {
        "has_history": memory.has_history(),
        "turns_count": len(memory.turns) if memory.has_history() else 0,
        "max_turns": memory.max_turns
    }


@app.get("/memory/history", response_model=ConversationHistory)
async def get_conversation_history():
    """Get the full conversation history."""
    global memory
    if memory is None:
        raise HTTPException(status_code=500, detail="Memory not initialized")
    
    if not memory.has_history():
        return ConversationHistory(turns=[], total_turns=0)
    
    turns = []
    for turn in memory.turns:
        turns.append(ConversationTurn(
            query=turn.query,
            answer=turn.answer, 
            timestamp=turn.timestamp or ""
        ))
    
    return ConversationHistory(turns=turns, total_turns=len(turns))


@app.post("/memory/clear")
async def clear_memory():
    """Clear conversation memory."""
    global memory
    if memory is None:
        raise HTTPException(status_code=500, detail="Memory not initialized")
    
    memory.clear()
    return {"message": "Memory cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)