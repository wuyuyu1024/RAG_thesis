"""
Main application for RAG (Retrieval-Augmented Generation) system.

This application provides a conversational interface for querying PhD thesis content
using hybrid search (embedding + keyword) with optional cross-encoder reranking.
"""

from models import DialogMemory
from search import build_db
from generation import generate_answer_with_citation
from utils.config import DEBUG


def main():
    """Main application entry point."""
    collection = build_db()
    memory = DialogMemory(max_turns=10)
    
    print("Database build completed.")
    print("Dialog memory initialized. The system will remember the last 10 conversation turns.")
    print("Hybrid search enabled: combines embedding similarity and keyword search (BM25)")
    print("Embedding weight controls the balance (1.0 = pure embedding, 0.0 = pure keyword)")
    print("Optional cross-encoder reranking available for improved precision")
    
    while query := input("Enter your query (or 'exit' to quit): ").strip():
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
            
        # Get user input for search parameters
        n_results = int(input("Enter number of results to return: "))
        
        embedding_weight_input = input("Enter embedding weight (0.0-1.0, default 0.7): ").strip()
        embedding_weight = 0.7  # default
        if embedding_weight_input:
            try:
                embedding_weight = float(embedding_weight_input)
                if not 0.0 <= embedding_weight <= 1.0:
                    print("Invalid weight, using default 0.7")
                    embedding_weight = 0.7
            except ValueError:
                print("Invalid weight format, using default 0.7")
        
        # Ask about reranking
        reranking_input = input("Enable cross-encoder reranking? (y/n, default n): ").strip().lower()
        use_reranking = reranking_input in ['y', 'yes', '1', 'true']
        
        rerank_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # default
        if use_reranking:
            model_input = input("Cross-encoder model (default: ms-marco-MiniLM-L-6-v2): ").strip()
            if model_input:
                rerank_model = f"cross-encoder/{model_input}" if not model_input.startswith('cross-encoder/') else model_input
        
        # Generate answer using the modular pipeline
        answer = generate_answer_with_citation(
            query, collection, n_results, memory, 
            embedding_weight, use_reranking, rerank_model
        )
        print(f"Answer: {answer}")
        
        # Show memory status
        if DEBUG and memory.has_history():
            print(f"[DEBUG] Memory contains {len(memory.turns)} conversation turns.")


if __name__ == "__main__":
    main()