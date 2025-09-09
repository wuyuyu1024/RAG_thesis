"""Hybrid search combining embedding similarity and keyword search (BM25)."""

import math
import chromadb
from collections import Counter
from typing import List, Dict
from .reranking import rerank_cross_encoder
from utils.config import DEBUG


def compute_bm25_score(query_terms: List[str], document: str, doc_freq: Dict[str, int], 
                      total_docs: int, k1: float = 1.5, b: float = 0.75, 
                      avg_doc_len: float = 100) -> float:
    """Compute BM25 score for a document given query terms."""
    doc_terms = document.lower().split()
    doc_len = len(doc_terms)
    term_freq = Counter(doc_terms)
    
    score = 0.0
    for term in query_terms:
        if term in term_freq:
            tf = term_freq[term]
            df = doc_freq.get(term, 1)  # Default to 1 to avoid division by zero
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    
    return score


def keyword_search(query: str, documents: List[str], n_results: int) -> List[tuple]:
    """Perform keyword search using BM25 scoring."""
    query_terms = [term.lower().strip() for term in query.split()]
    
    # Calculate document frequencies
    doc_freq = Counter()
    total_docs = len(documents)
    avg_doc_len = sum(len(doc.split()) for doc in documents) / total_docs
    
    for doc in documents:
        unique_terms = set(doc.lower().split())
        for term in unique_terms:
            doc_freq[term] += 1
    
    # Calculate BM25 scores for all documents
    doc_scores = []
    for i, doc in enumerate(documents):
        score = compute_bm25_score(query_terms, doc, doc_freq, total_docs, avg_doc_len=avg_doc_len)
        doc_scores.append((i, doc, score))
    
    # Sort by score and return top results
    doc_scores.sort(key=lambda x: x[2], reverse=True)
    return doc_scores[:n_results]


def hybrid_search(query: str, collection: chromadb.Collection, n_results: int, 
                 embedding_weight: float = 0.7) -> List[str]:
    """Combine embedding similarity and keyword search with configurable weights."""
    assert 0.0 <= embedding_weight <= 1.0, "embedding_weight must be between 0.0 and 1.0"
    keyword_weight = 1.0 - embedding_weight
    
    # Get more results initially to have a larger pool for hybrid scoring
    initial_results = min(n_results * 5, 50)  # Cap at 50 to avoid too many results
    
    # Get embedding-based results with scores
    embedding_results = collection.query(
        query_texts=[query], 
        n_results=initial_results,
        include=['documents', 'distances']
    )
    
    documents = embedding_results['documents'][0]
    distances = embedding_results['distances'][0]
    
    # Convert distances to similarity scores (assuming cosine distance)
    embedding_scores = [1 / (1 + dist) for dist in distances]
    
    # Get keyword search results
    keyword_results = keyword_search(query, documents, len(documents))
    
    # Normalize scores to [0, 1] range
    if embedding_scores:
        max_embed_score = max(embedding_scores)
        min_embed_score = min(embedding_scores)
        if max_embed_score != min_embed_score:
            embedding_scores = [(score - min_embed_score) / (max_embed_score - min_embed_score) 
                              for score in embedding_scores]
    
    keyword_scores = [result[2] for result in keyword_results]
    if keyword_scores:
        max_keyword_score = max(keyword_scores)
        min_keyword_score = min(keyword_scores)
        if max_keyword_score != min_keyword_score:
            keyword_scores = [(score - min_keyword_score) / (max_keyword_score - min_keyword_score) 
                            for score in keyword_scores]
    
    # Combine scores
    combined_results = []
    for i, doc in enumerate(documents):
        embed_score = embedding_scores[i] if i < len(embedding_scores) else 0.0
        
        # Find corresponding keyword score
        keyword_score = 0.0
        for j, (idx, _, kw_score) in enumerate(keyword_results):
            if idx == i:
                keyword_score = keyword_scores[j] if j < len(keyword_scores) else 0.0
                break
        
        # Combine scores with weights
        final_score = embedding_weight * embed_score + keyword_weight * keyword_score
        combined_results.append((doc, final_score))
    
    # Sort by combined score and return top results
    combined_results.sort(key=lambda x: x[1], reverse=True)
    
    if DEBUG:
        print(f"Hybrid search - Embedding weight: {embedding_weight}, Keyword weight: {keyword_weight}")
        print("Top 3 combined scores:")
        for i, (doc, score) in enumerate(combined_results[:3]):
            print(f"  {i+1}. Score: {score:.3f}, Doc preview: {doc[:100]}...")
    
    return [doc for doc, _ in combined_results[:n_results]]


def query_db(query: str, n_results: int, collection: chromadb.Collection, 
             embedding_weight: float = 0.7, use_reranking: bool = True, 
             rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """Main query function with hybrid search and cross-encoder reranking (default on)."""
    assert n_results > 0, "Number of results must be greater than 0"
    assert isinstance(query, str), "Query must be a string"
    assert 0.0 <= embedding_weight <= 1.0, "embedding_weight must be between 0.0 and 1.0"

    # Stage 1: Hybrid search (embedding + keyword)
    # Always get 5x results for reranking pool
    initial_n_results = n_results * 5
    results = hybrid_search(query, collection, initial_n_results, embedding_weight)
    
    if DEBUG:
        print('Hybrid search results:', len(results), 'documents')
    
    # Stage 2: Cross-encoder reranking (default enabled)
    if use_reranking and len(results) > 1:
        if DEBUG:
            print(f"Applying cross-encoder reranking with model: {rerank_model}")
        reranked_results = rerank_cross_encoder(query, results, rerank_model)
        # Take only the top n_results after reranking
        final_results = reranked_results[:n_results]
    else:
        # If reranking disabled, just take top n_results from hybrid search
        final_results = results[:n_results]
    
    if DEBUG:
        print(f'Final results: {len(final_results)} documents\n')
    
    return final_results