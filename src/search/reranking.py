"""Cross-encoder reranking functionality with caching and error handling."""

from sentence_transformers import CrossEncoder
from typing import List
from utils.config import DEBUG


class CrossEncoderReranker:
    """Singleton class for managing cross-encoder models with caching and error handling."""
    _instances = {}
    
    def __new__(cls, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        if model_name not in cls._instances:
            cls._instances[model_name] = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instances[model_name]._initialized = False
        return cls._instances[model_name]
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        if self._initialized:
            return
            
        self.model_name = model_name
        self.model = None
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """Load the cross-encoder model with error handling."""
        try:
            if DEBUG:
                print(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            if DEBUG:
                print("Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"Error loading cross-encoder model {self.model_name}: {e}")
            print("Falling back to no reranking")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.model is not None
    
    def predict_batch(self, query_doc_pairs: list) -> list:
        """Predict scores for a batch of query-document pairs."""
        if not self.is_available():
            return [0.0] * len(query_doc_pairs)
        
        try:
            scores = self.model.predict(query_doc_pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            print(f"Error during cross-encoder prediction: {e}")
            return [0.0] * len(query_doc_pairs)


def rerank_cross_encoder(query: str, results: List[str], 
                        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> List[str]:
    """Rerank results using cross-encoder with improved error handling and caching."""
    if not results:
        return results
    
    reranker = CrossEncoderReranker(model_name)
    
    if not reranker.is_available():
        if DEBUG:
            print("Cross-encoder not available, returning original order")
        return results
    
    # Create query-document pairs
    query_doc_pairs = [(query, doc) for doc in results]
    
    # Get scores in batch
    scores = reranker.predict_batch(query_doc_pairs)
    
    # Sort by scores in descending order
    ranked_results = [doc for _, doc in sorted(zip(scores, results), reverse=True)]
    
    if DEBUG:
        print(f"Reranked {len(results)} results using {model_name}")
        print("Top 3 reranking scores:")
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i, (idx, score) in enumerate(sorted_scores[:3]):
            print(f"  {i+1}. Score: {score:.3f}, Doc preview: {results[idx][:100]}...")
    
    return ranked_results