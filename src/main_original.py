import chromadb
from google import genai
from google.genai import types
import dotenv
import os
import sys
from dataclasses import dataclass
import json 
import re
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
from collections import Counter
import math

DEBUG = True

@dataclass
class DialogTurn:
    query: str
    answer: str
    retrieved_docs: List[str]
    citations: Optional[List[str]] = None

class DialogMemory:
    def __init__(self, max_turns: int = 10):
        self.turns: List[DialogTurn] = []
        self.max_turns = max_turns
    
    def add_turn(self, query: str, answer: str, retrieved_docs: List[str], citations: Optional[List[str]] = None):
        turn = DialogTurn(query, answer, retrieved_docs, citations)
        self.turns.append(turn)
        
        # Keep only the last max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]
    
    def get_context(self, last_n_turns: int = 3) -> str:
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-last_n_turns:] if len(self.turns) >= last_n_turns else self.turns
        
        context = "Previous conversation context:\n"
        for i, turn in enumerate(recent_turns, 1):
            context += f"Turn {i}:\n"
            context += f"Q: {turn.query}\n"
            context += f"A: {turn.answer}\n\n"
        
        return context
    
    def has_history(self) -> bool:
        return len(self.turns) > 0

dotenv.load_dotenv()
if GEMINI_API_KEY := os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

def build_db(filename: str = "./data/chunks.txt"):
    chromadb_client = chromadb.Client()
    print("ChromaDB client initialized successfully.")

    collection = chromadb_client.create_collection(name="PhD_thesis")

    # read the text file
    chunks = []
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            if len(line.strip()) < 10:
                continue
            chunks.append(line.strip())

    print(f"Loaded {len(chunks)} chunks from the file.") 
    # insert the chunks into the collection
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )
    return collection


def compute_bm25_score(query_terms: List[str], document: str, doc_freq: Dict[str, int], total_docs: int, k1: float = 1.5, b: float = 0.75, avg_doc_len: float = 100) -> float:
    """Compute BM25 score for a document given query terms"""
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
    """Perform keyword search using BM25 scoring"""
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

def hybrid_search(query: str, collection: chromadb.Collection, n_results: int, embedding_weight: float = 0.7) -> List[str]:
    """Combine embedding similarity and keyword search with configurable weights"""
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

def query_db(query: str, n_results: int, collection: chromadb.Collection, embedding_weight: float = 0.7, 
              use_reranking: bool = False, rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    assert n_results > 0, "Number of results must be greater than 0"
    assert isinstance(query, str), "Query must be a string"
    assert 0.0 <= embedding_weight <= 1.0, "embedding_weight must be between 0.0 and 1.0"

    # Stage 1: Hybrid search (embedding + keyword)
    # Get more results if reranking is enabled to have a larger pool
    initial_n_results = n_results * 2 if use_reranking else n_results
    results = hybrid_search(query, collection, initial_n_results, embedding_weight)
    
    if DEBUG:
        print('Hybrid search results:', len(results), 'documents')
    
    # Stage 2: Optional cross-encoder reranking
    if use_reranking and len(results) > 1:
        if DEBUG:
            print(f"Applying cross-encoder reranking with model: {rerank_model}")
        reranked_results = rerank_cross_encoder(query, results, rerank_model)
        # Take only the top n_results after reranking
        final_results = reranked_results[:n_results]
    else:
        final_results = results[:n_results]
    
    if DEBUG:
        print(f'Final results: {len(final_results)} documents\n')
    
    return final_results


class CrossEncoderReranker:
    """Singleton class for managing cross-encoder models with caching and error handling"""
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
        """Load the cross-encoder model with error handling"""
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
        """Check if the model is successfully loaded"""
        return self.model is not None
    
    def predict_batch(self, query_doc_pairs: list) -> list:
        """Predict scores for a batch of query-document pairs"""
        if not self.is_available():
            return [0.0] * len(query_doc_pairs)
        
        try:
            scores = self.model.predict(query_doc_pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            print(f"Error during cross-encoder prediction: {e}")
            return [0.0] * len(query_doc_pairs)

def rerank_cross_encoder(query: str, results: list[str], model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> list[str]:
    """Rerank results using cross-encoder with improved error handling and caching"""
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

def generate_answer_single(query: str, retrial_results: list[str], memory: Optional[DialogMemory] = None):

    prompt = "You are an expert in computer science specializing in information visualization and machine learning. You aim to answer the user's query about a PhD thesis, using the following retrieved Latex documents as the extended knowledge, but not limit to this. Do not mention you are referring to these documents.\n\n"
    
    # Add conversation context if memory exists
    if memory and memory.has_history():
        prompt += memory.get_context() + "\n"
    
    prompt += "Current Query: " + query + "\n\n"
    prompt += "Documents:\n" 

    for doc in retrial_results:
        prompt += "- " + doc + "\n"
    # prompt += "\nAnswer:"

    print("Prompt for Gemini AI:\n", prompt)

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    anwser = response.text.strip()
    print("Answer:", anwser)
    return anwser


def check_reference(query: str) -> bool:
    """
    Check if the query contains a reference to a specific document.
    This is a placeholder function and should be implemented based on actual requirements.
    """
  
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Does the following query requires any citations/references (bibliography)? answer 1 for Yes, 0 for No. No other output. Query: {query}",
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1
        )
    )
    # assert response.text.strip() in ["0", "1"], "Response must be either '0' or '1'"
    print("Check reference response:", response.text.strip())
    return response.text.strip() == "1"

def find_citation(query: str, retrial_results: list[str]) -> list[str] | None:
    """
    Find citation keys from LaTeX documents using regex patterns.
    Looks for \citep{...} and \citeyear{...} commands.
    """
    citation_keys = []
    
    # Regex pattern to match \citep{...} and \citeyear{...}
    citation_pattern = r"\\cite(?:p|year)\{([^}]+)\}"
    
    for chunk in retrial_results:
        matches = re.findall(citation_pattern, chunk)
        for match in matches:
            # Handle multiple citations separated by commas
            keys = [key.strip() for key in match.split(',')]
            citation_keys.extend(keys)
    
    # Remove duplicates while preserving order
    unique_keys = []
    for key in citation_keys:
        if key and key not in unique_keys:
            unique_keys.append(key)
    
    if DEBUG and unique_keys:
        print(f"Found citation keys: {unique_keys}")
    
    return unique_keys if unique_keys else None


def find_reference(keys: list[str], bib='data/bib_entries.json') -> str:
    refs = []
    # open the json file
    with open(bib, "r", encoding="utf-8") as file:
        bib_entries = json.load(file)
    for key in keys:
        key = key.strip()
        if key in bib_entries:
            refs.append(bib_entries[key])
        else:
            print(f"Warning: Key '{key}' not found in the bibliography entries.")
    return refs


def generate_answer_with_citation(query: str, collection: chromadb.Collection, n_results: int = 3, memory: Optional[DialogMemory] = None, 
                                embedding_weight: float = 0.7, use_reranking: bool = False, rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    
    retrial_results = query_db(query, n_results, collection, embedding_weight, use_reranking, rerank_model)
    citations_keys = None
    
    ## check if the query requires a reference
    if not check_reference(query):
        answer = generate_answer_single(query, retrial_results, memory)
        # print("Answer:", answer)
    else:
        citations_keys = find_citation(query, retrial_results)
        if citations_keys is None:
            # print("No citations found for the query.")
            answer = generate_answer_single(query, retrial_results, memory)
            # print("Answer:", answer)
        else:
            # print("Citations keys found:", citations_keys)
            references = find_reference(citations_keys)
            if not references:
                # print("No references found for the citation keys.")
                answer = generate_answer_single(query, retrial_results, memory)
            else:
                # print("References found:", references)
                refs = "\n".join(references) + "NOTE: DONT FILL IN EXTRA INFORMATION ABOUT THE REFERENCES, JUST OUTPUT THEM AS THEY ARE."
                retrial_results.append(f"\nReferences:\n{refs}")
                
                answer = generate_answer_single(query, retrial_results=retrial_results, memory=memory)
                # print("Answer:", answer)
    
    # Add the turn to memory if memory is provided
    if memory:
        memory.add_turn(query, answer, retrial_results, citations_keys)
    
    return answer

if __name__ == "__main__":
    collection = build_db()
    memory = DialogMemory(max_turns=10)  # Initialize dialog memory
    print("Database build completed.")
    print("Dialog memory initialized. The system will remember the last 10 conversation turns.")
    print("Hybrid search enabled: combines embedding similarity and keyword search (BM25)")
    print("Embedding weight controls the balance (1.0 = pure embedding, 0.0 = pure keyword)")
    print("Optional cross-encoder reranking available for improved precision")
    
    while query := input("Enter your query (or 'exit' to quit): ").strip():
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
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
        
        answer = generate_answer_with_citation(query, collection, n_results, memory, embedding_weight, use_reranking, rerank_model)
        print(f"Answer: {answer}")
        
        # Show memory status
        if DEBUG and memory.has_history():
            print(f"[DEBUG] Memory contains {len(memory.turns)} conversation turns.")
 

    # print("Answer generation completed.")