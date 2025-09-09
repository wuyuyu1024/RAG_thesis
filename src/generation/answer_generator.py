"""Answer generation using Gemini AI with citation support."""

from typing import List, Optional
import chromadb
from google import genai
from models.memory import DialogMemory
from search.hybrid_search import query_db
from generation.citation_handler import check_reference, find_citation, find_reference
from utils.config import GEMINI_API_KEY, DEBUG


def generate_answer_single(query: str, retrial_results: List[str], 
                          memory: Optional[DialogMemory] = None) -> str:
    """Generate a single answer using Gemini AI with retrieved documents."""
    prompt = "You are an expert in computer science specializing in information visualization and machine learning. You aim to answer the user's query about a PhD thesis, using the following retrieved Latex documents as the extended knowledge, but not limit to this. Do not mention you are referring to these documents.\nAlso remove the latex format from the context, replace the format with markdown if needed.\n\n"
    
    # Add conversation context if memory exists
    if memory and memory.has_history():
        prompt += memory.get_context() + "\n"
    
    prompt += "Current Query: " + query + "\n\n"
    prompt += "Documents:\n" 

    for doc in retrial_results:
        prompt += "- " + doc + "\n"

    print("Prompt for Gemini AI:\n", prompt)

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text.strip()
    # print("Answer:", answer)
    return answer


def generate_answer_with_citation(query: str, collection: chromadb.Collection, 
                                 n_results: int = 3, memory: Optional[DialogMemory] = None, 
                                 embedding_weight: float = 0.7, use_reranking: bool = True, 
                                 rerank_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> str:
    """Generate answer with citation support using the full RAG pipeline."""
    retrial_results = query_db(query, n_results, collection, embedding_weight, use_reranking, rerank_model)
    citations_keys = None
    
    # Check if the query requires a reference
    if not check_reference(query):
        answer = generate_answer_single(query, retrial_results, memory)
    else:
        citations_keys = find_citation(query, retrial_results)
        if citations_keys is None:
            answer = generate_answer_single(query, retrial_results, memory)
        else:
            references = find_reference(citations_keys)
            if not references:
                answer = generate_answer_single(query, retrial_results, memory)
            else:
                refs = "\n".join(references) + "\nNOTE: DONT FILL IN EXTRA INFORMATION ABOUT THE REFERENCES, JUST OUTPUT THEM AS THEY ARE. ASLO DONT MENTION THE CITATION KEYS IF YOU USE THEM IN THE ANSWER."
                retrial_results.append(f"\nReferences:\n{refs}")
                
                answer = generate_answer_single(query, retrial_results=retrial_results, memory=memory)
    
    # Add the turn to memory if memory is provided
    if memory:
        memory.add_turn(query, answer, retrial_results, citations_keys)
    
    return answer