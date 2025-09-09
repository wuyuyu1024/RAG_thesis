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


def query_db(query: str, n_results: int, collection: chromadb.Collection):
    assert n_results > 0, "Number of results must be greater than 0"
    assert isinstance(query, str), "Query must be a string"

    results = collection.query(query_texts=[query], n_results=n_results*3)['documents'][0]
    reranked_results = rerank_cross_encoder(query, results)[:n_results]
    print('raw results:', results, '\n')
    print('reranked results:', reranked_results)
    return reranked_results


def rerank_cross_encoder(query: str, results: list[str]) -> list[str]:
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = model.predict([(query, doc) for doc in results])
    ranked_results = [doc for _, doc in sorted(zip(scores, results), reverse=True)]
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


def generate_answer_with_citation(query: str, collection: chromadb.Collection, n_results: int = 3, memory: Optional[DialogMemory] = None):
    
    retrial_results = query_db(query, n_results, collection)
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
    
    while query := input("Enter your query (or 'exit' to quit): ").strip():
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        n_results = int(input("Enter number of results to return: "))
        answer = generate_answer_with_citation(query, collection, n_results, memory)
        print(f"Answer: {answer}")
        
        # Show memory status
        if DEBUG and memory.has_history():
            print(f"[DEBUG] Memory contains {len(memory.turns)} conversation turns.")
 

    # print("Answer generation completed.")