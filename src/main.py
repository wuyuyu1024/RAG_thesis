import chromadb
from google import genai
from google.genai import types
import dotenv
import os
import sys
from dataclasses import dataclass
import json

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

    results = collection.query(query_texts=[query], n_results=n_results)
    return results

def generate_answer_single(query: str, retrial_results: list[str]):

    prompt = "You are an expert in computer science specializing in information visualization and machine learning. You aim to answer the user's query about a PhD thesis, using the following retrieved Latex documents as the extended knowledge, but not limit to this. Do not mention you are referring to these documents.\n\n"
    prompt += "Query: " + query + "\n\n"
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
    print("AI Response:", anwser)
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

def find_citation(query: str, retrial_results: str) -> list[str] | None:
    """
    Find the citation for a specific document based on the query.
    This is a placeholder function and should be implemented based on actual requirements.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Find the related citations keys in the document for the following query: {query}. The related documents are: {retrial_results}",
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=100,
            system_instruction= "You should find the citation keys from the .tex document related to the query. Output the keys in a single line, separated by commas. If no keys are found, output 'No keys found'."
        )
    )
    if response.text.strip() == "No keys found":
        return None
    return response.text.strip().split(",")

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


def generate_answer_with_citation(query: str, collection: chromadb.Collection, n_results: int = 3):
    
    retrial_results = query_db(query, n_results, collection)['documents'][0]
    ## check if the query requires a reference
    if not check_reference(query):
        anwser = generate_answer_single(query, retrial_results)
        print("Answer:", anwser)
    else:
        citations_keys = find_citation(query, retrial_results)
        if citations_keys is None:
            print("No citations found for the query.")
            anwser = generate_answer_single(query, retrial_results)
            # print("Answer:", anwser)
        else:
            print("Citations keys found:", citations_keys)
            references = find_reference(citations_keys)
            if not references:
                print("No references found for the citation keys.")
                anwser = generate_answer_single(query, retrial_results)
            else:
                print("References found:", references)
                refs = "\n".join(references) + "NOTE: DONT FILL IN EXTRA INFORMATION ABOUT THE REFERENCES, JUST OUTPUT THEM AS THEY ARE."
                retrial_results.append(f"\nReferences:\n{refs}")
                
                anwser = generate_answer_single(query, retrial_results=retrial_results)
                # print("Answer:", anwser)

if __name__ == "__main__":
    collection = build_db()
    print("Database build completed.")
    
    while query:= input("Enter your query (or 'exit' to quit): ").strip():
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        n_results = int(input("Enter number of results to return: "))
        generate_answer_with_citation(query, collection, n_results)
 

    # print("Answer generation completed.")