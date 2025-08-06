import chromadb
from google import genai
import dotenv
import os

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

def generate_answer_single(query: str, collection: chromadb.Collection, n_results: int = 3):

    retrial_results = query_db(query, n_results, collection)

    if not retrial_results or not retrial_results['documents']:
        print("No results found in the database.")
        return None

    prompt = "You are an expert in computer science specializing in information visualization and machine learning. You aim to answer the user's query about a PhD thesis, using the following retrieved Latex documents as the extended knowledge, but not limit to this. Do not mention you are referring to these documents.\n\n"
    prompt += "Query: " + query + "\n\n"
    prompt += "Documents:\n" 

    for doc in retrial_results['documents'][0]:
        prompt += "- " + doc + "\n"
    prompt += "\nAnswer:"

    print("Prompt for Gemini AI:\n", prompt)

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print(response.text)

if __name__ == "__main__":
    collection = build_db()
    print("Database build completed.")
    
    while query:= input("Enter your query (or 'exit' to quit): ").strip():
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        n_results = int(input("Enter number of results to return: "))
        generate_answer_single(query, collection, n_results)
 

    # print("Answer generation completed.")