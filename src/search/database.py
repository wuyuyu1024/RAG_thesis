"""ChromaDB database setup and management."""

import chromadb
from utils.config import DEBUG


def build_db(filename: str = "./data/chunks.txt"):
    """Build and populate ChromaDB collection from text chunks."""
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