"""Citation detection and reference lookup functionality."""

import re
import json
from typing import List, Optional
from google import genai
from google.genai import types
from utils.config import DEBUG, GEMINI_API_KEY


def check_reference(query: str) -> bool:
    """Check if the query requires citations/references using Gemini AI."""
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"Does the following query requires any citations/references (bibliography)? answer 1 for Yes, 0 for No. No other output. Query: {query}",
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1
        )
    )
    print("Check reference response:", response.text.strip())
    return response.text.strip() == "1"


def find_citation(query: str, retrial_results: List[str]) -> Optional[List[str]]:
    """Find citation keys from LaTeX documents using regex patterns."""
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


def find_reference(keys: List[str], bib: str = 'data/bib_entries.json') -> List[str]:
    """Look up references from bibliography entries JSON file."""
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