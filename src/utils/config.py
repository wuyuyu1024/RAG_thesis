"""Configuration settings for the RAG thesis application."""

import os
import dotenv

dotenv.load_dotenv()

DEBUG = True

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")