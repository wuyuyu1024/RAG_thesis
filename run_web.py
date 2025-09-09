#!/usr/bin/env python3
"""
Startup script for RAG Thesis Web Application.

This script starts the FastAPI web server for the RAG system.
Access the web interface at: http://localhost:8000
"""

import uvicorn
import os
import sys

# Add src directory to path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("Starting RAG Thesis Web Application...")
    print("Web interface will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        app_dir="src"
    )