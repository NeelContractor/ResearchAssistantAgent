import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass

# Groq
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Search
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "3"))

# Graph
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "25"))