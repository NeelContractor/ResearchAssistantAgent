import os
from dotenv import load_dotenv

load_dotenv()

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Search
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
# SERP_API_KEY: str = os.getenv("SERP_API_KEY", "")
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "3"))

# Graph
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", "25"))