import os

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen3:0.6b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Agent configuration
AGENT_TEMPERATURE = 0.7
AGENT_MAX_TOKENS = 500

# Data directories configuration
DATA_DIR = os.getenv("DATA_DIR", "/app/data")

# Document directories (inside DATA_DIR)
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_docs")
CLEAR_DOCS_DIR = os.path.join(DATA_DIR, "clear_docs")
