"""
Configuration module for FuquaAssist RAG system.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DATA_DIR = Path(os.getenv("PDF_DATA_DIR", str(DATA_DIR / "pdf")))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(DATA_DIR / "processed")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")))

# Ensure directories exist
PDF_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# API KEYS
# ================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ARLIAI_API_KEY = os.getenv("ARLIAI_API_KEY")

# ================================================================
# EMBEDDING CONFIGURATION
# ================================================================

# Default embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Available embedding models
AVAILABLE_EMBEDDINGS = [
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "gtr-t5-base"
]

# ================================================================
# CHUNKING PARAMETERS
# ================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ================================================================
# RETRIEVAL PARAMETERS
# ================================================================

TOP_K = int(os.getenv("TOP_K", 5))
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 0.8))

# ================================================================
# FAISS INDEX
# ================================================================

FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(PROJECT_ROOT / "faiss.index")))
INDEX_METADATA_PATH = Path(os.getenv("INDEX_METADATA_PATH", str(PROJECT_ROOT / "index_meta.json")))

# ================================================================
# LLM CONFIGURATION
# ================================================================

LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

# Available LLM models
AVAILABLE_LLMS = {
    "deepseek": "deepseek-chat",
    "gemini": "gemini-1.5-flash",
    "arliai": "Gemma-3-27B-it"
}

# ================================================================
# SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """You are FuquaAssist, an AI assistant for the Fuqua School of Business.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the CONTEXT provided below
2. If the context does not contain the answer, say "I don't have that information in the provided documents"
3. Do NOT use external knowledge or make assumptions
4. Be precise and cite specific policies when possible
5. If asked about procedures, provide step-by-step instructions from the context

CONTEXT:
{context}

Question: {question}

Answer based ONLY on the context above:"""

# ================================================================
# EVALUATION
# ================================================================

EVAL_QA_FILE = Path(os.getenv("EVAL_QA_FILE", str(PROJECT_ROOT / "eval" / "eval_qa.jsonl")))

# ================================================================
# SERVER CONFIGURATION
# ================================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000").split(",")

# ================================================================
# LOGGING
# ================================================================

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ================================================================
# TOKENIZER SETTINGS
# ================================================================

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# ================================================================
# VALIDATION
# ================================================================

def validate_config() -> bool:
    """Validate that required configuration is present."""
    errors = []
    
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        errors.append("At least one API key (OPENAI_API_KEY or GOOGLE_API_KEY) must be set")
    
    if EMBEDDING_MODEL not in AVAILABLE_EMBEDDINGS:
        errors.append(f"Invalid EMBEDDING_MODEL: {EMBEDDING_MODEL}. Must be one of {AVAILABLE_EMBEDDINGS}")
    
    if CHUNK_SIZE <= 0:
        errors.append(f"CHUNK_SIZE must be positive, got {CHUNK_SIZE}")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})")
    
    if TOP_K <= 0:
        errors.append(f"TOP_K must be positive, got {TOP_K}")
    
    if errors:
        for error in errors:
            print(f"❌ Configuration Error: {error}")
        return False
    
    return True

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_embedding_model() -> str:
    """Get the current embedding model."""
    return EMBEDDING_MODEL

def get_llm_config() -> dict:
    """Get LLM configuration dictionary."""
    return {
        "model": LLM_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL
    }

def get_chunking_config() -> dict:
    """Get chunking configuration dictionary."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP
    }

def get_retrieval_config() -> dict:
    """Get retrieval configuration dictionary."""
    return {
        "top_k": TOP_K,
        "distance_threshold": DISTANCE_THRESHOLD
    }

# ================================================================
# CONFIGURATION SUMMARY
# ================================================================

def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("FuquaAssist Configuration Summary")
    print("=" * 60)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Chunk Size: {CHUNK_SIZE} tokens")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} tokens")
    print(f"Top K Retrieval: {TOP_K}")
    print(f"Distance Threshold: {DISTANCE_THRESHOLD}")
    print(f"PDF Data Directory: {PDF_DATA_DIR}")
    print(f"FAISS Index: {FAISS_INDEX_PATH}")
    print(f"Debug Mode: {DEBUG}")
    print("=" * 60)

# Validate configuration on import
if __name__ != "__main__":
    if not validate_config():
        print("\n⚠️  Configuration validation failed. Please check your .env file.")
