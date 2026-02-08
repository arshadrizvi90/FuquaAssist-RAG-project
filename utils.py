"""
Utility functions for FuquaAssist RAG system.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def save_json(data: Any, filepath: Path):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def ensure_dir(directory: Path):
    """Ensure directory exists, create if it doesn't."""
    directory.mkdir(parents=True, exist_ok=True)


def get_file_list(directory: Path, extension: str = ".pdf") -> List[Path]:
    """Get list of files with specific extension in directory."""
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist")
        return []
    
    files = list(directory.glob(f"*{extension}"))
    logger.info(f"Found {len(files)} {extension} files in {directory}")
    return files


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and standardizing format."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common PDF artifacts
    text = text.replace('\x00', '')
    return text.strip()


def chunk_text_sliding_window(
    text: str,
    chunk_size: int = 600,
    overlap: int = 50,
    tokenizer = None
) -> List[str]:
    """
    Chunk text using sliding window approach.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in tokens
        overlap: Number of overlapping tokens between chunks
        tokenizer: Optional tokenizer (if None, uses simple whitespace split)
    
    Returns:
        List of text chunks
    """
    if tokenizer:
        # Use provided tokenizer
        tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            start += (chunk_size - overlap)
    else:
        # Simple word-based chunking
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            start += (chunk_size - overlap)
    
    logger.info(f"Created {len(chunks)} chunks from text of {len(text)} characters")
    return chunks


def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def format_retrieved_context(chunks: List[str], max_length: int = 3000) -> str:
    """
    Format retrieved chunks into a single context string.
    
    Args:
        chunks: List of retrieved text chunks
        max_length: Maximum total character length
    
    Returns:
        Formatted context string
    """
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks):
        chunk_text = f"[Document {i+1}]\n{chunk}\n"
        
        if current_length + len(chunk_text) > max_length:
            break
        
        context_parts.append(chunk_text)
        current_length += len(chunk_text)
    
    return "\n".join(context_parts)


def clean_pdf_text(text: str) -> str:
    """Clean common PDF artifacts from extracted text."""
    # Remove page numbers
    import re
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common headers/footers patterns
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Normalize whitespace
    text = normalize_text(text)
    
    return text


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {elapsed:.2f} seconds")


def validate_api_keys() -> Dict[str, bool]:
    """Check which API keys are configured."""
    keys = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        "ARLIAI_API_KEY": bool(os.getenv("ARLIAI_API_KEY"))
    }
    
    for key, present in keys.items():
        status = "✓" if present else "✗"
        logger.info(f"{status} {key}: {'Configured' if present else 'Not configured'}")
    
    return keys


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    import torch
    
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    
    return device


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Rough estimate of token count from character count."""
    return len(text) // chars_per_token


def truncate_text(text: str, max_tokens: int = 500, chars_per_token: int = 4) -> str:
    """Truncate text to approximate token limit."""
    max_chars = max_tokens * chars_per_token
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
