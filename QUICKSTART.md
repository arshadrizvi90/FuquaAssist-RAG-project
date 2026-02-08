# FuquaAssist Quick Start Guide

Get up and running with FuquaAssist in under 10 minutes!

## Prerequisites

- Python 3.8 or higher
- At least one LLM API key (DeepSeek, Gemini, or ArliAI)
- 2GB free disk space

## Installation (5 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/fuqua-assist.git
cd fuqua-assist
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install uvicorn google-generativeai
```

Optional: Suppress tokenizer warnings
```bash
export TOKENIZERS_PARALLELISM=false
```

### Step 4: Configure API Keys

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: At least one of these
export OPENAI_API_KEY="sk-your-deepseek-key"  # For DeepSeek
export OPENAI_BASE_URL="https://api.deepseek.com"

export GOOGLE_API_KEY="your-gemini-key"      # For Gemini

export ARLIAI_API_KEY="your-arliai-key"      # For ArliAI
```

## Quick Test (2 minutes)

### Option 1: Run Web Interface

```bash
uvicorn server:app --reload --port 8000
```

Then open `index.html` in your browser or visit `http://localhost:8000/`

### Option 2: Run Evaluation

Test the system with predefined questions:

```bash
python3 eval.py --qa-file eval_qa.jsonl --mode rag --k 5
```

## Understanding the System

### The RAG Pipeline

```
User Question → Embedding → FAISS Search → Retrieve Top 5 Chunks → LLM → Answer
```

### Default Configuration

- **Embedding Model**: `all-mpnet-base-v2` (best accuracy)
- **Chunk Size**: 600 tokens
- **Overlap**: 50 tokens
- **Top K**: 5 retrieved documents
- **LLM**: DeepSeek-Chat

## Common Tasks

### Add Your Own Documents

1. Place PDF files in `data/pdf/`
2. The system will automatically process them on next run

### Change Embedding Model

Edit `.env`:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Faster but less accurate
# OR
EMBEDDING_MODEL=intfloat/e5-large-v2  # Slower but more accurate
```

### Run Full Evaluation

Test all models and embeddings:

```bash
bash run_all.sh
```

This takes ~10-15 minutes and tests:
- 6 embedding models
- 3 LLM backends
- Generates comparison reports

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### Issue: API key errors

**Solution:** Make sure keys are exported in your current shell:
```bash
source .env  # Re-load environment variables
```

### Issue: FAISS index errors

**Solution:** Delete old index and rebuild:
```bash
rm faiss.index index_meta.json
python3 -c "from src.pipeline import build_index; build_index()"
```

### Issue: Out of memory

**Solution:** Use a smaller embedding model:
```bash
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Next Steps

1. **Add your documents**: Place PDFs in `data/pdf/`
2. **Customize settings**: Edit `config.py` or `.env`
3. **Run evaluations**: Test accuracy with `eval.py`
4. **Deploy**: Follow deployment guide in `docs/DEPLOYMENT.md`

## Performance Expectations

### With all-mpnet-base-v2 (Recommended)
- Exact Match: ~60%
- Retrieval Hit@5: ~80%
- Hallucination: <5%

### With all-MiniLM-L6-v2 (Fast)
- Exact Match: ~45%
- Retrieval Hit@5: ~65%
- Hallucination: <10%

## Getting Help

- Read full [README.md](README.md)
- Check [docs/](docs/) for detailed guides
- Report issues on GitHub
- Contact team members

---

**Ready in 10 minutes!** ⚡
