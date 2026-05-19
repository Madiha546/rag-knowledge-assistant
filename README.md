# RAG-Powered Knowledge Assistant

A Retrieval-Augmented Generation (RAG) system that enables semantic search over custom document collections using FAISS vector store and Groq LLaMA 3.1.

## Architecture

```
User Query
    ↓
[Flask REST API]
    ↓
[Embedding Model] → sentence-transformers/all-MiniLM-L6-v2
    ↓
[FAISS Vector Store] → similarity_search (top-k chunks)
    ↓
[Groq LLaMA 3.1] → generate answer from retrieved context
    ↓
JSON Response
```

## Features

- **Semantic Search** — Embeds queries and retrieves the most relevant document chunks using cosine similarity
- **Chunking Strategy** — Recursive character splitting with configurable chunk size (500 tokens) and overlap (50 tokens) to preserve context boundaries
- **FAISS Index** — Persisted to disk for fast reload without re-embedding
- **Groq LLaMA 3.1** — Low-latency LLM inference for answer generation
- **Hallucination Reduction** — Prompt engineered to answer strictly from retrieved context

## Setup

```bash
# Clone and install
git clone https://github.com/Madiha546/rag-knowledge-assistant
cd rag-knowledge-assistant
pip install -r requirements.txt

# Add environment variables
echo "GROQ_API_KEY=your_groq_api_key" > .env

# Run
python app.py
```

## API Endpoints

### POST /ingest
Ingest documents into the vector store.
```json
{
  "texts": ["Your document text here..."],
  "metadatas": [{"source": "doc1.pdf"}]
}
```

### POST /query
Query the knowledge base.
```json
{
  "query": "What is the main topic?",
  "top_k": 3
}
```

### GET /health
Check system status and index availability.

## Tech Stack

- **Python** — Core language
- **Flask** — REST API framework
- **LangChain** — Document processing and chunking
- **FAISS** — Vector similarity search
- **sentence-transformers** — Local embedding model (no API cost)
- **Groq LLaMA 3.1** — Answer generation

## Future Improvements

- [ ] PDF/DOCX file upload support
- [ ] Reranking layer (Cohere/BGE reranker)
- [ ] Streaming responses
- [ ] Multi-collection support
- [ ] React frontend dashboard
