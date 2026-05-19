from flask import Flask, request, jsonify
from rag_pipeline import RAGPipeline
from groq import Groq
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)

# Initialize RAG pipeline and Groq client
rag = RAGPipeline()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Try loading existing index on startup
rag.load_index()


def build_prompt(context: str, query: str) -> str:
    return f"""You are a helpful knowledge assistant. Use only the provided context to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "index_loaded": rag.vector_store is not None,
        "timestamp": time.time()
    })


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Ingest documents into the RAG vector store.
    Expects JSON: { "texts": [...], "metadatas": [...] (optional) }
    """
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "Request must include 'texts' field"}), 400

    texts = data["texts"]
    metadatas = data.get("metadatas", None)

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400

    chunk_count = rag.ingest_documents(texts, metadatas)
    rag.save_index()

    return jsonify({
        "message": "Documents ingested successfully",
        "chunks_stored": chunk_count
    })


@app.route("/query", methods=["POST"])
def query():
    """
    Query the RAG system.
    Expects JSON: { "query": "your question here", "top_k": 3 (optional) }
    """
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Request must include 'query' field"}), 400

    user_query = data["query"].strip()
    top_k = data.get("top_k", 3)

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Step 1: Retrieve relevant chunks
    retrieved_docs = rag.retrieve(user_query, k=top_k)

    if not retrieved_docs:
        return jsonify({
            "query": user_query,
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": []
        })

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Generate answer via Groq LLaMA
    prompt = build_prompt(context, user_query)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    return jsonify({
        "query": user_query,
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
