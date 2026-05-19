from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using FAISS vector store
    and HuggingFace sentence-transformers for embeddings.
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        print("[RAG] Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        print("[RAG] Embedding model loaded successfully.")

    def ingest_documents(self, texts: list[str], metadatas: list[dict] = None) -> int:
        """
        Ingest raw text documents into the FAISS vector store.
        Splits into chunks and embeds each chunk.
        Returns total number of chunks stored.
        """
        all_docs = []

        for i, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            meta = metadatas[i] if metadatas else {"source": f"document_{i}"}
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=meta))

        if not all_docs:
            print("[RAG] No valid chunks to ingest.")
            return 0

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_docs, self.embeddings)
        else:
            self.vector_store.add_documents(all_docs)

        print(f"[RAG] Ingested {len(all_docs)} chunks into vector store.")
        return len(all_docs)

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """
        Retrieve top-k most relevant document chunks for a given query.
        Uses cosine similarity over FAISS index.
        """
        if self.vector_store is None:
            print("[RAG] Vector store is empty. Please ingest documents first.")
            return []

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def save_index(self, path: str = "faiss_index"):
        """Persist FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"[RAG] Index saved to '{path}'.")

    def load_index(self, path: str = "faiss_index"):
        """Load a previously saved FAISS index from disk."""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[RAG] Index loaded from '{path}'.")
        else:
            print(f"[RAG] No existing index found at '{path}'. Starting fresh.")
