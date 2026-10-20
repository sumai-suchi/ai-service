import os
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_STORE_DIR = os.path.join("data", "vector_store")

# Cache vector store in memory after first load
_vector_store: Optional[FAISS] = None


def get_vector_store() -> Optional[FAISS]:
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(VECTOR_STORE_DIR):
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vector_store = FAISS.load_local(
            VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vector_store


def retrieve_policy_context(query: str, top_k: int = 3) -> str:
    """Search knowledge base and return matching policy context."""
    store = get_vector_store()
    if not store:
        return ""

    docs = store.similarity_search(query, k=top_k)
    if not docs:
        return ""

    context_parts = [f"- {doc.page_content.strip()}" for doc in docs]
    return "\n\n".join(context_parts)