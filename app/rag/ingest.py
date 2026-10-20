import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_DIR = os.path.join("data", "knowledge")
VECTOR_STORE_DIR = os.path.join("data", "vector_store")


def run_ingestion():
    print(f"Loading markdown files from {KNOWLEDGE_DIR}...")
    
    # 1. Load markdown files
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Create Hugging Face Embeddings (Runs locally on CPU for free)
    print("Generating Hugging Face embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Save to FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    
    print(f"Successfully saved FAISS index to {VECTOR_STORE_DIR}!")


if __name__ == "__main__":
    run_ingestion()