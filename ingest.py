import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, INDEX_DIR
from utils import ensure_directories

# ✅ Embeddings (CPU only)
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

def load_document(file_path):
    """Load a document based on file type."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def ingest_files(file_paths):
    """Ingest multiple files into FAISS index."""
    ensure_directories(INDEX_DIR)

    docs = []
    for file_path in file_paths:
        docs.extend(load_document(file_path))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Create or update FAISS index
    if os.path.exists(INDEX_DIR):
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(split_docs)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(INDEX_DIR)
    return f"Ingested {len(split_docs)} chunks."





