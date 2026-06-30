import chromadb
from chromadb.utils import embedding_functions

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "tech_docs"

_embedding_function = None
_chroma_client = None


def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
    return _embedding_function


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )
