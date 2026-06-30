import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from embeddings import get_chroma_client, get_embedding_function, get_collection
from ingest import DocumentProcessor

load_dotenv()

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-4-v2", max_length=512
        )
    return _cross_encoder


def discover_docs():
    docs = []
    if os.path.exists("docs"):
        for filename in os.listdir("docs"):
            if filename.endswith(".pdf") or filename.endswith(".txt"):
                docs.append(os.path.join("docs", filename))
    return docs


def ensure_indexed(collection):
    if collection.count() > 0:
        return collection

    docs = discover_docs()
    if not docs:
        return collection

    processor = DocumentProcessor()
    processor.ingest_documents(docs)

    return get_chroma_client().get_collection(
        COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


class RAGQuery:
    def __init__(self):
        self.client = get_chroma_client()
        self.collection = get_collection()
        self.collection = ensure_indexed(self.collection)
        self.cross_encoder = get_cross_encoder()

    def retrieve(self, query, top_k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=8,
            include=["documents", "metadatas"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return {"documents": [[]], "metadatas": [[]]}

        pairs = [[query, doc] for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        scored = list(zip(docs, metas, scores))
        scored.sort(key=lambda x: x[2], reverse=True)

        top_docs = [item[0] for item in scored[:top_k]]
        top_metas = [item[1] for item in scored[:top_k]]

        return {
            "documents": [top_docs],
            "metadatas": [top_metas],
        }
