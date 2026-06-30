import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class RAGQuery:
    def __init__(self):
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Use smaller, faster cross-encoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2', max_length=512)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection("tech_docs")

    def retrieve(self, query, top_k=3):
        # 1. Fast bi-encoder retrieval
        results = self.collection.query(
            query_texts=[query],
            n_results=8,                    # Reduced candidates
            include=["documents", "metadatas"]
        )
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        if not docs:
            return {'documents': [[]], 'metadatas': [[]]}
        
        # 2. Fast cross-encoder reranking
        pairs = [[query, doc] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        
        # 3. Sort and take top
        scored = list(zip(docs, metas, scores))
        scored.sort(key=lambda x: x[2], reverse=True)
        
        top_docs = [item[0] for item in scored[:top_k]]
        top_metas = [item[1] for item in scored[:top_k]]
        
        return {
            'documents': [top_docs],
            'metadatas': [top_metas]
        }