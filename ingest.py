import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pypdf

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="tech_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )

    def load_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def ingest_documents(self, doc_paths):
        for path in doc_paths:
            if path.endswith('.pdf'):
                text = self.load_pdf(path)
            else:
                with open(path, 'r') as f:
                    text = f.read()
            
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    ids=[f"chunk_{os.path.basename(path)}_{i}"],
                    metadatas=[{"source": path, "chunk": i}]
                )
        print(f"Indexed {len(chunks)} chunks from documents")

if __name__ == "__main__":
    processor = DocumentProcessor()
    
    os.makedirs("docs", exist_ok=True)
    sample_text = """RAG stands for Retrieval Augmented Generation.
It combines retrieval of relevant documents with generation from LLMs.
This improves accuracy and reduces hallucinations in answers.
Vector embeddings capture semantic meaning of text chunks.
ChromaDB provides fast similarity search over these embeddings."""
    
    with open("docs/sample_tech_doc.txt", "w") as f:
        f.write(sample_text)
    
    processor.ingest_documents(["docs/sample_tech_doc.txt"])