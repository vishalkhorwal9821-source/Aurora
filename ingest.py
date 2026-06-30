import os
import pypdf

from embeddings import get_chroma_client, get_collection


class DocumentProcessor:
    def __init__(self, client=None, collection=None):
        self.client = client or get_chroma_client()
        self.collection = collection or get_collection()

    def load_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ingest_documents(self, doc_paths):
        total_chunks = 0
        for path in doc_paths:
            if path.endswith(".pdf"):
                text = self.load_pdf(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

            chunks = self.chunk_text(text)
            total_chunks += len(chunks)

            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    ids=[f"chunk_{os.path.basename(path)}_{i}"],
                    metadatas=[{"source": path, "chunk": i}],
                )

        print(f"Indexed {total_chunks} chunks from {len(doc_paths)} documents")


if __name__ == "__main__":
    processor = DocumentProcessor()

    os.makedirs("docs", exist_ok=True)
    sample_text = """RAG stands for Retrieval Augmented Generation.
It combines retrieval of relevant documents with generation from LLMs.
This improves accuracy and reduces hallucinations in answers.
Vector embeddings capture semantic meaning of text chunks.
ChromaDB provides fast similarity search over these embeddings."""

    with open("docs/sample_tech_doc.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    processor.ingest_documents(["docs/sample_tech_doc.txt"])
