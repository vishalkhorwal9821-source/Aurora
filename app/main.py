from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from query import RAGQuery
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Aurora - Tech Docs Assistant")

rag = RAGQuery()

# Try to load Groq
try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    USE_GROQ = True
except:
    client = None
    USE_GROQ = False
    print("Groq not available. Using fallback.")

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class Source(BaseModel):
    source: str
    chunk: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question too short")
    
    results = rag.retrieve(request.question, request.top_k)
    
    if not results or not results['documents'][0]:
        return QueryResponse(answer="I don't know.", sources=[])
    
    context = "\n\n".join(results['documents'][0])
    sources = [Source(source=m.get('source', 'unknown'), chunk=m.get('chunk', 0)) for m in results['metadatas'][0]]
    
    prompt = f"Context:\n{context}\n\nQuestion: {request.question}\nAnswer concisely:"
    
    if USE_GROQ and client:
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM error: {str(e)}"
    else:
        answer = "Groq not configured. Using fallback answer."
    
    return QueryResponse(answer=answer, sources=sources)

@app.get("/health")
async def health():
    return {"status": "healthy", "groq": USE_GROQ}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)