import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from query import RAGQuery

load_dotenv()


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_rag():
    return RAGQuery()


@st.cache_resource
def load_groq_client():
    api_key = get_groq_api_key()
    if not api_key:
        return None
    return Groq(api_key=api_key)


def generate_answer(question, top_k=4):
    rag = load_rag()
    groq_client = load_groq_client()

    results = rag.retrieve(question, top_k)
    if not results or not results["documents"][0]:
        return "I don't know.", []

    context = "\n\n".join(results["documents"][0])
    sources = [
        {"source": m.get("source", "unknown"), "chunk": m.get("chunk", 0)}
        for m in results["metadatas"][0]
    ]

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"

    if not groq_client:
        return "Groq API key not configured.", sources

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    return answer, sources


st.set_page_config(
    page_title="Aurora",
    page_icon="⭕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    .stChatMessage {
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border-radius: 50px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("# ⭕")
with col2:
    st.title("Aurora")
    st.caption("**Next-Gen RAG Assistant** • Intelligent • Grounded • Real-time")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"], avatar="⭕" if msg["role"] == "assistant" else "👤"
        ):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⭕"):
        with st.spinner("Aurora is thinking..."):
            answer, sources = generate_answer(prompt)

            st.markdown(answer)

            if sources:
                st.markdown("**Sources**")
                for source in sources:
                    st.caption(
                        f"📄 {source['source']} (chunk {source.get('chunk', 0)})"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Made for testing only</p>",
    unsafe_allow_html=True,
)
