import streamlit as st
import requests
from datetime import datetime

st.set_page_config(
    page_title="Aurora",
    page_icon="⭕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Theme + Background
st.markdown("""
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
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("# ⭕")
with col2:
    st.title("Aurora")
    st.caption("**Next-Gen RAG Assistant** • Intelligent • Grounded • Real-time")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Container
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="⭕" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⭕"):
        with st.spinner("Aurora is thinking..."):
            try:
                resp = requests.post(
                    "http://127.0.0.1:8000",
                    json={"question": prompt, "top_k": 4},
                    timeout=25
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("**Sources**")
                        for s in sources:
                            st.caption(f"📄 {s['source']} (chunk {s.get('chunk', 0)})")
                else:
                    st.error("Backend error")
                    answer = "Error"
            except Exception as e:
                st.error("Cannot connect to backend. Is it running?")
                answer = "Connection error"

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Made for testing only</p>", unsafe_allow_html=True)