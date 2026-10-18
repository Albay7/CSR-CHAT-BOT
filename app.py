"""
Cambridge Support RAG Bot - Streamlit Chat Interface
Provides an interactive, collegiate-themed conversational interface
powered by retrieval-augmented generation.
"""

import os
import streamlit as st
from rag_engine import CambridgeRAGEngine

# Page Configuration
st.set_page_config(
    page_title="Cambridge Support RAG Bot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for University of Cambridge aesthetic
st.markdown("""
<style>
    /* Cambridge Navy & Accent Colors */
    :root {
        --cambridge-blue: #0072cf;
        --cambridge-navy: #0b2341;
        --cambridge-light-blue: #e8f4fd;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0b2341 0%, #174276 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    .source-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }

    .source-card a {
        color: #0072cf;
        text-decoration: none;
        font-weight: 600;
    }

    .source-card a:hover {
        text-decoration: underline;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine():
    """Initializes and caches the RAG engine instance."""
    return CambridgeRAGEngine()


# Initialize RAG Engine
engine = get_rag_engine()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the Cambridge Student & Applicant Support Bot!**\n\n"
                "I can help you with questions about:\n"
                "- 📚 **Undergraduate & Postgraduate Admissions** & Deadlines\n"
                "- 🏛️ **The Collegiate System** and choosing a college\n"
                "- 🎓 **Supervisions** and Academic Guidance\n"
                "- 💷 **Cambridge Bursary Scheme** & Financial Support\n"
                "- 🏥 **Health, Counselling, & Disability Services (ADRC)**\n"
                "- 💼 **Careers Service** & Student Life\n\n"
                "What would you like to explore today?"
            ),
            "sources": []
        }
    ]

# Sidebar
with st.sidebar:
    st.title("🎓 Cambridge Support")
    st.markdown("Retrieval-Augmented Generation (RAG) assistant for University of Cambridge services.")

    st.divider()

    # System Status
    st.subheader("System Status")
    if engine.is_configured:
        st.success(f"🟢 **LLM Active**: {engine.provider.capitalize()} OpenAI")
        st.caption(f"Chat Model: `{engine.chat_model}`")
        st.caption(f"Embedding: `{engine.embed_model}`")
    else:
        st.info("🟡 **Offline Knowledge Mode**")
        st.caption("Add your API key to `.env` to enable full LLM generation.")

    st.caption(f"📄 Indexed Knowledge Base: **{len(engine.documents)} documents**")

    st.divider()

    # Configuration Slider
    st.subheader("Search Parameters")
    top_k = st.slider("Top Documents (k)", min_value=1, max_value=6, value=3)

    st.divider()

    # Starter Questions
    st.subheader("💡 Suggested Questions")
    sample_queries = [
        "What are the application deadlines for Cambridge?",
        "How does the collegiate system work?",
        "What is the Cambridge Bursary Scheme?",
        "What is a supervision and who is my Director of Studies?",
        "Where can I find mental health and counselling support?",
    ]

    for sq in sample_queries:
        if st.button(sq, key=f"sq_{hash(sq)}", use_container_width=True):
            st.session_state.next_prompt = sq

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header Display
st.markdown("""
<div class="main-header">
    <h1>🎓 Cambridge Student & Applicant Support</h1>
    <p>Powered by Azure OpenAI & Semantic Document Retrieval</p>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Verified References & Official Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src.get('title')}** ({src.get('category')}): "
                        f"[{src.get('url')}]({src.get('url')})"
                    )

# Handle prompt either from button click or user input
user_query = None
if "next_prompt" in st.session_state and st.session_state.next_prompt:
    user_query = st.session_state.pop("next_prompt")
else:
    user_query = st.chat_input("Ask a question about Cambridge study, admissions, or support...")

if user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching Cambridge knowledge base and generating answer..."):
            history_tuples = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ["user", "assistant"]
            ]
            response_data = engine.generate_response(
                query=user_query,
                chat_history=history_tuples,
                top_k=top_k
            )

            answer = response_data["answer"]
            sources = response_data.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("📚 Verified References & Official Sources"):
                    for src in sources:
                        st.markdown(
                            f"- **{src.get('title')}** ({src.get('category')}): "
                            f"[{src.get('url')}]({src.get('url')})"
                        )

            # Store in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
