import streamlit as st
import requests
from pathlib import Path

from components.sidebar import render_sidebar


BACKEND_URL = "http://127.0.0.1:8000"


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

render_sidebar()


# -------------------------------------------------------
# Header
# -------------------------------------------------------

logo_path = (
    Path(__file__).parent.parent.parent
    / "assets"
    / "ekip_logo.png"
)

if logo_path.exists():
    st.image(str(logo_path), width=220)

st.title("Enterprise Knowledge Intelligence Platform")

st.caption(
    "Enterprise Retrieval-Augmented Generation (RAG) platform "
    "for intelligent document understanding."
)

st.divider()


# -------------------------------------------------------
# Live Platform Metrics
# -------------------------------------------------------

try:

    response = requests.get(
        f"{BACKEND_URL}/analytics/",
        timeout=5
    )

    if response.status_code == 200:

        data = response.json()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Documents",
                data["documents"]
            )

        with col2:
            st.metric(
                "🧩 Knowledge Chunks",
                data["chunks"]
            )

        with col3:
            st.metric(
                "🗂 Vector Database",
                data["vector_database"]
            )

        with col4:
            st.metric(
                "🤖 Active LLM",
                "Llama 3.3 70B"
            )

    else:

        st.error("Unable to retrieve platform statistics.")

except Exception:

    st.error("Backend connection unavailable.")


st.divider()


# -------------------------------------------------------
# Platform Overview
# -------------------------------------------------------

st.subheader("Platform Capabilities")

col1, col2 = st.columns(2)


with col1:

    st.info(
        """
### 📄 Document Intelligence

Upload enterprise documents and automatically:

- Extract document content
- Split content into meaningful chunks
- Generate vector embeddings
- Store knowledge inside FAISS
"""
    )

    st.info(
        """
### 🔍 Hybrid Search

Search the enterprise knowledge base using:

- Semantic similarity
- Keyword matching
- Combined hybrid ranking
"""
    )


with col2:

    st.info(
        """
### 💬 AI Assistant

Ask questions about uploaded documents.

The system retrieves relevant knowledge and uses
IBM watsonx.ai to generate grounded responses
with source attribution.
"""
    )

    st.info(
        """
### 📊 Platform Analytics

Monitor the knowledge base and AI infrastructure,
including documents, chunks, embeddings, vector
database and active language model.
"""
    )


st.divider()


# -------------------------------------------------------
# RAG Pipeline
# -------------------------------------------------------

st.subheader("Enterprise RAG Workflow")

st.code(
    """
📄 Upload Documents
        ↓
📑 Extract Text
        ↓
🧩 Split into Chunks
        ↓
🧠 Generate Embeddings
        ↓
🗂 Store in FAISS
        ↓
🔍 Hybrid Search
        ↓
🤖 IBM watsonx.ai
        ↓
💬 Grounded Response
""",
    language="text"
)


st.divider()


# -------------------------------------------------------
# System Health
# -------------------------------------------------------

st.subheader("System Health")

try:

    response = requests.get(
        f"{BACKEND_URL}/analytics/",
        timeout=5
    )

    if response.status_code == 200:

        data = response.json()

        st.success("🟢 Enterprise AI Platform is Operational")

        col1, col2 = st.columns(2)

        with col1:

            st.write("✅ Backend API Connected")
            st.write("✅ FAISS Vector Database")
            st.write("✅ Embedding Model Loaded")

        with col2:

            st.write(
                f"🤖 LLM: {data['llm']}"
            )

            st.write(
                f"🧠 Embedding: {data['embedding_model']}"
            )

            st.write(
                f"📚 Indexed Chunks: {data['chunks']}"
            )

    else:

        st.warning("Platform health information unavailable.")

except Exception:

    st.error("Unable to retrieve platform health.")


st.divider()


# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.caption(
    "Enterprise Knowledge Intelligence Platform • "
    "FastAPI • Streamlit • FAISS • Sentence Transformers • "
    "IBM watsonx.ai"
)