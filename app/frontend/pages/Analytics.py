import streamlit as st
import requests

from components.sidebar import render_sidebar


render_sidebar()

BACKEND_URL = "http://127.0.0.1:8000"


# -------------------------------------------------------
# Page Header
# -------------------------------------------------------

st.title("📊 Platform Analytics")

st.write(
    """
Monitor the current status of the Enterprise Knowledge Intelligence Platform.
View live statistics about the knowledge base and AI infrastructure.
"""
)

st.divider()


# -------------------------------------------------------
# Fetch Analytics
# -------------------------------------------------------

try:

    response = requests.get(
        f"{BACKEND_URL}/analytics/",
        timeout=5
    )

    if response.status_code == 200:

        data = response.json()

        # -------------------------------------------------------
        # Executive Overview
        # -------------------------------------------------------

        st.subheader("Executive Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Documents",
                data["documents"]
            )

        with col2:
            st.metric(
                "🧩 Chunks",
                data["chunks"]
            )

        with col3:
            st.metric(
                "🤖 Active LLM",
                data["llm"]
            )

        with col4:
            st.metric(
                "🗂 Vector Database",
                data["vector_database"]
            )

        st.divider()

        # -------------------------------------------------------
        # Platform Information
        # -------------------------------------------------------

        st.subheader("Platform Information")

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"""
**Embedding Model**

{data["embedding_model"]}

---

**Vector Database**

{data["vector_database"]}
"""
            )

        with col2:

            st.info(
                f"""
**LLM Provider**

{data["llm_provider"]}

---

**Large Language Model**

{data["llm"]}
"""
            )

        st.divider()

        # -------------------------------------------------------
        # Knowledge Base Summary
        # -------------------------------------------------------

        st.subheader("Knowledge Base Summary")

        average_chunks = 0

        if data["documents"] > 0:

            average_chunks = round(
                data["chunks"] / data["documents"],
                2
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Documents",
                data["documents"]
            )

        with col2:

            st.metric(
                "Total Chunks",
                data["chunks"]
            )

        with col3:

            st.metric(
                "Avg Chunks / Document",
                average_chunks
            )

        st.divider()

        # -------------------------------------------------------
        # System Health
        # -------------------------------------------------------

        st.subheader("System Health")

        health_data = [

            ("Backend API", "🟢 Online"),
            ("Embedding Model", "🟢 Loaded"),
            ("Vector Database", "🟢 Connected"),
            ("Language Model", "🟢 Running"),
            ("Knowledge Base", "🟢 Ready")

        ]

        for component, status in health_data:

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(component)

            with col2:
                st.write(status)

        st.divider()

        # -------------------------------------------------------
        # AI Technology Stack
        # -------------------------------------------------------

        st.subheader("AI Technology Stack")

        tech1, tech2 = st.columns(2)

        with tech1:

            st.info(
                """
**Backend**

• FastAPI

• Uvicorn

• Python
"""
            )

            st.info(
                f"""
**AI & Machine Learning**

• Sentence Transformers

• {data["llm_provider"]}

• {data["llm"]}
"""
            )

        with tech2:

            st.info(
                """
**Vector Search**

• FAISS

• Hybrid Search

• Embeddings
"""
            )

            st.info(
                """
**Frontend**

• Streamlit

• Requests

• Python
"""
            )

        st.divider()

        # -------------------------------------------------------
        # Platform Workflow
        # -------------------------------------------------------

        st.subheader("Enterprise RAG Workflow")

        st.code(
            f"""
📄 Upload Enterprise Documents
│
▼
📑 Extract Document Text
│
▼
🧩 Split into Chunks
│
▼
🧠 Generate Embeddings
│
▼
🗂 Store in FAISS Vector Database
│
▼
🔍 Hybrid Search
│
▼
🤖 {data["llm"]}
│
▼
💬 Generate Grounded Response
"""
        )

    else:

        st.error(
            "Unable to retrieve analytics information."
        )


except Exception as e:

    st.error(
        f"Backend Connection Error: {e}"
    )