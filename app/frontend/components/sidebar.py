import streamlit as st
import requests
from pathlib import Path

BACKEND_URL = "http://127.0.0.1:8000"


def render_sidebar():

    with st.sidebar:

        # ---------------------------------------------------
        # Logo
        # ---------------------------------------------------

        logo_path = Path(__file__).parent.parent.parent / "assets" / "ekip_logo.png"

        if logo_path.exists():
            st.image(str(logo_path), width=180)

        st.markdown("## Enterprise AI Platform")

        st.caption("Enterprise Knowledge Intelligence Platform")

        st.divider()

        # ---------------------------------------------------
        # Platform Status
        # ---------------------------------------------------

        st.subheader("🟢 Platform Status")

        try:

            response = requests.get(
                f"{BACKEND_URL}/analytics/",
                timeout=3
            )

            if response.status_code == 200:

                data = response.json()

                st.success("System Healthy")

                st.metric(
                    "Documents",
                    data["documents"]
                )

                st.metric(
                    "Chunks",
                    data["chunks"]
                )

                st.write(f"**Vector DB** : {data['vector_database']}")

                st.write(f"**Embedding** : {data['embedding_model']}")

                st.write(f"**LLM** : {data['llm']}")

            else:

                st.error("Backend Unavailable")

        except:

            st.error("Backend Offline")

        st.divider()

        # ---------------------------------------------------
        # About
        # ---------------------------------------------------

        st.subheader("About")

        st.caption("Enterprise Knowledge Intelligence Platform")

        st.caption("Version 1.0")

        st.caption("Built with")

        st.markdown("""
- ⚡ FastAPI
- 🎨 Streamlit
- 🧠 IBM watsonx.ai
- 📦 FAISS
- 🔎 Sentence Transformers
""")