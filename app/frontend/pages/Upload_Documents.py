import streamlit as st
import requests
import pandas as pd

from components.sidebar import render_sidebar


render_sidebar()

BACKEND_URL = "http://127.0.0.1:8000"


# -------------------------------------------------------
# Page Title
# -------------------------------------------------------

st.title("📄 Upload Enterprise Documents")

st.write(
    """
Upload enterprise documents to build the knowledge base.

### Supported Formats

- PDF
- DOCX
- TXT
- Markdown
- CSV
- Excel (.xlsx)
- PowerPoint (.pptx)
- HTML
- JSON
"""
)

st.divider()


# -------------------------------------------------------
# File Upload
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a document",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "csv",
        "xlsx",
        "pptx",
        "html",
        "json"
    ]
)


# -------------------------------------------------------
# Upload Document
# -------------------------------------------------------

if uploaded_file is not None:

    st.info(
        f"Selected File: **{uploaded_file.name}**"
    )

    if st.button(
        "🚀 Upload Document",
        use_container_width=True
    ):

        with st.spinner(
            "Processing document..."
        ):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    uploaded_file.type
                )
            }

            response = requests.post(
                f"{BACKEND_URL}/documents/upload",
                files=files
            )

        if response.status_code == 200:

            result = response.json()

            st.success(
                "✅ Document uploaded successfully."
            )

            st.divider()

            # -------------------------------------------------------
            # Processing Summary
            # -------------------------------------------------------

            st.subheader("Processing Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label="Filename",
                    value=result["filename"]
                )

            with col2:

                st.metric(
                    label="Chunks Created",
                    value=result["chunks_created"]
                )

            st.divider()

            # -------------------------------------------------------
            # Knowledge Base
            # -------------------------------------------------------

            st.subheader("📂 Knowledge Base")

            try:

                documents_response = requests.get(
                    f"{BACKEND_URL}/documents/",
                    timeout=5
                )

                if documents_response.status_code == 200:

                    documents_data = (
                        documents_response.json()
                    )

                    documents = documents_data.get(
                        "documents",
                        []
                    )

                    if documents:

                        history = pd.DataFrame(
                            [
                                {
                                    "Filename": doc["filename"],
                                    "Type": doc["type"]
                                    .replace(".", "")
                                    .upper(),
                                    "Chunks": doc["chunks"],
                                    "Status": "✅ Ready"
                                }
                                for doc in documents
                            ]
                        )

                        st.dataframe(
                            history,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.info(
                            "No documents have been indexed yet."
                        )

                else:

                    st.warning(
                        "Unable to retrieve document list."
                    )

            except Exception as e:

                st.warning(
                    f"Unable to load knowledge base: {e}"
                )

            # -------------------------------------------------------
            # API Response
            # -------------------------------------------------------

            with st.expander(
                "View API Response"
            ):

                st.json(result)

        else:

            st.error(
                "❌ Upload failed."
            )

            st.code(
                response.text
            )


# -------------------------------------------------------
# Knowledge Base
# -------------------------------------------------------

st.divider()

st.subheader("📚 Current Knowledge Base")

try:

    documents_response = requests.get(
        f"{BACKEND_URL}/documents/",
        timeout=5
    )

    if documents_response.status_code == 200:

        documents_data = documents_response.json()

        documents = documents_data.get(
            "documents",
            []
        )

        if documents:

            knowledge_base = pd.DataFrame(
                [
                    {
                        "Filename": doc["filename"],
                        "Type": doc["type"]
                        .replace(".", "")
                        .upper(),
                        "Chunks": doc["chunks"],
                        "Status": "🟢 Ready"
                    }
                    for doc in documents
                ]
            )

            st.dataframe(
                knowledge_base,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No documents have been uploaded yet."
            )

    else:

        st.warning(
            "Unable to retrieve knowledge base."
        )

except Exception:

    st.warning(
        "Backend connection unavailable."
    )