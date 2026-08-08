import streamlit as st
import requests

from components.sidebar import render_sidebar

render_sidebar()

BACKEND_URL = "http://127.0.0.1:8000"

# -------------------------------------------------------
# Page Title
# -------------------------------------------------------

st.title("🔍 Semantic Search")

st.write(
    """
Search across uploaded enterprise documents using semantic similarity.
Retrieve the most relevant document chunks based on meaning rather than exact keywords.
"""
)

st.divider()

# -------------------------------------------------------
# Search Input
# -------------------------------------------------------

query = st.text_input(
    "Enter your search query"
)

top_k = st.slider(
    "Number of Results",
    min_value=1,
    max_value=10,
    value=5
)

# -------------------------------------------------------
# Search Button
# -------------------------------------------------------

if st.button("🔍 Search", use_container_width=True):

    if query.strip() == "":

        st.warning("Please enter a search query.")

    else:

        with st.spinner("Searching knowledge base..."):

            response = requests.post(
                f"{BACKEND_URL}/search/",
                json={
                    "query": query,
                    "top_k": top_k
                }
            )

        if response.status_code == 200:

            data = response.json()

            st.success(
                f"Found {data['results_found']} relevant result(s)."
            )

            st.divider()

            for result in data["results"]:

                with st.container(border=True):

                    st.markdown(
                        f"### 📄 {result['document']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Chunk Number:** {result['chunk_number']}"
                        )

                    with col2:

                        st.write(
                            f"**Search Rank:** #{result['rank']}"
                        )

                    st.caption("Document Preview")

                    st.write(result["preview"])

        else:

            st.error("Search request failed.")

            st.code(response.text)