import streamlit as st
import requests

from components.sidebar import render_sidebar


BACKEND_URL = "http://127.0.0.1:8000"


render_sidebar()


st.title("💬 Enterprise AI Assistant")

st.write(
    "Ask questions about your uploaded enterprise documents "
    "and continue the conversation using previous context."
)

st.divider()


# --------------------------------
# Conversation Memory
# --------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# --------------------------------
# Display Previous Conversation
# --------------------------------

for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            st.write(message["content"])


# --------------------------------
# User Question
# --------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    with st.chat_message("user"):

        st.write(question)

    with st.spinner(
        "Searching knowledge base and generating answer..."
    ):

        try:

            response = requests.post(

                f"{BACKEND_URL}/chat/",

                json={
                    "question": question,
                    "history": st.session_state.chat_history
                },

                timeout=120

            )

            if response.status_code == 200:

                data = response.json()

                answer = data["answer"]

                # --------------------------------
                # Save Conversation
                # --------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                # --------------------------------
                # Display Answer
                # --------------------------------

                with st.chat_message("assistant"):

                    st.write(answer)

                # --------------------------------
                # Sources
                # --------------------------------

                sources = data.get(
                    "sources",
                    []
                )

                if sources:

                    st.divider()

                    st.subheader(
                        "📚 Sources Used"
                    )

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        with st.expander(
                            f"{index}. 📄 "
                            f"{source['document']} "
                            f"| Chunk {source['chunk_number']}"
                        ):

                            st.write(
                                f"**Document:** "
                                f"{source['document']}"
                            )

                            st.write(
                                f"**Chunk Number:** "
                                f"{source['chunk_number']}"
                            )

                            st.write(
                                f"**Hybrid Score:** "
                                f"{source.get('score', 'N/A')}"
                            )

                            st.caption(
                                f"Chunk ID: "
                                f"{source['chunk_id']}"
                            )

                            st.markdown(
                                "**Document Preview**"
                            )

                            st.write(
                                source.get(
                                    "preview",
                                    "No preview available."
                                )
                            )

            else:

                st.error(
                    "Unable to generate answer."
                )

                st.code(
                    response.text
                )

        except requests.exceptions.RequestException as e:

            st.error(
                f"Backend connection error: {e}"
            )


# --------------------------------
# Clear Conversation
# --------------------------------

if st.session_state.chat_history:

    st.divider()

    if st.button("🗑 Clear Conversation"):

        st.session_state.chat_history = []

        st.rerun()