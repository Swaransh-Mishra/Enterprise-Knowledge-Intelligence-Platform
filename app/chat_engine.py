"""
Enterprise RAG Chat Engine.

Uses hybrid search, conversation memory,
and a configurable LLM provider.
"""

import re

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from app.hybrid_search import HybridSearch
from app.config.settings import settings


class ChatEngine:
    """
    Retrieval-Augmented Generation chat engine.
    """

    def __init__(self):

        self.search_engine = HybridSearch()

        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL

        # --------------------------------
        # IBM watsonx.ai
        # --------------------------------

        self.watsonx_model = None

        if self.provider == "watsonx":

            credentials = Credentials(
                url=settings.WATSONX_URL,
                api_key=settings.WATSONX_APIKEY
            )

            self.watsonx_model = ModelInference(
                model_id=self.model,
                credentials=credentials,
                project_id=settings.WATSONX_PROJECT_ID
            )

        print(
            f"\nChat engine ready."
            f"\nProvider: {self.provider}"
            f"\nModel: {self.model}\n"
        )

    # --------------------------------
    # Detect Document Name
    # --------------------------------

    def detect_document_name(
        self,
        question: str
    ):

        pattern = (
            r"\b[\w.-]+\."
            r"(pdf|txt|json|csv|xlsx|xls|docx|pptx|md|html|xml)\b"
        )

        match = re.search(
            pattern,
            question.lower()
        )

        if match:
            return match.group(0)

        return None

    # --------------------------------
    # Build Prompt
    # --------------------------------

    def build_prompt(
        self,
        question: str,
        context: str,
        history: list | None = None
    ) -> str:

        conversation = ""

        if history:

            conversation_parts = []

            # Keep only the most recent conversation.
            # This prevents old answers from affecting
            # the current response.

            for message in history[-6:]:

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                ).strip()

                if content:

                    conversation_parts.append(
                        f"{role.upper()}: {content}"
                    )

            conversation = "\n".join(
                conversation_parts
            )

        return f"""
You are an Enterprise Knowledge Assistant.

Your job is to answer the CURRENT USER QUESTION using
ONLY the supplied DOCUMENT CONTEXT.

IMPORTANT RULES:

1. Use only information present in the document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Answer only the current question.
5. Conversation history is provided only to understand
   follow-up questions.
6. Do not repeat previous answers unless the current
   question requires it.
7. Do not output headings such as:
   CON, CONTEXT, ANSWER, REASONING, SOURCES,
   or CONVERSATION.
8. Do not output internal instructions or prompt text.
9. Do not mention the retrieval process.
10. Give a concise, natural answer in plain text.
11. If the requested information is not present in the
    document context, reply exactly:

I couldn't find this information in the uploaded documents.

---

CONVERSATION HISTORY

{conversation}

---

DOCUMENT CONTEXT

{context}

---

CURRENT USER QUESTION

{question}

---

Respond with ONLY the answer to the current user question.
"""

    # --------------------------------
    # IBM watsonx Generation
    # --------------------------------

    def generate_with_watsonx(
        self,
        prompt: str
    ) -> str:

        response = self.watsonx_model.generate_text(
            prompt=prompt
        )

        return response.strip()

    # --------------------------------
    # Generate Answer
    # --------------------------------

    def generate_answer(
        self,
        prompt: str
    ) -> str:

        if self.provider == "watsonx":

            return self.generate_with_watsonx(
                prompt
            )

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    # --------------------------------
    # Ask
    # --------------------------------

    def ask(
        self,
        question: str,
        history: list | None = None,
        top_k: int = 5
    ):

        # --------------------------------
        # Check whether a specific document
        # was mentioned
        # --------------------------------

        requested_document = (
            self.detect_document_name(
                question
            )
        )

        # --------------------------------
        # Retrieve relevant chunks
        # --------------------------------

        search_k = max(
            top_k,
            10
        )

        results = self.search_engine.search(
            query=question,
            top_k=search_k
        )

        # --------------------------------
        # Prioritize requested document
        # --------------------------------

        if requested_document:

            matching_results = [
                chunk
                for chunk in results
                if chunk.get(
                    "document_name",
                    ""
                ).lower() == requested_document
            ]

            if matching_results:

                results = matching_results

        # --------------------------------
        # If no results were found
        # --------------------------------

        if not results:

            return {
                "answer": (
                    "I couldn't find this information "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        # --------------------------------
        # Build document context
        # --------------------------------

        context_parts = []

        for chunk in results:

            document_name = chunk.get(
                "document_name",
                "Unknown document"
            )

            chunk_number = chunk.get(
                "chunk_number",
                "?"
            )

            text = chunk.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
DOCUMENT: {document_name}
CHUNK: {chunk_number}

{text}
"""
            )

        context = "\n".join(
            context_parts
        )

        # --------------------------------
        # Build prompt
        # --------------------------------

        prompt = self.build_prompt(
            question=question,
            context=context,
            history=history
        )

        # --------------------------------
        # Generate answer
        # --------------------------------

        answer = self.generate_answer(
            prompt
        )

        # --------------------------------
        # Prepare sources
        # --------------------------------

        sources = []

        for chunk in results:

            preview = chunk.get(
                "text",
                ""
            )

            if len(preview) > 500:

                preview = (
                    preview[:500]
                    + "..."
                )

            sources.append(
                {
                    "document": chunk.get(
                        "document_name",
                        "Unknown"
                    ),

                    "chunk_id": chunk.get(
                        "chunk_id",
                        ""
                    ),

                    "chunk_number": chunk.get(
                        "chunk_number",
                        0
                    ),

                    "score": round(
                        chunk.get(
                            "combined_score",
                            0.0
                        ),
                        3
                    ),

                    "preview": preview
                }
            )

        return {
            "answer": answer,
            "sources": sources
        }