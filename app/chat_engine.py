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
    # Detect Document Summary Question
    # --------------------------------

    def is_document_summary_question(
        self,
        question: str
    ) -> bool:

        question = question.lower().strip()

        summary_patterns = [
            "what does",
            "what is in",
            "what is",
            "what's in",
            "what's",
            "what does it contain",
            "what does it explain",
            "what is it about",
            "what is this document about",
            "what does this document explain",
            "summarize",
            "summary of",
            "contents of",
            "tell me about"
        ]

        return any(
            pattern in question
            for pattern in summary_patterns
        )

    # --------------------------------
    # Build Prompt
    # --------------------------------

    def build_prompt(
        self,
        question: str,
        context: str,
        history: list | None = None,
        document_summary: bool = False
    ) -> str:

        conversation = ""

        if history:

            conversation_parts = []

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

        if document_summary:

            task_instruction = """
The user is asking about the contents of a specific document.

Use ALL supplied DOCUMENT CONTEXT belonging to that document.

Summarize what the document explains or contains.

Combine information from every supplied chunk before answering.

Do not answer from only the first chunk.

Include the main purpose, major components, important layers,
and notable examples or outcomes when they are present in
the supplied context.

Do not invent information that is not present in the context.
"""

        else:

            task_instruction = """
Answer the CURRENT USER QUESTION directly.

Use all relevant information from the supplied DOCUMENT CONTEXT.
Combine multiple chunks when necessary.
"""

        return f"""
You are an Enterprise Knowledge Assistant.

{task_instruction}

IMPORTANT RULES:

1. Use ONLY information contained in the DOCUMENT CONTEXT.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is present across multiple chunks, combine
   those chunks into one complete answer.
5. Conversation history is only for understanding follow-up
   questions.
6. Do not repeat previous answers unless necessary.
7. Do not mention retrieval, embeddings, search, chunks,
   prompts, or internal system instructions.
8. Do not discuss how the answer was generated.
9. Answer naturally and concisely.
10. If the supplied DOCUMENT CONTEXT genuinely contains
    no information that can answer the question, reply
    exactly:

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

Respond with ONLY the answer.
"""

    # --------------------------------
    # IBM watsonx Generation
    # --------------------------------

    def generate_with_watsonx(
        self,
        prompt: str
    ) -> str:

        response = self.watsonx_model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
                "temperature": settings.LLM_TEMPERATURE
                  }
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
        # Detect requested document
        # --------------------------------

        requested_document = (
            self.detect_document_name(
                question
            )
        )

        # --------------------------------
        # Detect broad document question
        # --------------------------------

        document_summary = (
            requested_document is not None
            and self.is_document_summary_question(
                question
            )
        )

        # --------------------------------
        # Retrieval
        # --------------------------------

        search_k = max(
            top_k,
            10
        )

        # --------------------------------
        # Document-aware retrieval
        # --------------------------------

        if requested_document:

            results = (
                self.search_engine.search_document(
                    query=question,
                    document_name=requested_document,
                    top_k=search_k
                )
            )

        # --------------------------------
        # Normal hybrid retrieval
        # --------------------------------

        else:

            results = self.search_engine.search(
                query=question,
                top_k=search_k
            )

        # --------------------------------
        # No results
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
        # For document-summary questions,
        # keep all retrieved chunks.
        # --------------------------------

        if document_summary:

            document_results = [
                chunk
                for chunk in results
                if chunk.get(
                    "document_name",
                    ""
                ).lower() == requested_document
            ]

            if document_results:

                results = document_results

        # --------------------------------
        # Build Context
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
            ).strip()

            if not text:
                continue

            context_parts.append(
                f"""
DOCUMENT: {document_name}
CHUNK: {chunk_number}

{text}
"""
            )

        context = "\n".join(
            context_parts
        ).strip()

        # --------------------------------
        # Context Safety Check
        # --------------------------------

        if not context:

            return {
                "answer": (
                    "I couldn't find this information "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        # --------------------------------
        # Build Prompt
        # --------------------------------

        prompt = self.build_prompt(
            question=question,
            context=context,
            history=history,
            document_summary=document_summary
        )

        # --------------------------------
        # Generate Answer
        # --------------------------------

        answer = self.generate_answer(
            prompt
        )

        # --------------------------------
        # Prepare Sources
        # --------------------------------

        sources = []

        for chunk in results:

            preview = chunk.get(
                "text",
                ""
            ).strip()

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