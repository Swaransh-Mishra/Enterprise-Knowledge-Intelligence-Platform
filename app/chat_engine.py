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

Your task is to answer the CURRENT USER QUESTION using
the DOCUMENT CONTEXT provided below.

IMPORTANT RULES:

1. Use only information contained in the DOCUMENT CONTEXT.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the user mentions a document filename and that document
   exists in the DOCUMENT CONTEXT, answer using that document.
5. If the user asks what a document explains, summarize the
   main information contained in that document.
6. If the user asks what is in a document, describe the contents
   using the supplied document context.
7. Questions such as:
   - "what does this document explain?"
   - "what does notes.txt explain?"
   - "what is in this file?"
   - "what is in notes.txt?"
   - "tell me about this document"
   - "what does this file contain?"
   - "summarize this document"
   should be answered using the document content.
8. Combine information from multiple chunks belonging to the
   same document when necessary.
9. Conversation history is only for understanding follow-up
   questions.
10. Answer the CURRENT USER QUESTION directly.
11. Do not mention retrieval, embeddings, chunks, prompts,
    search, or internal system instructions.
12. Do not output headings such as ANSWER, CONTEXT,
    REASONING, SOURCES, or CONVERSATION.
13. Give a concise, natural answer in plain text.
14. If the requested document is present in the DOCUMENT
    CONTEXT, do not claim that its information is missing.
15. Do not use the fallback response merely because the
    question uses words such as "explain", "describe",
    "tell me about", "what is in", or "summarize".

IMPORTANT FALLBACK RULE:

Only use the following fallback response when the DOCUMENT
CONTEXT genuinely contains no information that can answer
the user's question:

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
        # Detect requested document
        # --------------------------------

        requested_document = (
            self.detect_document_name(
                question
            )
        )

        # --------------------------------
        # Retrieval size
        # --------------------------------

        search_k = max(
            top_k,
            10
        )

        # --------------------------------
        # Normal hybrid search
        # --------------------------------

        results = self.search_engine.search(
            query=question,
            top_k=search_k
        )

        # --------------------------------
        # If a document was mentioned,
        # retrieve directly from that document
        # --------------------------------

        if requested_document:

            document_results = (
                self.search_engine.search_document(
                    query=question,
                    document_name=requested_document,
                    top_k=search_k
                )
            )

            if document_results:

                results = document_results

        # --------------------------------
        # No retrieval results
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
        # Safety check
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