"""
Enterprise RAG Chat Engine.

Uses hybrid search, conversation memory,
document-aware retrieval, and a configurable
LLM provider.
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
    # Find Recent Document
    # --------------------------------

    def find_recent_document(
        self,
        history: list | None = None
    ):

        if not history:

            return None

        for message in reversed(
            history[-10:]
        ):

            content = message.get(
                "content",
                ""
            )

            if not isinstance(
                content,
                str
            ):

                continue

            document = (
                self.detect_document_name(
                    content
                )
            )

            if document:

                return document

        return None

    # --------------------------------
    # Detect Follow-up Question
    # --------------------------------

    def is_follow_up_question(
        self,
        question: str
    ) -> bool:

        question = (
            question
            .strip()
            .lower()
        )

        follow_up_patterns = [

            "what does it do",
            "what does that do",
            "what does this do",

            "what does it mean",
            "what does that mean",
            "what does this mean",

            "tell me more",
            "tell me more about it",
            "tell me more about that",
            "tell me more about this",

            "more about it",
            "more about that",
            "more about this",

            "explain it",
            "explain that",
            "explain this",
            "explain more",

            "how does it work",
            "how does that work",
            "how does this work",

            "why is it important",
            "why does it matter",

            "what about it",
            "what about that",
            "what about this"
        ]

        return any(
            pattern in question
            for pattern in follow_up_patterns
        )

    # --------------------------------
    # Get Previous User Question
    # --------------------------------

    def get_previous_user_question(
        self,
        history: list | None = None
    ):

        if not history:

            return None

        for message in reversed(history):

            if message.get(
                "role"
            ) != "user":

                continue

            content = message.get(
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                content = content.strip()

                if content:

                    return content

        return None

    # --------------------------------
    # Get Previous Assistant Answer
    # --------------------------------

    def get_previous_answer(
        self,
        history: list | None = None
    ):

        if not history:

            return None

        for message in reversed(history):

            if message.get(
                "role"
            ) != "assistant":

                continue

            content = message.get(
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                content = content.strip()

                if content:

                    return content

        return None

    # --------------------------------
    # Extract Main Subject
    # --------------------------------

    def extract_follow_up_entity(
        self,
        answer: str
    ):

        if not answer:

            return None

        known_entities = [

            "Perception Layer",
            "Coordination Layer",
            "Strategic Layer",
            "Interaction Layer",

            "Light Gradient Boosting Machine (LightGBM)",
            "Light Gradient Boosting Machine",
            "LightGBM",

            "XGBoost",

            "Random Forest",
            "RandomForest",

            "CatBoost",

            "Logistic Regression",

            "Decision Tree"
        ]

        for entity in known_entities:

            if (
                entity.lower()
                in answer.lower()
            ):

                return entity

        return None

    # --------------------------------
    # Resolve Follow-up
    # --------------------------------

    def resolve_follow_up(
        self,
        question: str,
        history: list | None = None
    ):

        if not history:

            return question

        if not self.is_follow_up_question(
            question
        ):

            return question

        previous_question = (
            self.get_previous_user_question(
                history
            )
        )

        previous_answer = (
            self.get_previous_answer(
                history
            )
        )

        document = (
            self.find_recent_document(
                history
            )
        )

        entity = None

        if previous_answer:

            entity = (
                self.extract_follow_up_entity(
                    previous_answer
                )
            )

        # --------------------------------
        # Follow-up with document + entity
        # --------------------------------

        if document and entity:

            return (
                f"{question}. "
                f"The previous topic was {entity} "
                f"in {document}. "
                f"Find additional information about "
                f"{entity} from {document}."
            )

        # --------------------------------
        # Follow-up with entity only
        # --------------------------------

        if entity:

            return (
                f"{question}. "
                f"The previous topic was {entity}. "
                f"Find additional information about "
                f"{entity}."
            )

        # --------------------------------
        # Follow-up with document only
        # --------------------------------

        if document:

            return (
                f"{question}. "
                f"The conversation is about {document}. "
                f"Find additional relevant information "
                f"from {document}."
            )

        # --------------------------------
        # Final fallback
        # --------------------------------

        if previous_question:

            return (
                f"Previous question: "
                f"{previous_question}. "
                f"Follow-up: {question}."
            )

        return question

    # --------------------------------
    # Detect Document Summary Question
    # --------------------------------

    def is_document_summary_question(
        self,
        question: str
    ) -> bool:

        question = (
            question
            .lower()
            .strip()
        )

        patterns = [

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
            for pattern in patterns
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

            for message in history[-8:]:

                role = message.get(
                    "role",
                    "user"
                )

                content = message.get(
                    "content",
                    ""
                )

                if not isinstance(
                    content,
                    str
                ):

                    continue

                content = content.strip()

                if content:

                    conversation_parts.append(
                        f"{role.upper()}: {content}"
                    )

            conversation = "\n".join(
                conversation_parts
            )

        if document_summary:

            task_instruction = """
The user is asking about a specific document.

Use the supplied DOCUMENT CONTEXT.

Combine relevant information from the supplied
sections before answering.

Give a clear summary of what the document explains.

Include important components, examples,
demonstrations, and outcomes when available.

Do not invent information.
"""

        else:

            task_instruction = """
Answer the CURRENT USER QUESTION directly.

Use the supplied DOCUMENT CONTEXT.

If the question is a follow-up, use the conversation
history to understand what "it", "that", or "this"
refers to.

If the user asks for more information, provide
additional relevant information from the document.

Do not simply repeat the previous answer.
"""

        return f"""
You are an Enterprise Knowledge Assistant.

{task_instruction}

IMPORTANT RULES:

1. Use ONLY information contained in the
DOCUMENT CONTEXT.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Conversation history is used only to understand
follow-up references.

5. When answering a follow-up, identify the subject
from the previous conversation.

6. When the user asks for more information, provide
additional information from the document.

7. Avoid repeating the previous answer unless
necessary for clarity.

8. Do not mention retrieval, embeddings, search,
chunks, prompts, or internal instructions.

9. Do not discuss how the answer was generated.

10. Answer naturally and concisely.

11. If the DOCUMENT CONTEXT does not contain
information that can answer the question, reply
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

        response = (
            self.watsonx_model.generate_text(
                prompt=prompt,
                params={
                    "max_new_tokens":
                        settings.LLM_MAX_NEW_TOKENS,
                    "temperature":
                        settings.LLM_TEMPERATURE
                }
            )
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
            f"Unsupported LLM provider: "
            f"{self.provider}"
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
        # Detect Whether This Is a Follow-up
        # --------------------------------

        follow_up = (
            self.is_follow_up_question(
                question
            )
        )

        # --------------------------------
        # Resolve Follow-up
        # --------------------------------

        retrieval_question = (
            self.resolve_follow_up(
                question=question,
                history=history
            )
        )

        # --------------------------------
        # Detect Explicit Document
        # --------------------------------

        requested_document = (
            self.detect_document_name(
                retrieval_question
            )
        )

        # --------------------------------
        # Use Recent Document ONLY
        # for Follow-up Questions
        # --------------------------------

        if (
            not requested_document
            and follow_up
        ):

            requested_document = (
                self.find_recent_document(
                    history
                )
            )

        # --------------------------------
        # Detect Summary Question
        # --------------------------------

        document_summary = (
            requested_document is not None
            and self.is_document_summary_question(
                question
            )
            and not follow_up
        )

        # --------------------------------
        # Retrieval Size
        # --------------------------------

        search_k = max(
            top_k,
            10
        )

        # --------------------------------
        # Search
        # --------------------------------

        if requested_document:

            results = (
                self.search_engine.search_document(
                    query=retrieval_question,
                    document_name=requested_document,
                    top_k=search_k
                )
            )

        else:

            results = (
                self.search_engine.search(
                    query=retrieval_question,
                    top_k=search_k
                )
            )

        # --------------------------------
        # No Results
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
        # Keep Requested Document
        # --------------------------------

        if requested_document:

            document_results = [

                chunk

                for chunk in results

                if chunk.get(
                    "document_name",
                    ""
                ).lower()
                == requested_document.lower()
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

        context = (
            "\n".join(
                context_parts
            )
            .strip()
        )

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

        for chunk in results[:top_k]:

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