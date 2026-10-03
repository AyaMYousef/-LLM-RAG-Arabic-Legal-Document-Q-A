from __future__ import annotations


SYSTEM_PROMPT = """You are an assistant for the Egyptian Civil Code.

Answer the user's question using only the legal context provided to you.

Rules:
1. Do not invent or assume legal provisions that are not present in the context.
2. If the provided context does not contain enough information to answer, say that the retrieved context is insufficient.
3. Cite the relevant Egyptian Civil Code article number(s) in your answer.
4. Prefer the Arabic legal text when answering Arabic questions.
5. You may use the English text to clarify the meaning when necessary.
6. Distinguish clearly between what is stated in the legal text and any explanation you provide.
7. Do not provide unsupported legal conclusions.
"""


def build_prompt(question: str, context: str) -> str:
    """Build the user prompt for grounded legal question answering."""

    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not context.strip():
        raise ValueError("Context must not be empty.")

    return f"""Legal context:

{context}

User question:

{question}

Answer the question using only the legal context above.
Include the relevant article number(s) in your answer.
"""