from src.rag.retriever import Retriever
from src.rag.context_builder import build_context
from src.rag.prompt_builder import SYSTEM_PROMPT, build_prompt


retriever = Retriever()

question = "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟"

results = retriever.search(question, k=3)

context = build_context(results)

prompt = build_prompt(
    question=question,
    context=context,
)

print("=" * 80)
print("SYSTEM PROMPT")
print("=" * 80)
print(SYSTEM_PROMPT)

print("\n" + "=" * 80)
print("USER PROMPT")
print("=" * 80)
print(prompt)