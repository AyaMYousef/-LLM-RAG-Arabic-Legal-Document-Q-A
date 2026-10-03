from src.rag.retriever import Retriever
from src.rag.context_builder import build_context


retriever = Retriever()

question = "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟"

results = retriever.search(question, k=3)

context = build_context(results)

print("=" * 80)
print("RETRIEVED CONTEXT")
print("=" * 80)
print(context)