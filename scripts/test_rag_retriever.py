from src.rag.retriever import Retriever

retriever = Retriever()

question = "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟"

results = retriever.search(question, k=1)

result = results[0]
metadata = result["metadata"]

print(f"score={result['score']:.4f}")
print(f"article={metadata['article_number']}")
print(f"citation={metadata['citation']}")
print(f"source_page={metadata['source_page']}")
print()
print("TEXT EN:")
print(metadata.get("text_en"))
print()
print("TEXT AR:")
print(metadata.get("text_ar"))