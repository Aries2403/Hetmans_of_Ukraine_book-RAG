# evaluate_retrieval.py
import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from config import settings  # ← БЕРЕМО З config.py

# === Налаштування ===
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
TOP_K = 3
EMBEDDING_MODEL = settings.EMBEDDING_MODEL  # ← multilingual-e5-large
TEST_QUERIES_FILE = "test_queries.json"

# === Завантаження ===
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)
model = SentenceTransformer(EMBEDDING_MODEL)

# === Тестові запити ===
test_queries = [
    {
        "query": "Хто вважається засновником Запорозької Січі?",
        "expected_doc": "Байда-Вишневецький.txt",
        "topic": "Дмитро Вишневецький"
    },
    {
        "query": "Який гетьман очолив повстання 1621 року під Хотином?",
        "expected_doc": "Конашевич-Сагайдачний.txt",
        "topic": "Петро Сагайдачний"
    },
    {
        "query": "Хто підписав Переяславську угоду з Москвою?",
        "expected_doc": "Хмельницький.txt",
        "topic": "Богдан Хмельницький"
    },
    {
        "query": "Який гетьман уклав союз із Карлом XII проти Петра I?",
        "expected_doc": "Мазепа.txt",
        "topic": "Іван Мазепа"
    },
    {
        "query": "Хто був останнім гетьманом України?",
        "expected_doc": "Розумовський.txt",
        "topic": "Кирило Розумовський"
    },
    {
        "query": "Чи був син Богдана Хмельницього гетьманом?",
        "expected_doc": "Хмельницький(син).txt",
        "topic": "Юрій-Гедеон Хмельницький(син)"
    },
    {
        "query": "Що хорошого зробив Мазепа для України?",
        "expected_doc": "Мазепа.txt",
        "topic": "Іван Мазепа"
    },
    {
        "query": "Хто був першим гетьманом України??",
        "expected_doc": "Байда-Вишневецький.txt",
        "topic": "Дмитро Байда-Вишневецький"
    }
]

# === Унікальність за ID ===
def deduplicate_by_id(results):
    seen = set()
    unique = []
    for meta, dist, doc in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
        chunk_id = f"{meta['doc_path']}#{meta['chunk_number']}"
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append((meta, dist, doc))
    return unique


# === Оцінка ===
hits = 0
results_list = []

print("Запуск оцінки якості ретрівалу з моделлю multilingual-e5-large...\n")

for case in test_queries:
    query = case["query"]
    expected = case["expected_doc"]

    query_emb = model.encode([query]).tolist()
    res = collection.query(query_embeddings=query_emb, n_results=TOP_K, include=["metadatas", "distances", "documents"])

    unique_res = deduplicate_by_id(res)
    top_metas = [m[0] for m in unique_res[:TOP_K]]

    found = any(os.path.basename(m["doc_path"]) == expected for m in top_metas)
    hits += int(found)

    top_docs = [os.path.basename(m["doc_path"]) for m in top_metas]

    results_list.append({
        "query": query,
        "expected": expected,
        "found": top_docs,
        "hit": int(found)
    })

# === Вивід ===
print("=" * 60)
print("ОЦІНКА ЯКОСТІ РЕТРІВАЛУ (модель: multilingual-e5-large)")
print("=" * 60)

for r in results_list:
    status = "ПРАВИЛЬНО" if r["hit"] else "ПОМИЛКА"
    print(f"{status} {r['query']}")
    print(f"   Очікувано: {r['expected']}")
    print(f"   Знайдено: {', '.join(r['found'])}\n")

print(f"Середній Hit@{TOP_K}: {hits / len(test_queries):.2f} ({hits}/{len(test_queries)})")

# === Пояснення про нормалізацію ===
print("\n" + "=" * 60)
print("ПРО НОРМАЛІЗАЦІЮ ВЕКТОРІВ")
print("=" * 60)
print(f"• Модель: {EMBEDDING_MODEL}")
print("• Вектори вже L2-нормалізовані (довжина ≈ 1.0)")
print("• ChromaDB використовує cosine similarity за замовчуванням")
print("• Ручна нормалізація НЕ застосовується")
print("• Перевірка:")
sample_emb = model.encode("тест")
print(f"   Довжина вектора: {np.linalg.norm(sample_emb):.6f} ≈ 1.0")