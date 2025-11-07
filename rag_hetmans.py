# rag_hetmans.py
from config import settings
from openai import OpenAI
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer


# === Попередження Hugging Face (вже в .env, але на всяк випадок) ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = settings.HF_HUB_DISABLE_SYMLINKS_WARNING

# === Використовуємо налаштування ===
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
TOP_K = settings.TOP_K
CORPUS_DIR = settings.CORPUS_DIR
CHUNKS_FILE = settings.CHUNKS_FILE
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
LLM_MODEL = settings.LLM_MODEL


# === Крок 2: Чанки ===
def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def create_chunks():
    if os.path.exists(CHUNKS_FILE):
        print(f"Чанки вже є: {CHUNKS_FILE}")
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    chunks = []
    files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')])

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(CORPUS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        lines = text.strip().split('\n', 1)
        doc_name = lines[0].strip()
        doc_text = lines[1] if len(lines) > 1 else ""

        doc_id = f"hetman_{idx:02d}"

        chunk_list = split_into_chunks(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(chunk_list, 1):
            chunks.append({
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_path": filepath,
                "chunk_number": chunk_idx,
                "chunk_text": chunk.strip()
            })

    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Створено {len(chunks)} чанків → {CHUNKS_FILE}")
    return chunks


# === Крок 3: Індекс ===
def build_index(chunks):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0:
        print("Індекс вже існує. Пропускаємо.")
        return collection

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    metadatas = [{
        "doc_id": c["doc_id"],
        "doc_name": c["doc_name"],
        "doc_path": c["doc_path"],
        "chunk_number": c["chunk_number"]
    } for c in chunks]

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Індекс створено: {collection.count()} чанків")
    return collection


# === УНІКАЛЬНІСТЬ ЗА ID (Спосіб 2) ===
def deduplicate_by_id(results):
    seen = set()
    unique = []
    for meta, dist, doc in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
        chunk_id = f"{meta['doc_path']}#{meta['chunk_number']}"
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append((meta, dist, doc))
    return unique


# === LLM через OpenAI (v1.0+) ===
def generate_response(query, context_chunks):
    context = "\n".join([f"[{i + 1}] {chunk}" for i, (_, _, chunk) in enumerate(context_chunks)])

    prompt = f"""Ти — експерт з історії України. Використовуй ТІЛЬКИ наведений контекст.
Відповідай коротко (1-3 речення), з номерами джерел [1], [2] тощо.
Якщо не впевнений — скажи: "Немає точної відповіді".

Запит: {query}

Контекст:
{context}

Відповідь:"""

    try:
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # ↓ для точності
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Помилка LLM: {e}"


# === Основний цикл ===
def main():
    print("Завантаження чанків...")
    chunks = create_chunks()

    print("Перевірка індексу...")
    collection = build_index(chunks)

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\n" + "=" * 60)
    print("RAG-система з OpenAI готова! (q — вихід)")
    print("=" * 60)

    while True:
        query = input("\nЗапит: ").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            print("До зустрічі!")
            break
        if len(query) < 3:
            print("Запит занадто короткий.")
            continue

        # Пошук
        query_emb = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_emb,
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        # Унікальність за ID
        unique_results = deduplicate_by_id(results)
        top_chunks = unique_results[:3]  # топ-3 унікальних

        # LLM
        response = generate_response(query, top_chunks)

        # Джерела
        sources = []
        for i, (meta, _, _) in enumerate(top_chunks, 1):
            sources.append(f"[{i}] {meta['doc_name']} (чанк {meta['chunk_number']})")

        print("\nВідповідь:")
        print(response)
        print("\nДжерела:")
        print(" | ".join(sources))


if __name__ == "__main__":
    main()