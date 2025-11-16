# rag_hetmans.py — ТІЛЬКИ індексація ( для диплою)
from config import settings
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# === Налаштування ===
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = settings.HF_HUB_DISABLE_SYMLINKS_WARNING

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
CORPUS_DIR = settings.CORPUS_DIR
CHUNKS_FILE = settings.CHUNKS_FILE
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL


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

    # ПЕРЕВІРКА: чи існує директорія
    if not os.path.exists(CORPUS_DIR):
        print(f"❌ ПОМИЛКА: Директорія {CORPUS_DIR} не існує!")
        return []

    files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')])

    # ПЕРЕВІРКА: чи є .txt файли
    if not files:
        print(f"❌ ПОМИЛКА: Немає .txt файлів у {CORPUS_DIR}!")
        print(f"Вміст директорії: {os.listdir(CORPUS_DIR)}")
        return []

    print(f"✅ Знайдено {len(files)} файлів: {files[:3]}...")

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
    # ПЕРЕВІРКА: чи є чанки
    if not chunks:
        print("❌ ПОМИЛКА: Немає чанків для індексації!")
        return None

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0:
        print(f"✅ Індекс уже існує: {collection.count()} чанків. Пропускаємо.")
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


# === ЗАПУСК ТІЛЬКИ ДЛЯ ІНДЕКСАЦІЇ (render) ===
if __name__ == "__main__":
    print("Запуск індексації для Render...")
    chunks = create_chunks()
    build_index(chunks)
    print("Індексація завершена. Готово до запуску app.py")