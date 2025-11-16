from config import settings
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import gc

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = settings.HF_HUB_DISABLE_SYMLINKS_WARNING

CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
CORPUS_DIR = settings.CORPUS_DIR
CHUNKS_FILE = settings.CHUNKS_FILE
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL


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

    if not os.path.exists(CORPUS_DIR):
        print(f"❌ ПОМИЛКА: Директорія {CORPUS_DIR} не існує!")
        return []

    files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith('.txt')])

    if not files:
        print(f"❌ ПОМИЛКА: Немає .txt файлів у {CORPUS_DIR}!")
        return []

    print(f"✅ Знайдено {len(files)} файлів")

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

    print(f"Створено {len(chunks)} чанків")
    return chunks


def build_index(chunks):
    if not chunks:
        print("❌ Немає чанків!")
        return None

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Видаляємо стару несумісну базу
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Видалено стару колекцію")
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    print("Завантаження моделі (це займе час)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["chunk_text"] for c in chunks]

    # МАЛЕНЬКІ БАТЧІ - економія RAM
    batch_size = 8
    all_embeddings = []

    print(f"Створення embeddings ({len(texts)} текстів) по {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(emb.tolist())
        print(f"✓ {min(i + batch_size, len(texts))}/{len(texts)}")
        gc.collect()

    metadatas = [{
        "doc_id": c["doc_id"],
        "doc_name": c["doc_name"],
        "doc_path": c["doc_path"],
        "chunk_number": c["chunk_number"]
    } for c in chunks]

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Додавання до бази...")
    collection.add(
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"🎉 ГОТОВО! Індекс: {collection.count()} чанків")
    return collection


if __name__ == "__main__":
    print("=== СТВОРЕННЯ БАЗИ ===")
    chunks = create_chunks()
    build_index(chunks)
    print("=== ІНДЕКСАЦІЯ ЗАВЕРШЕНА ===")