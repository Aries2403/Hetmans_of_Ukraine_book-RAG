# app.py — ФІНАЛЬНА ВЕРСІЯ (логіка з rag_hetmans.py)
import streamlit as st
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import settings

# === Налаштування (з config.py) ===
CACHE_FILE = "cache.json"
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
TOP_K = settings.TOP_K

# === Кеш ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}

# === УНІКАЛЬНІСТЬ ЗА ID (як у rag_hetmans.py) ===
def deduplicate_by_id(results):
    seen = set()
    unique = []
    for meta, dist, doc in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
        chunk_id = f"{meta['doc_path']}#{meta['chunk_number']}"
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append((meta, dist, doc))
    return unique

# === LLM (точно як у rag_hetmans.py) ===
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
            temperature=0.2,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Помилка LLM: {e}"

# === RAG-запит (як у rag_hetmans.py) ===
def rag_query(query):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    unique_results = deduplicate_by_id(results)
    top_chunks = unique_results[:3]

    response = generate_response(query, top_chunks)

    sources = []
    for i, (meta, _, _) in enumerate(top_chunks, 1):
        sources.append(f"[{i}] {meta['doc_name']} (чанк {meta['chunk_number']})")

    return response, sources

# === GUI ===
st.set_page_config(page_title="Гетьмани України", layout="centered")
st.title("Гетьмани України — RAG")
st.write("Запитуй — відповідаю **тільки за книгою**! Використовуй `фото Ім'я Прізвище` для фото.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# === Відображення історії ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Джерела"):
                for s in msg["sources"]:
                    st.write(s)
        if "image" in msg and msg["image"]:
            st.image(msg["image"], width=200)

# === Ввід ===
if query := st.chat_input("Введіть запитання або команду..."):
    query = query.strip()
    if not query:
        st.warning("Введіть запитання.")
        st.stop()

    # === Зберігаємо запит користувача ===
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Шукаю..."):
            answer = ""
            sources = []
            image_path = None

            # === КОМАНДА: фото Ім'я Прізвище ===
            if query.lower().startswith("фото "):
                name = query[5:].strip()
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = f"images/hetmans/{name}{ext}"
                    if os.path.exists(candidate):
                        image_path = candidate
                        break

                if image_path:
                    st.image(image_path, caption=name, width=200)
                    answer = f"Ось фото: **{name}**"
                else:
                    answer = "Фото не знайдено."

            # === RAG-запит ===
            else:
                if query in cache:
                    answer = cache[query]["answer"]
                    sources = cache[query]["sources"]
                else:
                    answer, sources = rag_query(query)
                    cache[query] = {"answer": answer, "sources": sources}
                    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)

                st.markdown(answer)
                with st.expander("Джерела"):
                    for s in sources:
                        st.write(s)

            # === Зберігаємо відповідь ===
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "image": image_path  # None, якщо не було фото
            })