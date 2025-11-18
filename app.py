import streamlit as st
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import settings

# === Налаштування ===
CACHE_FILE = "cache.json"
CHROMA_PATH = settings.CHROMA_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
TOP_K = settings.TOP_K

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}

# === CSS ДИЗАЙН (БЕЗПЕЧНА ВЕРСІЯ) ===
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0057b7 0%, #004c99 30%, #ffd700 100%) !important;
    }

    h1 {
        color: #ffd700 !important;
        text-align: center;
        font-size: 3rem !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        color: #ffffff;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# === ОБКЛАДИНКА + ЗАГОЛОВОК В ОДНОМУ КОНТЕЙНЕРІ ===
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists("book_cover.jpg"):
        st.image("book_cover.jpg", width=100)
with col2:
    st.markdown("<h1>🇺🇦 Усі Гетьмани України</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p class="subtitle">RAG-система для семантичного пошуку по книзі "Усі гетьмани України"<br>авт. Реент О.П., Коляда І.А. · 2008</p>
    """, unsafe_allow_html=True)


# === УНІКАЛЬНІСТЬ ===
def deduplicate_by_id(results):
    seen = set()
    unique = []
    for meta, dist, doc in zip(results["metadatas"][0], results["distances"][0], results["documents"][0]):
        chunk_id = f"{meta['doc_path']}#{meta['chunk_number']}"
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append((meta, dist, doc))
    return unique


# === LLM ===
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


# === RAG ===
def rag_query(query):
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        return "⚠️ База даних ще не готова. Спробуйте пізніше.", []

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


# === ІСТОРІЯ ===
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Джерела"):
                for s in msg["sources"]:
                    st.write(s)
        if "image" in msg and msg["image"]:
            st.image(msg["image"], width=200)

# === ВВІД ===
if query := st.chat_input("Запитай про гетьмана або напиши фото Ім'я Прізвище"):
    query = query.strip()
    if not query:
        st.warning("Введіть запитання.")
        st.stop()

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
                    try:
                        st.image(image_path, caption=name, width=200)
                        answer = f"Ось фото: **{name}**"
                    except Exception as e:
                        answer = f"⚠️ Фото тимчасово недоступне. Помилка: {type(e).__name__}"
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

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "image": image_path
            })