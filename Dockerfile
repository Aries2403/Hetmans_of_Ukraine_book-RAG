FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py config.py ./
COPY src/ ./src/
COPY images/ ./images/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

RUN echo "✅ ChromaDB база скопійована" && \
    ls -la /app/chroma_db/ && \
    echo "✅ Фото скопійовано:" && \
    ls /app/images/hetmans/*.jpg /app/images/hetmans/*.png 2>/dev/null | wc -l

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ТІЛЬКИ Streamlit - БЕЗ індексації!
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]