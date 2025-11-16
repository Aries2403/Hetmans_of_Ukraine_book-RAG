FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY app.py ./
COPY config.py ./
COPY rag_hetmans.py ./
COPY src/ ./src/
COPY images ./images/

# Просто копіюємо файли з репозиторію
COPY data/ ./data/

# Перевірка
RUN echo "✅ Файли скопійовано:" && \
    ls -la /app/data/hetman_files/ && \
    echo "Кількість .txt файлів:" && \
    find /app/data/hetman_files -name "*.txt" | wc -l

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["sh", "-c", "python rag_hetmans.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]