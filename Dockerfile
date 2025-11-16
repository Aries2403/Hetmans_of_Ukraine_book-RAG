FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY app.py ./
COPY config.py ./
COPY rag_hetmans.py ./
COPY src/ ./src/
COPY images ./images/

# Завантаження файлів у /app/data (НЕ у Volume!)
RUN pip install gdown && \
    mkdir -p /app/data/hetman_files && \
    gdown --id 1ThwXMEjv0MSZ6538DGaMsU2rwpi0j8vF -O /app/data/hetman_files.zip && \
    unzip /app/data/hetman_files.zip -d /app/data/hetman_files/ && \
    rm /app/data/hetman_files.zip && \
    echo "✅ Файли завантажено:" && ls -la /app/data/hetman_files/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Спочатку індексація (створить базу у Volume), потім Streamlit
ENTRYPOINT ["sh", "-c", "python rag_hetmans.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]