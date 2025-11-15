FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY app.py ./
COPY config.py ./
COPY rag_hetmans.py ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

# Створюємо папку для Volume
RUN mkdir -p /app/hetman_files

# АВТОМАТИЧНЕ ЗАВАНТАЖЕННЯ ДАНИХ (1.5 ГБ) ТА РОЗПАКУВАННЯ
RUN wget -O /app/hetman_files/hetman_files.zip 'https://drive.google.com/uc?export=download&id=1ThwXMEjv0MSZ6538DGaMsU2rwpi0j8vF' && \
    unzip /app/hetman_files/hetman_files.zip -d /app/hetman_files/ && \
    rm /app/hetman_files/hetman_files.zip

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]