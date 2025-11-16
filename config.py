from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # === КЛЮЧІ ===
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    # === ШЛЯХИ ===
    CORPUS_DIR: str = "/app/hetman_files"
    CHROMA_PATH: str = "/app/chroma_db"
    CHUNKS_FILE: str = "chunks.json"
    COLLECTION_NAME: str = "hetmans"

    # === МОДЕЛІ ===
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LLM_MODEL: str = "gpt-3.5-turbo"

    # === ПАРАМЕТРИ ===
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5

    # === ІНШЕ ===
    HF_HUB_DISABLE_SYMLINKS_WARNING: str = "1"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# === ЕКЗЕМПЛЯР ===
settings = Settings()