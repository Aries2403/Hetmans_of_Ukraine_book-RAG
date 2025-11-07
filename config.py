# config.py
from dotenv import load_dotenv
load_dotenv()  # ← Завантажує .env

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
import os

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")

    CORPUS_DIR: str = "hetman_files"
    CHUNKS_FILE: str = "chunks.json"
    CHROMA_PATH: str = "chroma_db"
    COLLECTION_NAME: str = "hetmans"

    CHUNK_SIZE: int = Field(500, ge=300, le=800)
    CHUNK_OVERLAP: int = Field(150, ge=100, le=300)
    TOP_K: int = Field(5, ge=1, le=10)

    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    LLM_MODEL: str = "gpt-3.5-turbo"

    HF_HUB_DISABLE_SYMLINKS_WARNING: str = "1"

    @field_validator("CORPUS_DIR")
    def check_dir_exists(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Папка не існує: {v}")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()