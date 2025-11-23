from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
    base_dir: Path
    uploads_dir: Path
    storage_dir: Path
    vector_store_path: Path
    log_file: Path
    chunk_size: int
    chunk_overlap: int
    chunk_size_ar: int
    chunk_overlap_ar: int
    groq_api_key: str
    groq_model_name: str
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_path: Path | None = None
    embedding_cache_dir: Path | None = None
    whisper_model_name: str = "medium"
    whisper_device: str = "auto"
    vector_store_backend: str = "auto"
    environment: str = "development"
    reranker_model_name: str | None = "BAAI/bge-reranker-large"
    reranker_device: str = "auto"
    reranker_candidate_limit: int = 18
    enable_hybrid_retrieval: bool = True
    bm25_index_path: Path | None = None

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "AppConfig":
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        base_dir = Path(os.getenv("BASE_DIR", Path.cwd()))
        uploads_dir = base_dir / os.getenv("UPLOADS_DIR", "uploads")
        storage_dir = base_dir / os.getenv("STORAGE_DIR", "storage")
        vector_store_path = base_dir / os.getenv("VECTOR_STORE_PATH", "storage/vector_store")
        log_file = base_dir / os.getenv("LOG_FILE", "logs/app.log")
        embedding_cache_dir_env = os.getenv("EMBEDDING_CACHE_DIR", "storage/models")
        embedding_cache_dir = (
            base_dir / embedding_cache_dir_env if embedding_cache_dir_env else None
        )
        embedding_model_path_env = os.getenv("EMBEDDING_MODEL_PATH")
        embedding_model_path = (
            (base_dir / embedding_model_path_env).resolve()
            if embedding_model_path_env and not os.path.isabs(embedding_model_path_env)
            else (Path(embedding_model_path_env).resolve() if embedding_model_path_env else None)
        )

        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        chunk_size_ar = int(os.getenv("CHUNK_SIZE_AR", "1200"))
        chunk_overlap_ar = int(os.getenv("CHUNK_OVERLAP_AR", "300"))

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please populate it in the .env file.")

        # Using Llama 3.1 70B for best RAG quality, or Mixtral for speed
        groq_model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")
        embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
        )
        whisper_model_name = os.getenv("WHISPER_MODEL_NAME", "medium")
        whisper_device = os.getenv("WHISPER_DEVICE", "auto").lower()
        vector_store_backend = os.getenv("VECTOR_STORE_BACKEND", "auto").lower()
        environment = os.getenv("ENVIRONMENT", "development")
        reranker_model_name_env = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
        reranker_model_name = reranker_model_name_env.strip() or None
        reranker_device = os.getenv("RERANKER_DEVICE", "auto").lower()
        reranker_candidate_limit = int(os.getenv("RERANKER_CANDIDATE_LIMIT", "18"))
        enable_hybrid_retrieval = os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() in {"1", "true", "yes"}
        bm25_index_path_env = os.getenv("BM25_INDEX_PATH", "storage/bm25_index.json")
        bm25_index_path = (
            base_dir / bm25_index_path_env if not os.path.isabs(bm25_index_path_env) else Path(bm25_index_path_env)
        )

        config = cls(
            base_dir=base_dir,
            uploads_dir=uploads_dir,
            storage_dir=storage_dir,
            vector_store_path=vector_store_path,
            log_file=log_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_size_ar=chunk_size_ar,
            chunk_overlap_ar=chunk_overlap_ar,
            groq_api_key=groq_api_key,
            groq_model_name=groq_model_name,
            embedding_model_name=embedding_model_name,
            embedding_model_path=embedding_model_path,
            embedding_cache_dir=embedding_cache_dir,
            whisper_model_name=whisper_model_name,
            whisper_device=whisper_device,
            vector_store_backend=vector_store_backend,
            environment=environment,
            reranker_model_name=reranker_model_name,
            reranker_device=reranker_device,
            reranker_candidate_limit=reranker_candidate_limit,
            enable_hybrid_retrieval=enable_hybrid_retrieval,
            bm25_index_path=bm25_index_path,
        )
        config.ensure_directories()
        return config

    def ensure_directories(self) -> None:
        required_paths = {
            self.uploads_dir,
            self.storage_dir,
            self.vector_store_path.parent,
            self.log_file.parent,
        }
        if self.embedding_cache_dir:
            required_paths.add(self.embedding_cache_dir)
        if self.embedding_model_path:
            required_paths.add(self.embedding_model_path.parent)
        if self.bm25_index_path:
            required_paths.add(Path(self.bm25_index_path).parent)
        for path in required_paths:
            path.mkdir(parents=True, exist_ok=True)
