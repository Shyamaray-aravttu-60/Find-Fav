"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional; plain env vars work without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parents[2]


def _str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    p = Path(_str(name, default))
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class Settings:
    # Catalog / files
    products_path: Path
    frontend_dir: Path
    cache_dir: Path
    only_active: bool
    # Ollama
    ollama_url: str
    embed_model: str
    llm_model: str
    ollama_timeout: float
    llm_timeout: float
    # Ranking
    w_semantic: float
    w_keyword: float
    sem_floor: float  # cosine at/below which semantic relevance is 0
    sem_ceil: float  # cosine at/above which semantic relevance is 1
    min_score: float  # results scoring below this are dropped (-> empty state)
    sem_gate: float  # unrecognised queries need a plain-text cosine >= this to return anything
    # Server
    cors_origins: list

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            products_path=_path("PRODUCTS_PATH", "data/products.json"),
            frontend_dir=ROOT_DIR / "frontend",
            cache_dir=_path("CACHE_DIR", ".cache"),
            only_active=_bool("ONLY_ACTIVE", True),
            ollama_url=_str("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            embed_model=_str("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            llm_model=_str("OLLAMA_LLM_MODEL", "llama3.2"),
            ollama_timeout=_float("OLLAMA_TIMEOUT", 30),
            llm_timeout=_float("OLLAMA_LLM_TIMEOUT", 25),
            w_semantic=_float("WEIGHT_SEMANTIC", 0.6),
            w_keyword=_float("WEIGHT_KEYWORD", 0.4),
            sem_floor=_float("SEM_FLOOR", 0.45),
            sem_ceil=_float("SEM_CEIL", 0.75),
            min_score=_float("MIN_SCORE", 0.20),
            sem_gate=_float("SEM_GATE", 0.55),
            cors_origins=[o.strip() for o in _str("CORS_ORIGINS", "*").split(",") if o.strip()],
        )


settings = Settings.from_env()
