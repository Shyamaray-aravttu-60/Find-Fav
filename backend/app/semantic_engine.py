"""Embedding-based relevance: Ollama embeddings + cosine similarity, with an on-disk cache."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .catalog import Product
from .ollama_client import OllamaClient, OllamaError

log = logging.getLogger("recommender.semantic")

BATCH_SIZE = 16


def _unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity becomes a plain dot product."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class SemanticEngine:
    def __init__(self, products: list, client: OllamaClient, cache_dir: Path):
        self.products = products
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.matrix: Optional[np.ndarray] = None  # (n_products, dim), unit rows
        self._build_lock = threading.Lock()
        self._query_cache: dict = {}
        # nomic-embed-text was trained with task prefixes; other models don't use them.
        nomic = "nomic" in client.embed_model.lower()
        self.doc_prefix = "search_document: " if nomic else ""
        self.query_prefix = "search_query: " if nomic else ""

    @property
    def ready(self) -> bool:
        return self.matrix is not None

    # ------------------------------------------------------------------ index building
    def _fingerprint(self, texts: list) -> str:
        h = hashlib.sha1(self.client.embed_model.encode())
        for p, t in zip(self.products, texts):
            h.update(f"\n{p.product_id}\t{t}".encode())
        return h.hexdigest()

    def _cache_path(self) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.client.embed_model)
        return self.cache_dir / f"embeddings_{safe}.npz"

    def ensure_ready(self, block: bool = True) -> bool:
        """Load or build the product embedding matrix.  ``block=False`` returns immediately
        (False) if another thread is already building it - used on the request path."""
        if self.ready:
            return True
        if not self._build_lock.acquire(blocking=block):
            return False
        try:
            if self.ready:
                return True
            texts = [self.doc_prefix + p.search_text() for p in self.products]
            fingerprint = self._fingerprint(texts)
            cache = self._cache_path()

            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as npz:
                        if str(npz["fingerprint"]) == fingerprint:
                            self.matrix = npz["vectors"]
                            log.info("Loaded %d cached embeddings from %s", len(self.matrix), cache)
                            return True
                except (OSError, ValueError, KeyError):
                    log.warning("Embedding cache unreadable, rebuilding")

            if self.client.cooling_down:
                return False
            try:
                log.info("Embedding %d products with %s ...", len(texts), self.client.embed_model)
                chunks = [self.client.embed(texts[i:i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)]
                matrix = _unit(np.vstack(chunks).astype(np.float32))
            except OllamaError:
                return False

            self.matrix = matrix
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(cache, vectors=matrix, fingerprint=np.array(fingerprint))
            except OSError:
                log.warning("Could not write embedding cache", exc_info=True)
            return True
        finally:
            self._build_lock.release()

    # ------------------------------------------------------------------ querying
    def similarities(self, query_text: str) -> Optional[np.ndarray]:
        """Cosine similarity of the query against every product, or None if unavailable."""
        if not self.ensure_ready(block=False):
            return None
        vec = self._query_cache.get(query_text)
        if vec is None:
            try:
                vec = _unit(self.client.embed([self.query_prefix + query_text]))[0]
            except OllamaError:
                return None
            if len(self._query_cache) > 256:
                self._query_cache.clear()
            self._query_cache[query_text] = vec
        if vec.shape[0] != self.matrix.shape[1]:
            return None  # embedding model changed under a stale cache
        return self.matrix @ vec
