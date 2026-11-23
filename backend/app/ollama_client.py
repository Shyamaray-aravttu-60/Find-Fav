"""Thin, defensive wrapper around the Ollama HTTP API (embeddings + JSON chat).

Every failure is converted to ``OllamaError`` and starts a short cooldown, so when Ollama is
not running the app answers instantly from the keyword engine instead of hanging on timeouts.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import numpy as np
import requests

log = logging.getLogger("recommender.ollama")

COOLDOWN_SECONDS = 20.0


class OllamaError(RuntimeError):
    pass


def _has_model(installed: list, wanted: str) -> bool:
    """'llama3.2' matches an installed 'llama3.2:latest'; 'llama3.2:3b' must match exactly."""
    if ":" in wanted:
        return wanted in installed
    return any(name == wanted or name.split(":")[0] == wanted for name in installed)


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, llm_model: str,
                 timeout: float = 30, llm_timeout: float = 25):
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.timeout = timeout
        self.llm_timeout = llm_timeout
        self.last_error: Optional[str] = None
        self._down_until = 0.0
        self._lock = threading.Lock()
        self._status_cache: tuple = (0.0, {})

    # ------------------------------------------------------------------ plumbing
    @property
    def cooling_down(self) -> bool:
        return time.monotonic() < self._down_until

    def _fail(self, message: str) -> OllamaError:
        with self._lock:
            self.last_error = message
            self._down_until = time.monotonic() + COOLDOWN_SECONDS
        log.warning("Ollama call failed: %s", message)
        return OllamaError(message)

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        if self.cooling_down:
            raise OllamaError(self.last_error or "Ollama unavailable (cooling down)")
        try:
            resp = requests.post(f"{self.base_url}{path}", json=body, timeout=timeout)
        except requests.RequestException as exc:
            raise self._fail(f"cannot reach Ollama at {self.base_url} ({type(exc).__name__})") from exc
        if resp.status_code != 200:
            detail = ""
            try:
                detail = resp.json().get("error", "")
            except ValueError:
                pass
            raise self._fail(f"Ollama {path} returned {resp.status_code}: {detail or resp.text[:120]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise self._fail(f"Ollama {path} returned invalid JSON") from exc

    # ------------------------------------------------------------------ status
    def status(self, max_age: float = 5.0) -> dict:
        """Reachability + which of our two models are pulled. Cached briefly (UI polls this)."""
        now = time.monotonic()
        if now - self._status_cache[0] < max_age and self._status_cache[1]:
            return self._status_cache[1]
        info = {"reachable": False, "embed_model_available": False, "llm_model_available": False}
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.5)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            info.update(reachable=True,
                        embed_model_available=_has_model(names, self.embed_model),
                        llm_model_available=_has_model(names, self.llm_model),
                        installed_models=names)
            if self.last_error and not self.cooling_down:
                self.last_error = None
        except (requests.RequestException, ValueError):
            info["error"] = f"cannot reach Ollama at {self.base_url}"
        self._status_cache = (now, info)
        return info

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list) -> np.ndarray:
        """Embed a list of strings -> float32 array (len(texts), dim). Not normalised."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        try:
            data = self._post("/api/embed", {"model": self.embed_model, "input": texts,
                                             "truncate": True}, self.timeout)
            vectors = data.get("embeddings")
        except OllamaError as exc:
            # Older Ollama versions only have /api/embeddings (one prompt per call).
            if "404" not in str(exc) or "not found" in str(exc).lower():
                raise
            self._down_until = 0.0
            vectors = [self._post("/api/embeddings", {"model": self.embed_model, "prompt": t},
                                  self.timeout).get("embedding") for t in texts]
        if not vectors or len(vectors) != len(texts) or any(not v for v in vectors):
            raise self._fail("Ollama returned an unexpected embeddings payload")
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------------ llm
    def chat_json(self, system: str, user: str) -> dict:
        """Ask the chat model for a JSON object; returns the parsed dict."""
        data = self._post("/api/chat", {
            "model": self.llm_model,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 220},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, self.llm_timeout)
        content = (data.get("message") or {}).get("content", "")
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise OllamaError("LLM did not return valid JSON") from exc
        if not isinstance(parsed, dict):
            raise OllamaError("LLM JSON was not an object")
        return parsed
