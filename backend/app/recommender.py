"""Hybrid occasion recommender: keyword rules + Ollama embeddings (+ optional LLM expansion).

Pipeline for one query
  1. Match the text against the curated occasion lexicon (typo tolerant).
  2. (hybrid/semantic) Ask the Ollama chat model for extra product keywords - cached.
  3. Keyword score: BM25-style over weighted fields using the user's words, lexicon themes
     and LLM keywords.
  4. Semantic score: cosine(query embedding, product embedding) rescaled to 0-1.
  5. Blend:  final = w_sem * semantic + w_kw * keyword   (weights renormalised)
  6. Demote products made for a *different* specific event (Holi hamper for "Diwali").
  7. Drop anything under MIN_SCORE (-> empty state), sort, return top-k with explanations.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from .catalog import Product, load_products
from .config import Settings
from .keyword_engine import KeywordEngine
from .lexicon import concept_terms, events_conflict, events_in_text, match_occasions
from .ollama_client import OllamaClient, OllamaError
from .semantic_engine import SemanticEngine
from .text_utils import normalize

log = logging.getLogger("recommender")

MODES = ("hybrid", "keyword", "semantic")
EVENT_MISMATCH_FACTOR = 0.55

_LLM_SYSTEM = (
    "You are a gift-recommendation assistant for an Indian online gift marketplace that sells "
    "chocolates, flowers, hampers, jewellery, perfumes, candles, stationery, apparel, watches, "
    "toys, wellness and eco-friendly products. Reply with a JSON object only."
)


class Recommender:
    def __init__(self, settings: Settings, products: Optional[list] = None,
                 client: Optional[OllamaClient] = None):
        self.settings = settings
        self.products: list = products if products is not None else load_products(
            settings.products_path, settings.only_active)
        self.client = client or OllamaClient(settings.ollama_url, settings.embed_model,
                                             settings.llm_model, settings.ollama_timeout,
                                             settings.llm_timeout)
        self.keyword = KeywordEngine(self.products)
        self.semantic = SemanticEngine(self.products, self.client, settings.cache_dir)
        self._product_events = [events_in_text(p.title) for p in self.products]
        self._llm_cache: dict = {}
        self._llm_lock = threading.Lock()
        log.info("Catalog ready: %d products", len(self.products))

    # ------------------------------------------------------------------ lifecycle
    def warm_up(self) -> None:
        """Build/load embeddings in the background so the first search is fast."""
        if not self.semantic.ensure_ready(block=True):
            log.warning("Semantic index not built yet (%s). Keyword mode still works.",
                        self.client.last_error or "Ollama unavailable")

    def status(self) -> dict:
        ollama = self.client.status()
        return {
            "catalog_size": len(self.products),
            "semantic_ready": self.semantic.ready,
            "ollama": {
                "url": self.settings.ollama_url,
                "embed_model": self.settings.embed_model,
                "llm_model": self.settings.llm_model,
                "last_error": self.client.last_error,
                **ollama,
            },
        }

    # ------------------------------------------------------------------ LLM expansion
    def _llm_keywords(self, occasion: str) -> tuple:
        """Ask the LLM (a) whether this is a real occasion/recipient/gifting theme and (b) for
        product themes that suit it.  Returns ``(is_occasion | None, keywords)``;
        ``(None, [])`` when the LLM is unavailable."""
        key = normalize(occasion).strip()
        with self._llm_lock:
            if key in self._llm_cache:
                return self._llm_cache[key]
        if self.client.cooling_down:
            return None, []
        prompt = (
            f'Input: "{occasion[:80]}"\n'
            'Return JSON {"is_occasion": true|false, "keywords": [...]}. '
            '"is_occasion" is false if the input is gibberish or is not an occasion, event, '
            "festival, recipient or gifting theme. When true, give 8 to 12 short lowercase "
            "words or two-word phrases naming product types, themes or moods that suit it; "
            "when false, give an empty list. No explanations."
        )
        try:
            parsed = self.client.chat_json(_LLM_SYSTEM, prompt)
        except OllamaError:
            return None, []
        flag = parsed.get("is_occasion")
        if isinstance(flag, str):
            flag = {"true": True, "false": False}.get(flag.strip().lower())
        valid = flag if isinstance(flag, bool) else None
        raw = parsed.get("keywords", [])
        keywords: list = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str):
                clean = re.sub(r"[^a-z0-9 ]+", "", normalize(item)).strip()
                if 2 <= len(clean) <= 30 and len(clean.split()) <= 3 and clean not in keywords:
                    keywords.append(clean)
        result = (valid, keywords[:12] if valid is not False else [])
        with self._llm_lock:
            if len(self._llm_cache) > 500:
                self._llm_cache.clear()
            self._llm_cache[key] = result
        return result

    # ------------------------------------------------------------------ main entry point
    def recommend(self, occasion: str, top_k: int = 12, mode: str = "hybrid",
                  use_llm: bool = True) -> dict:
        started = time.perf_counter()
        occasion = occasion.strip()
        mode = mode if mode in MODES else "hybrid"
        warnings: list = []
        s = self.settings

        lexicon_hits = match_occasions(occasion)
        matched_keys = {k for k, strength in lexicon_hits if strength >= 0.75}

        # -- optional LLM query expansion (never in pure keyword mode)
        llm_keywords: list = []
        llm_valid: Optional[bool] = None
        if use_llm and mode != "keyword":
            llm_valid, llm_keywords = self._llm_keywords(occasion)
            if llm_valid is None and self.client.cooling_down:
                warnings.append("The Ollama language model is unavailable, so query expansion was skipped.")

        # -- keyword engine
        terms = self.keyword.build_terms(occasion, llm_keywords, use_lexicon=True)
        kw = self.keyword.score(terms) if mode in ("hybrid", "keyword") else None

        # -- semantic engine
        cosine: Optional[np.ndarray] = None
        if mode in ("hybrid", "semantic"):
            themes = concept_terms(lexicon_hits, limit=8)
            query_text = f"Gift ideas for {occasion}"
            extras = list(dict.fromkeys(themes + llm_keywords[:8]))
            if extras:
                query_text += ". Related: " + ", ".join(extras)
            cosine = self.semantic.similarities(query_text)
            if cosine is None:
                warnings.append(
                    "Semantic search is unavailable (is Ollama running with the "
                    f"'{s.embed_model}' model?). Showing keyword matches instead.")

        sem = None
        if cosine is not None:
            sem = np.clip((cosine - s.sem_floor) / max(s.sem_ceil - s.sem_floor, 1e-6), 0.0, 1.0)

        # -- blend
        if sem is not None and kw is not None:
            if kw.scores.max() > 0:
                total_w = s.w_semantic + s.w_keyword
                final = (s.w_semantic * sem + s.w_keyword * kw.scores) / total_w
            else:  # no keyword evidence at all -> trust the embeddings alone
                final = sem.copy()
            mode_used = "hybrid"
        elif sem is not None:
            final, mode_used = sem.copy(), "semantic"
        else:
            if kw is None:  # semantic requested but unavailable -> fall back to keywords
                kw = self.keyword.score(terms)
            final, mode_used = kw.scores.copy(), "keyword"

        # -- evidence gate: refuse to "recommend" for gibberish.  A query needs *some* sign it
        #    is real: a recognised occasion, a word that exists in the catalog, an LLM
        #    confirmation, or (last resort) a plain-text embedding that genuinely resembles
        #    some product.  Without that, the blended score is just template/LLM noise.
        has_evidence = (
            bool(matched_keys)
            or self.keyword.has_direct_hit(occasion)
            or llm_valid is True
        )
        if not has_evidence:
            plain = None if llm_valid is False else self.semantic.similarities(occasion)
            if plain is None or float(plain.max()) < s.sem_gate:
                final[:] = 0.0

        # -- demote products built for a different specific event
        for i, ev in enumerate(self._product_events):
            if events_conflict(ev, matched_keys):
                final[i] *= EVENT_MISMATCH_FACTOR

        # -- rank
        keep = np.where(final >= s.min_score)[0]
        order = keep[np.argsort(-final[keep], kind="stable")]
        matched_terms = kw.matched if kw is not None else [[] for _ in self.products]

        results = []
        for idx in order[:top_k]:
            product: Product = self.products[int(idx)]
            results.append({
                **product.to_public(),
                "score": {
                    "final": round(float(final[idx]), 4),
                    "keyword": round(float(kw.scores[idx]), 4) if kw is not None else None,
                    "semantic": round(float(sem[idx]), 4) if sem is not None else None,
                },
                "matched_terms": matched_terms[int(idx)][:5],
            })

        shown_terms = [t.display for t in terms.values()]
        return {
            "occasion": occasion,
            "mode_requested": mode,
            "mode_used": mode_used,
            "engines": {"keyword": kw is not None, "semantic": sem is not None,
                        "llm": bool(llm_keywords)},
            "recognised_occasions": [k for k, _ in lexicon_hits],
            "expanded_terms": list(dict.fromkeys(llm_keywords + shown_terms))[:14],
            "warnings": warnings,
            "total_matches": int(len(keep)),
            "returned": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": results,
        }
