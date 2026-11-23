"""Offline tests: a deterministic fake Ollama stands in for the real server."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from backend.app.catalog import load_products, normalize_payload
from backend.app.config import settings as base_settings
from backend.app.lexicon import events_conflict, events_in_text, match_occasions
from backend.app.ollama_client import OllamaError
from backend.app.recommender import Recommender
from backend.app.text_utils import stem, tokenize


class FakeOllama:
    """Bag-of-words hashing 'embeddings' + canned LLM keywords. Can be switched off."""

    embed_model = "fake-embed"
    llm_model = "fake-llm"

    def __init__(self, up: bool = True):
        self.up = up
        self.last_error = None
        self.embed_calls = 0

    @property
    def cooling_down(self) -> bool:
        return not self.up

    def status(self, max_age: float = 5.0) -> dict:
        return {"reachable": self.up, "embed_model_available": self.up, "llm_model_available": self.up}

    def embed(self, texts):
        if not self.up:
            self.last_error = "down"
            raise OllamaError("down")
        self.embed_calls += 1
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1.0
        return out

    def chat_json(self, system, user):
        if not self.up:
            raise OllamaError("down")
        if "asdfghjkl" in user or "qzxv" in user:
            return {"is_occasion": False, "keywords": ["flowers", "chocolate"]}  # LLM must be ignored
        return {"is_occasion": True, "keywords": ["Sweets!", "dry fruits", "diya", "x" * 80, 42]}


@pytest.fixture(scope="module")
def products():
    return load_products(base_settings.products_path, only_active=True)


def make(products, tmp_path, up=True, **overrides):
    cfg = dict(cache_dir=tmp_path, min_score=0.05, sem_floor=0.0, sem_ceil=0.6)
    cfg.update(overrides)
    s = replace(base_settings, **cfg)
    rec = Recommender(s, products=products, client=FakeOllama(up))
    rec.semantic.ensure_ready()
    return rec


# ---------------------------------------------------------------- catalog
def test_normalize_accepts_both_schemas():
    spec = {"productId": 7, "vendorId": "v1", "vendor_category": "c", "vendor_category_desc": "Toys",
            "title": "Robot", "description": "Beeps", "files": [{"url": "http://x/y.png"}]}
    p = normalize_payload(spec)
    assert (p.product_id, p.vendor_id, p.vendor_category_desc, p.files) == ("7", "v1", "Toys", ["http://x/y.png"])
    real = {"id": "a", "shopId": "s", "categoryId": "c", "name": "Rose gold watch", "description": "Nice",
            "mainImage": "http://i/1.png", "price": "2499", "discountPercentage": "10", "stock": 3}
    p = normalize_payload(real)
    assert p.vendor_category_desc == "Watches" and p.files == ["http://i/1.png"]
    assert p.price_after_discount == 2249.1 and p.to_public()["image"] == "http://i/1.png"
    assert normalize_payload({"id": "x"}) is None  # no title


def test_inactive_listings_filtered(products):
    assert len(products) == 104  # 110 raw - 4 REVIEW_REQUIRED - 2 DRAFT


# ---------------------------------------------------------------- text + lexicon
def test_stemmer_keeps_christmas():
    assert stem("christmas") == "christmas" and stem("flowers") == "flower" and stem("watches") == "watch"


def test_occasion_matching_is_typo_tolerant():
    assert match_occasions("Valentine's Day")[0][0] == "valentines day"
    assert match_occasions("diwli")[0][0] == "diwali"
    assert match_occasions("birthdays")[0][0] == "birthday"
    assert match_occasions("qwertyuiop") == []


def test_event_conflict_logic():
    assert events_in_text("Holi Luxury Gift Hampers") == {"holi"}
    assert events_conflict({"holi"}, {"diwali"})
    assert not events_conflict({"holi"}, {"holi"})
    assert not events_conflict({"engagement"}, {"wedding"})  # related events
    assert not events_conflict(set(), {"diwali"})            # generic product
    assert not events_conflict({"holi"}, set())              # no event asked for


# ---------------------------------------------------------------- ranking
def titles(resp, n=3):
    return [r["title"] for r in resp["results"][:n]]


def test_keyword_mode_diwali_prefers_diwali_items(products, tmp_path):
    rec = make(products, tmp_path)
    resp = rec.recommend("Diwali", mode="keyword")
    assert resp["mode_used"] == "keyword" and not resp["engines"]["semantic"]
    assert all("Diwali" in t for t in titles(resp, 2))
    # other-festival hampers are demoted below the Diwali ones
    ranked = [r["title"] for r in resp["results"]]
    assert ranked.index("Diwali Luxury Gift Hampers") < ranked.index("Holi Luxury Gift Hampers")


def test_keyword_mode_valentines_finds_romantic_items(products, tmp_path):
    resp = make(products, tmp_path).recommend("Valentine's Day", mode="keyword", top_k=10)
    found = " ".join(titles(resp, 10)).lower()
    assert "rose" in found and "couple" in found or "letters" in found


def test_hybrid_uses_both_engines_and_explains(products, tmp_path):
    resp = make(products, tmp_path).recommend("Christmas", mode="hybrid")
    assert resp["mode_used"] == "hybrid" and resp["engines"] == {"keyword": True, "semantic": True, "llm": True}
    top = resp["results"][0]
    assert "Christmas" in top["title"]
    assert 0 < top["score"]["final"] <= 1 and top["score"]["keyword"] is not None and top["score"]["semantic"] is not None
    assert top["matched_terms"]
    scores = [r["score"]["final"] for r in resp["results"]]
    assert scores == sorted(scores, reverse=True)


def test_llm_output_is_sanitised(products, tmp_path):
    resp = make(products, tmp_path).recommend("Diwali")
    terms = resp["expanded_terms"]
    assert "sweets" in terms and "dry fruits" in terms
    assert not any(len(t) > 30 for t in terms)


def test_graceful_fallback_when_ollama_down(products, tmp_path):
    rec = make(products, tmp_path, up=False)
    resp = rec.recommend("Birthday", mode="hybrid")
    assert resp["mode_used"] == "keyword" and resp["results"]
    assert resp["warnings"] and not resp["engines"]["semantic"]
    resp = rec.recommend("Birthday", mode="semantic")  # explicit semantic, still degrades
    assert resp["mode_used"] == "keyword" and resp["warnings"]


def test_gibberish_gives_empty_results(products, tmp_path):
    s = replace(base_settings, cache_dir=tmp_path, min_score=0.2)
    # keyword mode, no Ollama at all
    rec = Recommender(s, products=products, client=FakeOllama(up=False))
    resp = rec.recommend("asdfghjkl", mode="keyword")
    assert resp["results"] == [] and resp["total_matches"] == 0
    # hybrid with a healthy Ollama: LLM says "not an occasion" and hallucinated keywords are ignored
    rec = make(products, tmp_path, min_score=0.2)
    for mode in ("hybrid", "semantic"):
        assert rec.recommend("asdfghjkl", mode=mode)["results"] == []
    assert rec.recommend("asdfghjkl", mode="hybrid")["expanded_terms"] == []


def test_unknown_but_real_occasion_still_gets_results(products, tmp_path):
    # "Pongal" is in neither the lexicon nor the catalog; the LLM vouches for it.
    rec = make(products, tmp_path, min_score=0.05)
    resp = rec.recommend("Pongal", mode="hybrid")
    assert resp["engines"]["llm"] and resp["results"]


def test_typo_and_plural_queries(products, tmp_path):
    rec = make(products, tmp_path)
    assert "chocolate" in titles(rec.recommend("chocolat", mode="keyword"), 3)[0].lower()
    assert "watch" in titles(rec.recommend("watches", mode="keyword"), 3)[0].lower()


def test_top_k_and_blank_input(products, tmp_path):
    rec = make(products, tmp_path)
    assert len(rec.recommend("gift", top_k=5)["results"]) <= 5


def test_embedding_cache_reused(products, tmp_path):
    rec1 = make(products, tmp_path)
    calls = rec1.client.embed_calls
    assert calls > 0
    rec2 = make(products, tmp_path)  # same cache dir -> no re-embedding of the catalog
    assert rec2.client.embed_calls == 0 and rec2.semantic.ready
