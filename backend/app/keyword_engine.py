"""Keyword / rule-based relevance (BM25-style) over weighted product fields.

Index time: every product is tokenised per field (title, category, short description,
description, brand) and folded into an inverted index of field-weighted term frequencies.

Query time: the caller supplies *weighted* query terms (the user's own words at weight 1.0,
lexicon themes and LLM keywords at lower weights).  Each term contributes
``weight * idf * saturating_tf`` to every product that contains it.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import get_close_matches

import numpy as np

from .catalog import Product
from .lexicon import OCCASION_LEXICON, concept_terms, match_occasions
from .text_utils import QUERY_GENERIC, STOPWORDS, stem, tokenize

# A hit in the title is worth 3x a hit in the long description.
FIELD_WEIGHTS = {"title": 3.0, "category": 2.0, "short": 1.5, "description": 1.0, "brand": 0.8}
K1 = 1.2  # term-frequency saturation
# Raw scores are divided by max(best raw score, KW_REF) so weak matches are not inflated to 1.0.
KW_REF = 4.0


@dataclass
class QueryTerm:
    weight: float
    display: str  # the human word to show as a "matched" chip


@dataclass
class KeywordResult:
    scores: np.ndarray  # 0-1, one per product
    matched: list  # per product: [display terms, strongest first]


class KeywordEngine:
    def __init__(self, products: list):
        self.n = len(products)
        self.postings: dict = defaultdict(dict)  # stem -> {product_idx: weighted_tf}
        for idx, p in enumerate(products):
            fields = {
                "title": p.title,
                "category": p.vendor_category_desc,
                "short": p.short_description,
                "description": p.description,
                "brand": p.brand,
            }
            for fname, text in fields.items():
                for tok, count in Counter(tokenize(text)).items():
                    self.postings[tok][idx] = self.postings[tok].get(idx, 0.0) + FIELD_WEIGHTS[fname] * count
        self.idf = {
            tok: math.log(1 + (self.n - len(docs) + 0.5) / (len(docs) + 0.5))
            for tok, docs in self.postings.items()
        }
        self.vocab = list(self.postings)

    # ------------------------------------------------------------------ query building
    def build_terms(self, occasion: str, extra_keywords: list = (), use_lexicon: bool = True) -> dict:
        """Combine the user's words, lexicon themes and (optional) LLM keywords into
        ``{stem: QueryTerm}``.  The highest weight wins when a stem appears several times."""
        terms: dict = {}

        def add(word: str, weight: float, display: str | None = None) -> None:
            for raw in tokenize(word, do_stem=False):
                if raw in STOPWORDS:
                    continue
                s = stem(raw)
                if s not in self.idf:  # typo tolerance: "chocolat" -> "chocolate"
                    close = get_close_matches(s, self.vocab, n=1, cutoff=0.86)
                    if not close:
                        continue
                    s, weight = close[0], weight * 0.8
                if s not in terms or terms[s].weight < weight:
                    terms[s] = QueryTerm(weight, display or raw)

        # 1) the user's own words (drop filler like "gift"/"day" unless nothing else remains)
        words = tokenize(occasion, do_stem=False)
        meaningful = [w for w in words if w not in QUERY_GENERIC]
        for w in meaningful or words:
            add(w, 1.0)

        # 2) curated occasion themes
        if use_lexicon:
            for key, strength in match_occasions(occasion):
                for term, w in OCCASION_LEXICON[key]["terms"].items():
                    add(term, 0.8 * w * strength)

        # 3) LLM-suggested keywords (multi-word phrases are split; weight shared across words)
        for phrase in extra_keywords:
            parts = [t for t in tokenize(phrase, do_stem=False) if t not in STOPWORDS]
            for part in parts:
                add(part, 0.6 / math.sqrt(max(len(parts), 1)))
        return terms

    def has_direct_hit(self, occasion: str) -> bool:
        """True if at least one of the user's own words exists in the catalog vocabulary
        (exactly, or as a near-typo).  Used to tell real queries from gibberish."""
        for raw in tokenize(occasion, do_stem=False):
            if raw in STOPWORDS or raw in QUERY_GENERIC:
                continue
            s = stem(raw)
            if s in self.idf or get_close_matches(s, self.vocab, n=1, cutoff=0.86):
                return True
        return False

    # ------------------------------------------------------------------ scoring
    def score(self, terms: dict) -> KeywordResult:
        raw = np.zeros(self.n)
        contrib: list = [dict() for _ in range(self.n)]
        for stem_, qt in terms.items():
            idf = self.idf.get(stem_)
            if idf is None:
                continue
            for idx, wtf in self.postings[stem_].items():
                s = qt.weight * idf * wtf * (K1 + 1) / (wtf + K1)
                raw[idx] += s
                contrib[idx][qt.display] = contrib[idx].get(qt.display, 0.0) + s
        peak = max(float(raw.max()) if self.n else 0.0, KW_REF)
        matched = [
            [d for d, _ in sorted(c.items(), key=lambda kv: -kv[1])] for c in contrib
        ]
        return KeywordResult(scores=raw / peak, matched=matched)
