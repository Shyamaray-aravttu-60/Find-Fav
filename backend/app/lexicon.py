"""Curated occasion knowledge: which product themes suit which occasion.

This is the *rule-based* half of the keyword engine.  Each occasion has
  * ``aliases`` - phrases a user might type (matched whole-word, with typo tolerance)
  * ``terms``   - product words/themes with a 0-1 weight describing how strongly they fit

It lets "Valentine's Day" find roses and chocolates even though neither product mentions
Valentine's, and gives the keyword mode something useful without any LLM.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .text_utils import normalize


def _e(aliases: list, terms: dict) -> dict:
    return {"aliases": aliases, "terms": terms}


OCCASION_LEXICON: dict = {
    "birthday": _e(
        ["birthday", "bday", "birth day", "happy birthday", "b day"],
        {"birthday": 1, "party": .6, "celebration": .6, "chocolate": .7, "flower": .6, "bouquet": .6,
         "teddy": .5, "plush": .4, "toy": .4, "personalised": .5, "perfume": .4, "watch": .4,
         "card": .5, "surprise": .4, "hamper": .4},
    ),
    "wedding": _e(
        ["wedding", "marriage", "shaadi", "bride", "groom", "bridal", "newlywed", "newly wed"],
        {"wedding": 1, "bridal": .7, "jewellery": .7, "necklace": .6, "earring": .5, "bracelet": .5,
         "hamper": .7, "luxury": .6, "gold": .5, "couple": .6, "saree": .5, "elegant": .4,
         "box": .4, "card": .4, "celebration": .5},
    ),
    "anniversary": _e(
        ["anniversary", "anniv"],
        {"anniversary": 1, "couple": .8, "romantic": .8, "love": .6, "rose": .6, "flower": .5,
         "perfume": .5, "jewellery": .5, "bracelet": .5, "chocolate": .5, "personalised": .5,
         "watch": .4, "candle": .4},
    ),
    "valentines day": _e(
        ["valentines day", "valentine", "valentines", "vday", "v day", "date night", "love"],
        {"valentine": 1, "romantic": 1, "love": .9, "couple": .9, "rose": .8, "flower": .6,
         "chocolate": .8, "perfume": .6, "heart": .5, "bracelet": .4, "candle": .5,
         "personalised": .5, "letter": .4, "teddy": .5, "forever": .4},
    ),
    "diwali": _e(
        ["diwali", "deepavali", "deepawali", "dipawali", "festival of lights", "dhanteras",
         "laxmi puja"],
        {"diwali": 1, "festival": .7, "festive": .7, "hamper": .9, "dry": .6, "fruit": .6,
         "luxury": .5, "sweet": .7, "diya": .7, "candle": .6, "decor": .5, "chocolate": .5},
    ),
    "holi": _e(
        ["holi", "festival of colours", "festival of colors"],
        {"holi": 1, "festival": .6, "festive": .6, "hamper": .8, "dry": .5, "fruit": .5,
         "colour": .4, "sweet": .5, "luxury": .4},
    ),
    "christmas": _e(
        ["christmas", "xmas", "x mas", "santa", "noel"],
        {"christmas": 1, "festive": .7, "hamper": .8, "dry": .4, "fruit": .4, "chocolate": .6,
         "candle": .6, "card": .5, "plush": .4, "teddy": .4, "wrapping": .5, "box": .4,
         "luxury": .4},
    ),
    "new year": _e(
        ["new year", "newyear", "new years", "nye"],
        {"celebration": .6, "party": .6, "hamper": .5, "chocolate": .5, "journal": .6,
         "notebook": .6, "planner": .6, "card": .4, "luxury": .4, "candle": .4},
    ),
    "housewarming": _e(
        ["housewarming", "house warming", "new home", "griha pravesh", "new house", "new flat"],
        {"housewarming": 1, "home": .8, "decor": .7, "plant": .7, "planter": .7, "candle": .7,
         "bamboo": .5, "natural": .4, "fiber": .4, "lifestyle": .4, "hamper": .4, "indoor": .5,
         "aromatic": .4},
    ),
    "baby shower": _e(
        ["baby shower", "newborn", "new born", "new baby", "baby", "godh bharai",
         "naming ceremony"],
        {"baby": 1, "newborn": 1, "shower": .9, "celebration": .5, "plush": .7, "teddy": .7,
         "soft": .6, "toy": .6, "cuddly": .5, "personalised": .4, "story": .4, "kid": .4,
         "organic": .4},
    ),
    "engagement": _e(
        ["engagement", "engaged", "roka", "sagai", "ring ceremony", "proposal"],
        {"engagement": 1, "couple": .8, "romantic": .6, "jewellery": .7, "bracelet": .6,
         "necklace": .5, "perfume": .5, "luxury": .5, "chocolate": .5, "flower": .5,
         "rose": .5, "celebration": .6, "personalised": .4, "hamper": .4},
    ),
    "mothers day": _e(
        ["mothers day", "mother", "mom", "mum", "mommy", "mummy", "maa"],
        {"flower": .8, "bouquet": .7, "perfume": .7, "jewellery": .7, "necklace": .6,
         "earring": .5, "bracelet": .5, "skincare": .7, "wellness": .6, "care": .5, "bath": .5,
         "tea": .5, "candle": .5, "saree": .5, "kurti": .5, "chocolate": .4, "hamper": .4,
         "personalised": .4, "love": .4},
    ),
    "fathers day": _e(
        ["fathers day", "father", "dad", "daddy", "papa", "baba"],
        {"watch": .8, "pen": .8, "shirt": .6, "polo": .5, "executive": .6, "perfume": .5,
         "fitness": .5, "health": .4, "wellness": .4, "tshirt": .4, "personalised": .4,
         "mug": .4, "notebook": .4},
    ),
    "raksha bandhan": _e(
        ["raksha bandhan", "rakshabandhan", "rakhi", "raksha", "bhai dooj", "bhaidooj",
         "brother", "sister", "sibling"],
        {"rakhi": 1, "raksha": .8, "brother": .7, "sister": .7, "sibling": .6, "chocolate": .8,
         "hamper": .7, "dry": .5, "fruit": .5, "watch": .6, "pen": .4, "card": .5,
         "personalised": .4, "festive": .5, "sweet": .6, "jewellery": .5, "bracelet": .5},
    ),
    "friendship day": _e(
        ["friendship day", "friendship", "friend", "bestie", "bff"],
        {"friend": 1, "friendship": 1, "bracelet": .7, "chocolate": .6, "keychain": .7,
         "plush": .5, "mug": .5, "card": .5, "journal": .5, "notebook": .5, "personalised": .5,
         "snack": .5, "candle": .4},
    ),
    "office party": _e(
        ["office party", "corporate", "corporate gift", "office", "colleague", "coworker",
         "co worker", "workplace", "client", "business", "boss", "staff", "team"],
        {"office": 1, "corporate": .9, "professional": .8, "executive": .8, "pen": .8,
         "notebook": .7, "journal": .7, "watch": .6, "hamper": .7, "dry": .4, "fruit": .4,
         "chocolate": .5, "wellness": .5, "desk": .5, "shirt": .5, "formal": .4, "planter": .4,
         "tea": .4},
    ),
    "party": _e(
        ["party", "get together", "celebration", "celebrate", "bash"],
        {"party": 1, "celebration": 1, "snack": .6, "chocolate": .6, "hamper": .5, "candle": .4,
         "dj": .4, "music": .4},
    ),
    "farewell": _e(
        ["farewell", "retirement", "retire", "goodbye", "send off", "leaving", "resignation"],
        {"farewell": 1, "pen": .7, "executive": .6, "watch": .7, "journal": .6, "notebook": .6,
         "card": .6, "personalised": .6, "hamper": .5, "plant": .4, "mug": .4},
    ),
    "congratulations": _e(
        ["graduation", "convocation", "graduate", "degree", "promotion", "new job",
         "congratulations", "congrats", "achievement", "success"],
        {"pen": .8, "executive": .6, "watch": .7, "notebook": .6, "journal": .6, "card": .6,
         "hamper": .6, "chocolate": .6, "flower": .5, "bouquet": .5, "luxury": .4,
         "personalised": .4, "celebration": .6},
    ),
    "get well soon": _e(
        ["get well", "get well soon", "recovery", "sick", "feel better", "wellness", "health",
         "self care", "selfcare", "relax", "stress", "mental health", "spa"],
        {"wellness": 1, "health": .8, "care": .8, "self": .6, "relax": .7, "calm": .6, "tea": .7,
         "herbal": .7, "organic": .6, "bath": .6, "skincare": .6, "candle": .6, "lavender": .6,
         "flower": .5, "fitness": .5, "mental": .6, "comfort": .6},
    ),
    "thank you": _e(
        ["thank you", "thanks", "thankyou", "gratitude", "appreciation", "thank"],
        {"thank": 1, "appreciation": .8, "card": .7, "hamper": .7, "chocolate": .7, "flower": .6,
         "bouquet": .5, "candle": .5, "tea": .5, "personalised": .4, "luxury": .4},
    ),
    "sorry": _e(
        ["sorry", "apology", "apologise", "apologize", "forgive", "make up"],
        {"sorry": 1, "apology": 1, "chocolate": .8, "flower": .8, "bouquet": .7, "rose": .6,
         "card": .7, "letter": .6, "teddy": .6, "plush": .5, "personalised": .4},
    ),
    "long distance": _e(
        ["long distance", "ldr", "missing you", "miss you", "far away"],
        {"distance": 1, "letter": .8, "open": .5, "couple": .7, "touch": .7, "bracelet": .6,
         "mug": .6, "personalised": .5, "star": .4, "map": .4, "love": .6, "romantic": .5},
    ),
    "eco friendly": _e(
        ["eco friendly", "eco", "sustainable", "green", "zero waste", "recycled", "environment",
         "plastic free", "organic", "vegan"],
        {"sustainable": 1, "eco": 1, "recycled": .9, "upcycled": .9, "organic": .8, "natural": .8,
         "plant": .7, "bamboo": .8, "herbal": .5, "fiber": .5, "planter": .5, "environment": .6},
    ),
    "kids": _e(
        ["kids", "kid", "children", "child", "toddler", "toy", "toys", "boy", "girl",
         "kids party", "kids birthday"],
        {"kid": 1, "child": .9, "toy": 1, "play": .6, "plush": .6, "teddy": .6, "craft": .8,
         "diy": .6, "slime": .6, "painting": .6, "coloring": .6, "musical": .7, "xylophone": .6,
         "drum": .5, "guitar": .5, "learning": .4, "creative": .6, "soft": .4},
    ),
    "eid": _e(
        ["eid", "eid mubarak", "ramadan", "ramzan", "iftar"],
        {"eid": 1, "festive": .7, "festival": .6, "hamper": .8, "dry": .8, "fruit": .8,
         "perfume": .7, "attar": .6, "luxury": .6, "sweet": .6, "chocolate": .5, "kurti": .4,
         "suit": .4},
    ),
    "karwa chauth": _e(
        ["karwa chauth", "karva chauth", "karwachauth"],
        {"couple": .7, "jewellery": .8, "bracelet": .6, "necklace": .6, "earring": .5,
         "saree": .6, "flower": .5, "rose": .4, "chocolate": .5, "perfume": .5, "hamper": .5,
         "romantic": .5, "festive": .5},
    ),
    "teachers day": _e(
        ["teachers day", "teacher", "teachers", "guru", "professor", "mentor", "tutor"],
        {"pen": .9, "notebook": .8, "journal": .8, "card": .6, "plant": .5, "planter": .4,
         "flower": .5, "bouquet": .4, "tea": .4, "mug": .4, "chocolate": .5, "personalised": .4,
         "executive": .5},
    ),
}

# Quick-pick chips shown in the UI (display text, in order).
SUGGESTED_OCCASIONS = [
    "Birthday", "Wedding", "Diwali", "Valentine's Day", "Office Party", "Anniversary",
    "Baby Shower", "Housewarming", "Holi", "Christmas", "Raksha Bandhan", "Get Well Soon",
]

_FUZZY_CUTOFF = 0.84


def match_occasions(query: str) -> list:
    """Return ``[(occasion_key, strength 0-1)]`` for every known occasion the query mentions.

    Exact whole-word alias hit -> 1.0.  Near-miss (typos, plurals: "diwli", "birthdays")
    -> similarity * 0.9.  Several occasions can match ("office party" hits two entries).
    """
    norm = normalize(query)
    norm = re.sub(r"[^a-z0-9 ]+", " ", norm)
    q_tokens = norm.split()
    hits: dict = {}
    for key, entry in OCCASION_LEXICON.items():
        best = 0.0
        for alias in entry["aliases"]:
            a = normalize(alias)
            if re.search(rf"\b{re.escape(a)}\b", norm):
                best = 1.0
                break
            if len(a) < 4:
                continue  # too short to fuzzy-match safely
            n = len(a.split())
            for i in range(len(q_tokens) - n + 1):
                ratio = SequenceMatcher(None, a, " ".join(q_tokens[i:i + n])).ratio()
                if ratio >= _FUZZY_CUTOFF:
                    best = max(best, ratio * 0.9)
        if best:
            hits[key] = best
    return sorted(hits.items(), key=lambda kv: -kv[1])


def concept_terms(matches: list, limit: int = 8) -> list:
    """Top product themes (highest weight first) across the matched occasions."""
    weights: dict = {}
    for key, strength in matches:
        for term, w in OCCASION_LEXICON[key]["terms"].items():
            weights[term] = max(weights.get(term, 0.0), w * strength)
    return [t for t, _ in sorted(weights.items(), key=lambda kv: -kv[1])[:limit]]


# --------------------------------------------------------------------------- event specificity
# Some catalog items are built for one specific event ("Holi Luxury Gift Hampers").  When the
# user asks for a *different* event they should be demoted, even though their generic words
# ("hamper", "festive") overlap.  Only discrete events count; broad themes like "kids" don't.
EVENT_OCCASIONS = frozenset({
    "diwali", "holi", "christmas", "eid", "raksha bandhan", "karwa chauth", "new year",
    "baby shower", "engagement", "housewarming", "wedding", "valentines day", "anniversary",
    "mothers day", "fathers day", "teachers day",
})

# Event pairs that are close enough not to penalise each other.
_RELATED = {
    frozenset(p) for p in [
        ("wedding", "engagement"), ("wedding", "anniversary"), ("engagement", "anniversary"),
        ("valentines day", "anniversary"), ("valentines day", "engagement"),
        ("karwa chauth", "anniversary"), ("diwali", "raksha bandhan"),
    ]
}


def events_in_text(text: str) -> set:
    """Discrete events a piece of text (a product title) explicitly names - exact aliases only."""
    norm = re.sub(r"[^a-z0-9 ]+", " ", normalize(text))
    found = set()
    for key in EVENT_OCCASIONS:
        for alias in OCCASION_LEXICON[key]["aliases"]:
            if re.search(rf"\b{re.escape(normalize(alias))}\b", norm):
                found.add(key)
                break
    return found


def events_conflict(product_events: set, query_occasions: set) -> bool:
    """True when the product is event-specific, the user asked for an event, and they don't fit."""
    wanted = query_occasions & EVENT_OCCASIONS
    if not product_events or not wanted:
        return False
    if product_events & wanted:
        return False
    return not any(frozenset((a, b)) in _RELATED for a in product_events for b in wanted)
