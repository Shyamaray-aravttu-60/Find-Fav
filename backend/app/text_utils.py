"""Small text-normalisation helpers shared by the keyword and lexicon code."""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words ignored everywhere (index and queries).
STOPWORDS = frozenset(
    """a an the and or of for to in on at with from by is are be as it its this that these those
    your you my our we i me us can will into up out so than then very more most""".split()
)

# Words that carry no signal in a *query* ("gifts for mum's day") but are fine in documents.
# Dropped from queries only if something meaningful remains.
QUERY_GENERIC = frozenset(
    "gift gifts present presents idea ideas day days best good perfect something".split()
)


def normalize(text: str) -> str:
    """Lowercase, strip accents (DÉCOR -> decor), drop apostrophes, '&' -> 'and'."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ")
    return text.replace("'", "").replace("\u2019", "")


def stem(tok: str) -> str:
    """A deliberately tiny plural-stripper (flowers->flower, watches->watch, christmas->christmas)."""
    if len(tok) <= 3 or tok.endswith(("ss", "us", "is", "as")):
        return tok
    if tok.endswith(("ches", "shes", "xes", "sses")):
        return tok[:-2]
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def tokenize(text: str, *, do_stem: bool = True, drop_stop: bool = True) -> list:
    toks = _TOKEN_RE.findall(normalize(text))
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks] if do_stem else toks
