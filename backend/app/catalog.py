"""Catalog loading + normalisation.

The assignment describes a payload with ``product_id / vendor_id / vendor_category /
vendor_category_desc / title / description / files``.  The supplied ``products.json`` uses a
slightly different shape (``id / shopId / categoryId / name / mainImage`` ...), so
``normalize_payload`` accepts **both** and maps them onto one internal ``Product``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- category labels
# The supplied catalog only has opaque ``categoryId`` values and no human-readable category
# description, so we infer one from the product text.  First matching rule wins, so order
# matters (e.g. "Rose gold watch" must hit *Watches* before *Flowers*).
# If a payload already carries ``vendor_category_desc`` that value is always used instead.
CATEGORY_RULES: list = [
    ("Watches", r"\bwatch(es)?\b"),
    ("Jewellery", r"necklace|nacklace|bracelet|earring|earing|choker|jewel"),
    ("Candles", r"\bcandles?\b"),
    ("Fragrance & Bath", r"perfume|parfum|fragrance|\bsoaps?\b|body oil|bath salt"),
    ("Chocolates", r"chocolate"),
    ("Hampers", r"hampers?"),
    ("Gift Cards", r"gift cards?|greeting|gretting"),
    ("Gift Packaging", r"wrapping|gift box|wax seal|gift sets?|bundles?"),
    ("Flowers", r"flower|bouquet|\brose\b|lilies|tulip|marigold|blossom"),
    ("Pens & Stationery", r"\bpens?\b|notebook|journal|stationery"),
    ("Wellness & Self-care", r"self-?care|wellness|fitness|skincare|herbal|\btea"),
    ("Eco-friendly", r"bamboo|recycled|upcycled|natural fib|planter|indoor plant"),
    ("Toys & Kids", r"\btoys?\b|plush|teddy|xylophone|drum|guitar|slime|craft|painting|\bdiy\b"),
    ("Personalised & Couple", r"couple|personali[sz]ed|custom|open when|\bmugs?\b|story ?book|star map"),
    ("Celebration Gifts", r"engagement|housewarming|baby shower|newborn"),
    ("Food & Snacks", r"snack|dry fruit"),
    (
        "Apparel",
        r"shirt|t-shirt|tshirt|kurti|saree|suit|cargo|jeans|pants|wears?\b|\bprint",
    ),
]
_COMPILED_RULES = [(label, re.compile(rx, re.I)) for label, rx in CATEGORY_RULES]


def infer_category_label(title: str, short: str = "", brand: str = "") -> str:
    """Best-effort human category: match the title first, then the short description."""
    for text in (title, short):
        for label, rx in _COMPILED_RULES:
            if text and rx.search(text):
                return label
    return brand or "General"


# --------------------------------------------------------------------------- model
@dataclass
class Product:
    product_id: str
    vendor_id: str
    vendor_category: str
    vendor_category_desc: str
    title: str
    description: str
    short_description: str = ""
    files: list = field(default_factory=list)
    brand: str = ""
    price: Optional[float] = None
    discount_percentage: float = 0.0
    stock: Optional[int] = None

    # -- text used by the engines -----------------------------------------------------
    def search_text(self) -> str:
        """One blob describing the product, used for embeddings (capped to keep requests small)."""
        parts = [
            self.title,
            f"Category: {self.vendor_category_desc}" if self.vendor_category_desc else "",
            self.short_description,
            self.description,
            f"Brand: {self.brand}" if self.brand else "",
        ]
        return ". ".join(p for p in parts if p)[:1200]

    def snippet(self, limit: int = 170) -> str:
        text = self.description if len(self.description) >= 40 else (
            self.short_description or self.description
        )
        if len(text) <= limit:
            return text
        cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:- ")
        return cut + "…"

    @property
    def price_after_discount(self) -> Optional[float]:
        if self.price is None:
            return None
        return round(self.price * (1 - self.discount_percentage / 100), 2)

    def to_public(self) -> dict:
        return {
            "product_id": self.product_id,
            "vendor_id": self.vendor_id,
            "vendor_category": self.vendor_category,
            "vendor_category_desc": self.vendor_category_desc,
            "title": self.title,
            "description": self.description,
            "snippet": self.snippet(),
            "files": self.files,
            "image": self.files[0] if self.files else None,
            "brand": self.brand,
            "price": self.price,
            "discount_percentage": self.discount_percentage,
            "price_after_discount": self.price_after_discount,
            "in_stock": None if self.stock is None else self.stock > 0,
        }


# --------------------------------------------------------------------------- parsing
def _clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\r", " ")).strip()


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _file_urls(payload: dict) -> list:
    """``files`` may hold plain URLs or objects with a url key; fall back to ``mainImage``."""
    urls: list = []
    for item in payload.get("files") or []:
        if isinstance(item, str):
            urls.append(item)
        elif isinstance(item, dict):
            url = item.get("url") or item.get("src") or item.get("path")
            if url:
                urls.append(str(url))
    for key in ("mainImage", "main_image", "image"):
        if payload.get(key):
            urls.append(str(payload[key]))
    for extra in payload.get("images") or []:
        if isinstance(extra, str):
            urls.append(extra)
    seen: set = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def normalize_payload(payload: dict) -> Optional[Product]:
    """Map one raw payload (either schema) onto ``Product``. Returns None if unusable."""
    product_id = payload.get("product_id") or payload.get("productId") or payload.get("id")
    vendor_id = (
        payload.get("vendor_id") or payload.get("vendorId") or payload.get("shopId") or ""
    )
    vendor_category = payload.get("vendor_category") or payload.get("categoryId") or ""
    title = _clean(payload.get("title") or payload.get("name"))
    if not product_id or not title:
        return None

    description = _clean(payload.get("description", ""))
    short = _clean(payload.get("shortDescription") or payload.get("short_description"))
    brand = _clean(payload.get("brand"))
    category_desc = _clean(payload.get("vendor_category_desc")) or infer_category_label(
        title, short, brand
    )
    stock = payload.get("stock")
    return Product(
        product_id=str(product_id),
        vendor_id=str(vendor_id),
        vendor_category=str(vendor_category),
        vendor_category_desc=category_desc,
        title=title,
        description=description or short,
        short_description=short,
        files=_file_urls(payload),
        brand=brand,
        price=_to_float(payload.get("price")),
        discount_percentage=_to_float(payload.get("discountPercentage")) or 0.0,
        stock=int(stock) if isinstance(stock, (int, float)) else None,
    )


def load_products(path: Path, only_active: bool = True) -> list:
    """Read the JSON catalog (a list, or an object wrapping a list) into ``Product``s."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), [])

    products: list = []
    seen: set = set()
    for payload in raw:
        if not isinstance(payload, dict) or payload.get("deletedAt"):
            continue
        if only_active and payload.get("status") not in (None, "ACTIVE"):
            continue  # skip DRAFT / REVIEW_REQUIRED listings
        product = normalize_payload(payload)
        if product and product.product_id not in seen:
            seen.add(product.product_id)
            products.append(product)
    return products
