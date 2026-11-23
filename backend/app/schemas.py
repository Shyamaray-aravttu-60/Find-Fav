"""Pydantic request/response models (these also drive the auto-generated /docs page)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    occasion: str = Field(..., min_length=1, max_length=120, examples=["Diwali"])
    top_k: int = Field(12, ge=1, le=50)
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid"
    use_llm: bool = Field(True, description="Let the Ollama chat model expand the occasion into keywords.")

    @field_validator("occasion")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("occasion must not be blank")
        return v


class Scores(BaseModel):
    final: float
    keyword: Optional[float] = None
    semantic: Optional[float] = None


class ProductOut(BaseModel):
    product_id: str
    vendor_id: str
    vendor_category: str
    vendor_category_desc: str
    title: str
    description: str
    snippet: str
    files: list
    image: Optional[str] = None
    brand: str = ""
    price: Optional[float] = None
    discount_percentage: float = 0
    price_after_discount: Optional[float] = None
    in_stock: Optional[bool] = None
    score: Scores
    matched_terms: list


class Engines(BaseModel):
    keyword: bool
    semantic: bool
    llm: bool


class RecommendResponse(BaseModel):
    occasion: str
    mode_requested: str
    mode_used: str
    engines: Engines
    recognised_occasions: list
    expanded_terms: list
    warnings: list
    total_matches: int
    returned: int
    elapsed_ms: float
    results: list[ProductOut]
