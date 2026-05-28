from __future__ import annotations

import re


def normalize_reference_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def filter_valid_reference_pages(pages: list[int], valid_pages: set[int], limit: int = 3) -> list[int]:
    result = []
    for page in pages:
        if page in valid_pages and page not in result:
            result.append(page)
        if len(result) >= limit:
            break
    return result


__all__ = ["filter_valid_reference_pages", "normalize_reference_text"]
