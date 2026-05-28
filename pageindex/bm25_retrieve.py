from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def tokenize(text: str) -> list[str]:
    """Tokenize English/numeric words and overlapping Chinese bigrams."""
    text = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9._/-]*", text)
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _deepest_section_path(structure: list[dict], page: int) -> str:
    best_path = ""

    def visit(nodes: list[dict], path: list[str]) -> None:
        nonlocal best_path
        for node in nodes or []:
            start = node.get("start_page", node.get("start_index"))
            end = node.get("end_page", node.get("end_index", start))
            if not isinstance(start, int) or not isinstance(end, int) or not (start <= page <= end):
                continue
            title = str(node.get("title", "") or "").strip()
            current = path + ([title] if title else [])
            best_path = " > ".join(current)
            visit(node.get("nodes", []), current)

    visit(structure or [], [])
    return best_path


def build_page_documents(pages: list[dict], structure: list[dict]) -> list[dict]:
    documents = []
    for item in pages or []:
        page = item.get("page")
        if not isinstance(page, int):
            continue
        content = str(item.get("content", "") or "")
        section_path = _deepest_section_path(structure, page)
        index_text = "\n".join(text for text in (section_path, content) if text)
        documents.append(
            {
                "page": page,
                "content": content,
                "section_path": section_path,
                "index_text": index_text,
                "tokens": tokenize(index_text),
            }
        )
    return documents


def search_bm25(documents: list[dict], queries: list[str], top_k: int) -> list[dict]:
    if not documents or not queries or top_k < 1:
        return []
    query_tokens = tokenize(" ".join(str(query) for query in queries if str(query).strip()))
    if not query_tokens:
        return []

    token_counts = [Counter(document["tokens"]) for document in documents]
    doc_lengths = [sum(counts.values()) for counts in token_counts]
    avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    document_frequency = defaultdict(int)
    for counts in token_counts:
        for token in counts:
            document_frequency[token] += 1

    scores = []
    query_counts = Counter(query_tokens)
    k1 = 1.5
    b = 0.75
    total_docs = len(documents)
    for document, counts, doc_length in zip(documents, token_counts, doc_lengths):
        score = 0.0
        for token, query_frequency in query_counts.items():
            term_frequency = counts.get(token, 0)
            if term_frequency == 0:
                continue
            idf = math.log(1.0 + (total_docs - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * (doc_length / (avg_length or 1.0)))
            score += query_frequency * idf * (term_frequency * (k1 + 1.0) / denominator)
        if score > 0:
            scores.append(
                {
                    "page": document["page"],
                    "section_path": document["section_path"],
                    "raw_score": score,
                }
            )

    scores.sort(key=lambda item: (-item["raw_score"], item["page"]))
    for rank, item in enumerate(scores[:top_k], start=1):
        item["rank"] = rank
    return scores[:top_k]


def retrieve_bm25_candidates(
    documents: list[dict],
    primary_queries: list[str],
    expanded_keywords: list[str],
    *,
    primary_top_k: int = 10,
    expanded_top_k: int = 20,
    primary_weight: float = 3.0,
    expanded_weight: float = 1.0,
    rrf_constant: int = 60,
) -> list[dict]:
    primary = search_bm25(documents, primary_queries, primary_top_k)
    expanded = search_bm25(documents, expanded_keywords, expanded_top_k)
    candidates: dict[int, dict] = {}

    for source, weight, results in (
        ("bm25_primary", primary_weight, primary),
        ("bm25_expanded", expanded_weight, expanded),
    ):
        for result in results:
            candidate = candidates.setdefault(
                result["page"],
                {
                    "page": result["page"],
                    "sources": [],
                    "section_path": result["section_path"],
                    "bm25_score": 0.0,
                    "bm25_ranks": {},
                    "bm25_raw_scores": {},
                    "reason": "",
                    "read_status": "pending",
                },
            )
            candidate["sources"].append(source)
            candidate["bm25_ranks"][source] = result["rank"]
            candidate["bm25_raw_scores"][source] = result["raw_score"]
            candidate["bm25_score"] += weight / (rrf_constant + result["rank"])

    for candidate in candidates.values():
        descriptions = []
        if "bm25_primary" in candidate["sources"]:
            descriptions.append("主查询命中")
        if "bm25_expanded" in candidate["sources"]:
            descriptions.append("扩展关键词命中")
        candidate["reason"] = "；".join(descriptions)

    return sorted(candidates.values(), key=lambda item: (-item["bm25_score"], item["page"]))


__all__ = ["build_page_documents", "retrieve_bm25_candidates", "search_bm25", "tokenize"]
