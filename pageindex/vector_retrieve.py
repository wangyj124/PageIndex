from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path

import litellm

from .llm import count_tokens


VECTOR_INDEX_VERSION = "v1"
logger = logging.getLogger(__name__)


def _safe_token_count(text: str) -> int:
    try:
        return count_tokens(text)
    except Exception:
        return max(1, len(text) // 2)


def _split_long_text(text: str, token_limit: int) -> list[str]:
    if _safe_token_count(text) <= token_limit:
        return [text]
    chunks = []
    current = ""
    for char in text:
        candidate = current + char
        if current and _safe_token_count(candidate) > token_limit:
            chunks.append(current)
            current = char
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_vector_chunks(
    page_documents: list[dict],
    *,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[dict]:
    chunks = []
    for document in page_documents:
        page = document["page"]
        paragraphs = [part.strip() for part in str(document.get("content", "")).splitlines() if part.strip()]
        if not paragraphs:
            continue
        expanded = []
        for paragraph in paragraphs:
            expanded.extend(_split_long_text(paragraph, chunk_tokens))

        current_parts: list[str] = []
        for paragraph in expanded:
            candidate = "\n".join([*current_parts, paragraph])
            if current_parts and _safe_token_count(candidate) > chunk_tokens:
                text = "\n".join(current_parts)
                chunks.append(
                    {
                        "chunk_id": f"p{page}_c{len([item for item in chunks if item['page'] == page]) + 1:03d}",
                        "page": page,
                        "section_path": document.get("section_path", ""),
                        "text": text,
                    }
                )
                overlap = ""
                for char in reversed(text):
                    candidate_overlap = char + overlap
                    if overlap and _safe_token_count(candidate_overlap) > overlap_tokens:
                        break
                    overlap = candidate_overlap
                current_parts = [overlap, paragraph] if overlap else [paragraph]
                while overlap and _safe_token_count("\n".join(current_parts)) > chunk_tokens:
                    overlap = overlap[1:]
                    current_parts = [overlap, paragraph] if overlap else [paragraph]
            else:
                current_parts.append(paragraph)
        if current_parts:
            chunks.append(
                {
                    "chunk_id": f"p{page}_c{len([item for item in chunks if item['page'] == page]) + 1:03d}",
                    "page": page,
                    "section_path": document.get("section_path", ""),
                    "text": "\n".join(current_parts),
                }
            )
    return chunks


def _index_cache_key(doc_id: str, source_sha256: str, model: str, chunk_tokens: int, overlap_tokens: int) -> str:
    content = "|".join(
        [VECTOR_INDEX_VERSION, doc_id, source_sha256, model, str(chunk_tokens), str(overlap_tokens)]
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _call_embedding(texts: list[str], model: str, embedding_fn=None) -> list[list[float]]:
    if embedding_fn is not None:
        result = embedding_fn(texts, model)
        if inspect.isawaitable(result):
            result = await result
        return result

    api_base = os.getenv("PAGEINDEX_VECTOR_API_BASE", "").strip()
    api_key = os.getenv("PAGEINDEX_VECTOR_API_KEY", "").strip()
    if not api_base or not api_key:
        raise ValueError("vector embedding service is not configured")

    def request():
        response = litellm.embedding(model=model, input=texts, api_base=api_base, api_key=api_key)
        vectors = []
        for item in response.data:
            vectors.append(item["embedding"] if isinstance(item, dict) else item.embedding)
        return vectors

    return await asyncio.to_thread(request)


def _embedding_batches(
    texts: list[str],
    *,
    batch_size: int,
    request_token_budget: int,
) -> list[list[str]]:
    batch_size = max(1, int(batch_size))
    request_token_budget = max(1, int(request_token_budget))
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for text in texts:
        token_count = max(1, _safe_token_count(text))
        if current and (
            len(current) >= batch_size or current_tokens + token_count > request_token_budget
        ):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += token_count

    if current:
        batches.append(current)
    return batches


async def _call_embedding_batched(
    texts: list[str],
    model: str,
    *,
    batch_size: int,
    request_token_budget: int,
    embedding_fn=None,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for batch in _embedding_batches(
        texts,
        batch_size=batch_size,
        request_token_budget=request_token_budget,
    ):
        batch_embeddings = await _call_embedding(batch, model, embedding_fn=embedding_fn)
        if len(batch_embeddings) != len(batch):
            raise ValueError("embedding response did not match batch input count")
        embeddings.extend(batch_embeddings)
    return embeddings


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_path, path)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


async def retrieve_vector_candidates(
    *,
    doc_id: str,
    source_sha256: str,
    page_documents: list[dict],
    query_text: str,
    processed_pages: set[int],
    workspace: Path | None,
    embedding_model: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
    chunk_top_k: int = 30,
    page_limit: int = 5,
    context_token_budget: int = 10000,
    embedding_batch_size: int = 64,
    embedding_request_token_budget: int = 8192,
    cache_enabled: bool = True,
    embedding_fn=None,
) -> list[dict]:
    chunks = build_vector_chunks(page_documents, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return []

    cache_payload = None
    cache_path = None
    if workspace is not None and cache_enabled:
        cache_key = _index_cache_key(doc_id, source_sha256, embedding_model, chunk_tokens, overlap_tokens)
        cache_path = Path(workspace) / "_retrieval" / "vector" / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cache_payload = None

    if cache_payload and len(cache_payload.get("chunks", [])) == len(chunks):
        logger.debug("Vector index cache hit for doc_id=%s", doc_id)
        indexed_chunks = cache_payload["chunks"]
    else:
        logger.debug("Building vector index for doc_id=%s chunks=%s", doc_id, len(chunks))
        embeddings = await _call_embedding_batched(
            [item["text"] for item in chunks],
            embedding_model,
            batch_size=embedding_batch_size,
            request_token_budget=embedding_request_token_budget,
            embedding_fn=embedding_fn,
        )
        if len(embeddings) != len(chunks):
            raise ValueError("embedding response did not match chunk count")
        indexed_chunks = [{**chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
        if cache_path is not None:
            _write_json_atomic(
                cache_path,
                {
                    "version": VECTOR_INDEX_VERSION,
                    "embedding_model": embedding_model,
                    "chunks": indexed_chunks,
                },
            )

    query_vectors = await _call_embedding_batched(
        [query_text],
        embedding_model,
        batch_size=embedding_batch_size,
        request_token_budget=embedding_request_token_budget,
        embedding_fn=embedding_fn,
    )
    if not query_vectors:
        return []
    query_vector = query_vectors[0]
    scored_chunks = [
        {**chunk, "similarity": _cosine_similarity(query_vector, chunk["embedding"])}
        for chunk in indexed_chunks
        if chunk["page"] not in processed_pages
    ]
    scored_chunks.sort(key=lambda item: (-item["similarity"], item["page"], item["chunk_id"]))
    page_scores: dict[int, dict] = {}
    for chunk in scored_chunks[:chunk_top_k]:
        current = page_scores.get(chunk["page"])
        if current is None or chunk["similarity"] > current["vector_score"]:
            page_scores[chunk["page"]] = {
                "page": chunk["page"],
                "sources": ["vector_fallback"],
                "section_path": chunk.get("section_path", ""),
                "vector_score": chunk["similarity"],
                "reason": "向量兜底命中",
                "read_status": "pending",
            }
    page_documents_by_page = {document["page"]: document for document in page_documents}
    selected = []
    used_tokens = 0
    context_token_budget = max(1, int(context_token_budget))
    for candidate in sorted(page_scores.values(), key=lambda item: (-item["vector_score"], item["page"])):
        if len(selected) >= page_limit:
            break
        page_tokens = _safe_token_count(page_documents_by_page[candidate["page"]].get("content", ""))
        if page_tokens > context_token_budget or used_tokens + page_tokens > context_token_budget:
            continue
        used_tokens += page_tokens
        selected.append(candidate)
    return selected


__all__ = ["VECTOR_INDEX_VERSION", "build_vector_chunks", "retrieve_vector_candidates"]
