import copy
import hashlib
import json
from pathlib import Path

from .tree_utils import remove_fields


def compute_file_sha256(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    path = Path(file_path)
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_doc_id(source_sha256):
    return f"doc_{source_sha256}"


def _normalize_for_hash(data):
    if isinstance(data, dict):
        return {key: _normalize_for_hash(value) for key, value in sorted(data.items())}
    if isinstance(data, list):
        return [_normalize_for_hash(item) for item in data]
    return data


def canonicalize_structure(structure):
    trimmed = remove_fields(copy.deepcopy(structure), fields=["text"])
    normalized = _normalize_for_hash(trimmed)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_tree_id(structure, *, index_strategy="", model="", doc_description=""):
    payload = {
        "index_strategy": index_strategy or "",
        "model": model or "",
        "doc_description": doc_description or "",
        "structure": json.loads(canonicalize_structure(structure)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"tree_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


__all__ = [
    "build_doc_id",
    "build_tree_id",
    "canonicalize_structure",
    "compute_file_sha256",
]
