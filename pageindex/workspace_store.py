import json
import os
from pathlib import Path

from .tree_utils import remove_fields

META_INDEX = "_meta.json"


class WorkspaceStore:
    def __init__(self, workspace):
        self.workspace = Path(workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    @staticmethod
    def make_meta_entry(doc):
        entry = {
            "type": doc.get("type", ""),
            "source_sha256": doc.get("source_sha256", ""),
            "tree_id": doc.get("tree_id", ""),
            "index_strategy": doc.get("index_strategy", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
        }
        if doc.get("type") == "pdf":
            entry["page_count"] = doc.get("page_count")
        elif doc.get("type") == "md":
            entry["line_count"] = doc.get("line_count")
        return entry

    def read_meta(self):
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def rebuild_meta(self):
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self.make_meta_entry(doc)
        return meta

    def save_meta(self, doc_id, entry):
        meta = self.read_meta() or self.rebuild_meta()
        meta[doc_id] = entry
        with open(self.workspace / META_INDEX, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def save_doc(self, doc_id, doc):
        payload = doc.copy()
        if payload.get("structure") and payload.get("type") == "pdf":
            payload["structure"] = remove_fields(payload["structure"], fields=["text"])
        with open(self.workspace / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.save_meta(doc_id, self.make_meta_entry(payload))

    def load_documents(self):
        meta = self.read_meta()
        if meta is None:
            meta = self.rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        documents = {}
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not os.path.isabs(doc["path"]):
                doc["path"] = str((self.workspace / doc["path"]).resolve())
            documents[doc_id] = doc
        return documents

    def load_doc_payload(self, doc_id):
        return self._read_json(self.workspace / f"{doc_id}.json")


__all__ = ["META_INDEX", "WorkspaceStore"]
