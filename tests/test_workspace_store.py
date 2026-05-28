import shutil
import uuid
from pathlib import Path

from pageindex.workspace_store import PAGE_CONTENT_DIR, WorkspaceStore


def make_temp_dir():
    path = Path("artifacts/test-tmp") / f"store-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_workspace_store_rebuilds_meta_when_index_is_corrupt():
    temp_dir = make_temp_dir()
    try:
        store = WorkspaceStore(temp_dir)
        doc = {
            "id": "doc-1",
            "type": "md",
            "path": "demo.md",
            "doc_name": "demo",
            "doc_description": "desc",
            "line_count": 3,
            "structure": [{"title": "Title", "text": "hello"}],
        }
        store.save_doc("doc-1", doc)
        (temp_dir / "_meta.json").write_text("[]", encoding="utf-8")

        documents = store.load_documents()

        assert "doc-1" in documents
        assert documents["doc-1"]["doc_name"] == "demo"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_workspace_store_load_payload():
    temp_dir = make_temp_dir()
    try:
        store = WorkspaceStore(temp_dir)
        doc = {
            "id": "doc-2",
            "type": "pdf",
            "path": "demo.pdf",
            "doc_name": "demo",
            "doc_description": "",
            "page_count": 1,
            "structure": [{"title": "Title", "text": "abc"}],
            "pages": [{"page": 1, "content": "abc"}],
        }
        store.save_doc("doc-2", doc)

        payload = store.load_doc_payload("doc-2")

        assert payload["pages"][0]["content"] == "abc"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_workspace_store_saves_dedicated_page_payload_without_structure():
    temp_dir = make_temp_dir()
    try:
        store = WorkspaceStore(temp_dir)
        store.save_page_payload("doc-pages", [{"page": 1, "content": "全文第一页"}])

        payload = store.load_page_payload("doc-pages")

        assert payload == {
            "doc_id": "doc-pages",
            "page_count": 1,
            "pages": [{"page": 1, "content": "全文第一页"}],
        }
        assert "structure" not in payload
        assert (temp_dir / PAGE_CONTENT_DIR / "doc-pages.json").is_file()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_workspace_store_missing_meta_is_silent(capsys):
    temp_dir = make_temp_dir()
    try:
        store = WorkspaceStore(temp_dir)

        assert store.read_meta() is None
        assert capsys.readouterr().out == ""
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_workspace_store_does_not_rebuild_retrieval_cache_as_document():
    temp_dir = make_temp_dir()
    try:
        store = WorkspaceStore(temp_dir)
        cache_dir = temp_dir / "_retrieval" / "vector"
        cache_dir.mkdir(parents=True)
        (cache_dir / "index.json").write_text('{"chunks": []}', encoding="utf-8")

        assert store.rebuild_meta() == {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
