import asyncio
import concurrent.futures
import os
from pathlib import Path

import PyPDF2

from .config import ConfigLoader
from .hybrid_index import build_pdf_pages_from_json_payload, prepare_hybrid_sources_from_pdf, run_hybrid_pipeline_for_sources
from .identity import build_doc_id, build_tree_id, compute_file_sha256
from .logging_utils import emit_progress_event
from .markdown import md_to_tree
from .page_index import page_index
from .retrieve import get_document, get_document_structure, get_page_content
from .workspace_store import WorkspaceStore


def _normalize_retrieve_model(model: str) -> str:
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()
    """

    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, workspace: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        self.summary_max_concurrency = getattr(opt, "summary_max_concurrency", None)
        self.store = WorkspaceStore(self.workspace) if self.workspace else None
        self.documents = self.store.load_documents() if self.store else {}

    def index(
        self,
        file_path: str,
        mode: str = "auto",
        strategy: str = "standard",
        json_path: str = None,
        hybrid_output_dir: str = None,
        progress_callback=None,
        progress_logger=None,
    ) -> str:
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_name = Path(file_path).name
        source_sha256 = compute_file_sha256(file_path)
        doc_id = build_doc_id(source_sha256)
        ext = os.path.splitext(file_path)[1].lower()
        is_pdf = ext == ".pdf"
        is_md = ext in [".md", ".markdown"]
        normalized_mode = mode
        if mode == "auto":
            normalized_mode = "pdf" if is_pdf else "md" if is_md else mode

        emit_progress_event(
            "starting",
            f"Starting index build for {doc_name}",
            doc_name=doc_name,
            extra={"mode": normalized_mode, "strategy": strategy, "doc_id": doc_id, "source_sha256": source_sha256},
            progress_callback=progress_callback,
            progress_logger=progress_logger,
        )

        cached_doc = self.documents.get(doc_id)
        if cached_doc and cached_doc.get("source_sha256") == source_sha256 and cached_doc.get("index_strategy") == strategy:
            cached_doc["path"] = file_path
            emit_progress_event(
                "completed",
                f"Cache hit. Reusing indexed document {doc_id}",
                doc_name=doc_name,
                extra={"doc_id": doc_id, "tree_id": cached_doc.get("tree_id", ""), "cache_hit": True},
                progress_callback=progress_callback,
                progress_logger=progress_logger,
            )
            if self.store:
                self.store.save_meta(doc_id, self.store.make_meta_entry(cached_doc))
            return doc_id

        try:
            if (mode == "pdf" or (mode == "auto" and is_pdf)) and strategy == "hybrid":
                emit_progress_event(
                    "preparing_sources",
                    "Preparing hybrid indexing sources",
                    doc_name=doc_name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                output_dir = hybrid_output_dir or (str(self.workspace) if self.workspace else "artifacts/results")
                opt = ConfigLoader().load(
                    {
                        "model": self.model,
                        "if_add_node_summary": "yes",
                        "if_add_doc_description": "yes",
                        "if_add_node_text": "no",
                        "if_add_node_id": "yes",
                        "summary_max_concurrency": self.summary_max_concurrency,
                    }
                )
                md_path, resolved_json_path = prepare_hybrid_sources_from_pdf(
                    file_path,
                    output_dir,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                if json_path:
                    resolved_json_path = json_path
                result, pdf_json_payload = run_hybrid_pipeline_for_sources(
                    source_path=file_path,
                    md_path=md_path,
                    json_path=resolved_json_path,
                    opt=opt,
                    summary_token_threshold=200,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                emit_progress_event(
                    "caching_pages",
                    "Caching per-page text extracted from hybrid JSON",
                    doc_name=doc_name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                pages = build_pdf_pages_from_json_payload(pdf_json_payload)
                self.documents[doc_id] = {
                    "id": doc_id,
                    "type": "pdf",
                    "source_sha256": source_sha256,
                    "index_strategy": "hybrid",
                    "path": file_path,
                    "doc_name": result.get("doc_name", ""),
                    "doc_description": result.get("doc_description", ""),
                    "page_count": len(pages),
                    "structure": result["structure"],
                    "pages": pages,
                }
            elif mode == "pdf" or (mode == "auto" and is_pdf):
                emit_progress_event(
                    "preparing_sources",
                    "Reading PDF and building standard PageIndex tree",
                    doc_name=doc_name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                result = page_index(
                    doc=file_path,
                    model=self.model,
                    if_add_node_summary="yes",
                    if_add_node_text="yes",
                    if_add_node_id="yes",
                    if_add_doc_description="yes",
                )
                emit_progress_event(
                    "caching_pages",
                    "Caching per-page PDF text",
                    doc_name=doc_name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                pages = []
                with open(file_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(pdf_reader.pages, 1):
                        pages.append({"page": i, "content": page.extract_text() or ""})

                self.documents[doc_id] = {
                    "id": doc_id,
                    "type": "pdf",
                    "source_sha256": source_sha256,
                    "index_strategy": "standard",
                    "path": file_path,
                    "doc_name": result.get("doc_name", ""),
                    "doc_description": result.get("doc_description", ""),
                    "page_count": len(pages),
                    "structure": result["structure"],
                    "pages": pages,
                }
            elif mode == "md" or (mode == "auto" and is_md):
                emit_progress_event(
                    "preparing_sources",
                    "Indexing markdown document",
                    doc_name=doc_name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
                coro = md_to_tree(
                    md_path=file_path,
                    if_thinning=False,
                    if_add_node_summary="yes",
                    summary_token_threshold=200,
                    model=self.model,
                    if_add_doc_description="yes",
                    if_add_node_text="yes",
                    if_add_node_id="yes",
                    summary_max_concurrency=self.summary_max_concurrency,
                )
                try:
                    asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(asyncio.run, coro).result()
                except RuntimeError:
                    result = asyncio.run(coro)
                self.documents[doc_id] = {
                    "id": doc_id,
                    "type": "md",
                    "source_sha256": source_sha256,
                    "index_strategy": strategy,
                    "path": file_path,
                    "doc_name": result.get("doc_name", ""),
                    "doc_description": result.get("doc_description", ""),
                    "line_count": result.get("line_count", 0),
                    "structure": result["structure"],
                }
            else:
                raise ValueError(f"Unsupported file format for: {file_path}")
        except Exception as exc:
            emit_progress_event(
                "failed",
                f"Indexing failed: {exc}",
                doc_name=doc_name,
                extra={"error_type": type(exc).__name__},
                progress_callback=progress_callback,
                progress_logger=progress_logger,
            )
            raise

        self.documents[doc_id]["tree_id"] = build_tree_id(
            self.documents[doc_id]["structure"],
            index_strategy=self.documents[doc_id].get("index_strategy", ""),
            model=self.model,
            doc_description=self.documents[doc_id].get("doc_description", ""),
        )

        if self.store:
            emit_progress_event(
                "saving_workspace",
                "Saving indexed document to workspace",
                doc_name=doc_name,
                extra={"workspace": str(self.workspace)},
                progress_callback=progress_callback,
                progress_logger=progress_logger,
            )
            self._save_doc(doc_id)
        emit_progress_event(
            "completed",
            f"Indexing complete. Document ID: {doc_id}",
            doc_name=doc_name,
            extra={"doc_id": doc_id, "tree_id": self.documents[doc_id].get("tree_id", ""), "cache_hit": False},
            progress_callback=progress_callback,
            progress_logger=progress_logger,
        )
        return doc_id

    def _save_doc(self, doc_id: str):
        self.store.save_doc(doc_id, self.documents[doc_id])
        self.documents[doc_id].pop("structure", None)
        self.documents[doc_id].pop("pages", None)

    def _ensure_doc_loaded(self, doc_id: str):
        doc = self.documents.get(doc_id)
        if not doc or doc.get("structure") is not None:
            return
        full = self.store.load_doc_payload(doc_id)
        if not full:
            return
        doc["structure"] = full.get("structure", [])
        if full.get("pages"):
            doc["pages"] = full["pages"]

    def get_document(self, doc_id: str) -> str:
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        if self.store:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        if self.store:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)

    def get_tree_id(self, doc_id: str) -> str:
        doc = self.documents.get(doc_id)
        if not doc:
            return ""
        return str(doc.get("tree_id", ""))


'''
import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

import PyPDF2

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader, remove_fields

META_INDEX = "_meta.json"


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """
    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, workspace: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})

            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'pdf',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'md',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0),
                'structure': result['structure'],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            'type': doc.get('type', ''),
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'path': doc.get('path', ''),
        }
        if doc.get('type') == 'pdf':
            entry['page_count'] = doc.get('page_count')
        elif doc.get('type') == 'md':
            entry['line_count'] = doc.get('line_count')
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        self.documents[doc_id].pop('structure', None)
        self.documents[doc_id].pop('pages', None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get('path') and not os.path.isabs(doc['path']):
                doc['path'] = str((self.workspace / doc['path']).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get('structure') is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc['structure'] = full.get('structure', [])
        if full.get('pages'):
            doc['pages'] = full['pages']

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """Return document tree structure JSON (without text fields)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
'''
