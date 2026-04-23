import json
import shutil
import uuid
from pathlib import Path

from pageindex.client import PageIndexClient
from pageindex.identity import build_doc_id, build_tree_id, compute_file_sha256
from pageindex.logging_utils import JsonLogger, emit_progress_event


def make_temp_dir():
    path = Path("artifacts/test-tmp") / f"client-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_client_hybrid_index_emits_progress_and_persists_workspace(monkeypatch):
    temp_dir = make_temp_dir()
    try:
        workspace = temp_dir / "workspace"
        pdf_path = temp_dir / "demo.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

        def fake_prepare(file_path, output_dir, progress_callback=None, progress_logger=None):
            md_path = temp_dir / "demo.md"
            json_path = temp_dir / "demo.json"
            md_path.write_text("# Root\n", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")
            emit_progress_event(
                "converting_pdf",
                "Converting PDF to markdown/json with opendataloader-pdf",
                doc_name=Path(file_path).name,
                progress_callback=progress_callback,
                progress_logger=progress_logger,
            )
            return md_path, json_path

        def fake_run(source_path, md_path, json_path, opt, summary_token_threshold, progress_callback=None, progress_logger=None):
            for stage, message in [
                ("loading_hybrid_sources", "Loading markdown and JSON hybrid sources"),
                ("aligning_headings", "Aligning markdown headings with PDF JSON"),
                ("reconstructing_tree", "Reconstructing corrected hierarchy and page intervals"),
                ("generating_summaries", "Generating hybrid node summaries"),
            ]:
                emit_progress_event(
                    stage,
                    message,
                    doc_name=Path(source_path).name,
                    progress_callback=progress_callback,
                    progress_logger=progress_logger,
                )
            return (
                {
                    "doc_name": "demo",
                    "doc_description": "hybrid document",
                    "structure": [
                        {
                            "title": "Root",
                            "start_page": 1,
                            "end_page": 2,
                            "summary": "summary",
                        }
                    ],
                },
                {
                    "kids": [
                        {"page number": 1, "content": "page one text"},
                        {"page number": 2, "content": "page two text"},
                    ]
                },
            )

        monkeypatch.setattr("pageindex.client.prepare_hybrid_sources_from_pdf", fake_prepare)
        monkeypatch.setattr("pageindex.client.run_hybrid_pipeline_for_sources", fake_run)

        events = []
        client = PageIndexClient(workspace=workspace)
        doc_id = client.index(str(pdf_path), strategy="hybrid", progress_callback=events.append)

        stages = [event["stage"] for event in events]
        assert stages == [
            "starting",
            "preparing_sources",
            "converting_pdf",
            "loading_hybrid_sources",
            "aligning_headings",
            "reconstructing_tree",
            "generating_summaries",
            "caching_pages",
            "saving_workspace",
            "completed",
        ]
        assert doc_id == build_doc_id(compute_file_sha256(pdf_path))

        reloaded_client = PageIndexClient(workspace=workspace)
        document = json.loads(reloaded_client.get_document(doc_id))
        structure = json.loads(reloaded_client.get_document_structure(doc_id))
        page_content = json.loads(reloaded_client.get_page_content(doc_id, "2"))
        expected_tree_id = build_tree_id(structure, index_strategy="hybrid", model=reloaded_client.model, doc_description="hybrid document")

        assert structure[0]["title"] == "Root"
        assert structure[0]["summary"] == "summary"
        assert page_content == [{"page": 2, "content": "page two text"}]
        assert document["tree_id"] == expected_tree_id
        assert reloaded_client.get_tree_id(doc_id) == expected_tree_id
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_client_index_prints_progress_without_callback(monkeypatch, capsys):
    temp_dir = make_temp_dir()
    try:
        workspace = temp_dir / "workspace"
        pdf_path = temp_dir / "demo.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

        monkeypatch.setattr(
            "pageindex.client.prepare_hybrid_sources_from_pdf",
            lambda file_path, output_dir, progress_callback=None, progress_logger=None: (temp_dir / "demo.md", temp_dir / "demo.json"),
        )
        monkeypatch.setattr(
            "pageindex.client.run_hybrid_pipeline_for_sources",
            lambda source_path, md_path, json_path, opt, summary_token_threshold, progress_callback=None, progress_logger=None: (
                {"doc_name": "demo", "doc_description": "", "structure": []},
                {"kids": []},
            ),
        )

        client = PageIndexClient(workspace=workspace)
        client.index(str(pdf_path), strategy="hybrid")

        output = capsys.readouterr().out
        assert "[  0%] [starting]" in output
        assert "[100%] [completed]" in output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_client_reuses_same_doc_id_for_same_pdf_bytes_across_paths(monkeypatch):
    temp_dir = make_temp_dir()
    try:
        workspace = temp_dir / "workspace"
        pdf_path_a = temp_dir / "demo-a.pdf"
        pdf_path_b = temp_dir / "demo-b.pdf"
        pdf_bytes = b"%PDF-1.4\n%same-payload\n"
        pdf_path_a.write_bytes(pdf_bytes)
        pdf_path_b.write_bytes(pdf_bytes)

        calls = {"prepare": 0, "run": 0}

        def fake_prepare(file_path, output_dir, progress_callback=None, progress_logger=None):
            calls["prepare"] += 1
            md_path = temp_dir / "demo.md"
            json_path = temp_dir / "demo.json"
            md_path.write_text("# Root\n", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")
            emit_progress_event(
                "converting_pdf",
                "Converting PDF to markdown/json with opendataloader-pdf",
                doc_name=Path(file_path).name,
                progress_callback=progress_callback,
                progress_logger=progress_logger,
            )
            return md_path, json_path

        def fake_run(source_path, md_path, json_path, opt, summary_token_threshold, progress_callback=None, progress_logger=None):
            calls["run"] += 1
            return (
                {
                    "doc_name": "demo",
                    "doc_description": "cached document",
                    "structure": [{"title": "Root", "start_page": 1, "end_page": 1, "summary": "root"}],
                },
                {"kids": [{"page number": 1, "content": "page one"}]},
            )

        monkeypatch.setattr("pageindex.client.prepare_hybrid_sources_from_pdf", fake_prepare)
        monkeypatch.setattr("pageindex.client.run_hybrid_pipeline_for_sources", fake_run)

        client = PageIndexClient(workspace=workspace)
        first_doc_id = client.index(str(pdf_path_a), strategy="hybrid")
        second_doc_id = client.index(str(pdf_path_b), strategy="hybrid")

        assert first_doc_id == second_doc_id == build_doc_id(compute_file_sha256(pdf_path_a))
        assert calls == {"prepare": 1, "run": 1}

        reloaded_client = PageIndexClient(workspace=workspace)
        reloaded_doc = json.loads(reloaded_client.get_document(first_doc_id))
        assert reloaded_doc["doc_id"] == first_doc_id
        assert reloaded_doc["tree_id"].startswith("tree_")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_json_logger_supports_standard_logging_args(tmp_path):
    logger = JsonLogger("demo.pdf", base_dir=str(tmp_path))

    logger.debug("Tree node %s spans %s pages", "Root", 3, node_id="001")

    log_file = next(tmp_path.glob("*.json"))
    payload = json.loads(log_file.read_text(encoding="utf-8"))

    assert payload[-1]["level"] == "DEBUG"
    assert payload[-1]["message"] == "Tree node Root spans 3 pages"
    assert payload[-1]["node_id"] == "001"
