import json
import shutil
import uuid
from pathlib import Path

from pageindex import cli


def make_temp_dir():
    path = Path("artifacts/test-tmp") / f"cli-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_payload(path, total_pages, kids):
    path.write_text(
        json.dumps(
            {
                "file name": "demo.pdf",
                "number of pages": total_pages,
                "kids": kids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_cli_markdown_writes_output():
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        md_path.write_text("# Title\n\nbody\n", encoding="utf-8")
        output_dir = temp_dir / "out"

        cli.main(
            [
                "--md_path",
                str(md_path),
                "--if-add-node-summary",
                "no",
                "--if-add-doc-description",
                "no",
                "--if-add-node-text",
                "yes",
                "--output-dir",
                str(output_dir),
            ]
        )

        payload = json.loads((output_dir / "demo_structure.json").read_text(encoding="utf-8"))
        assert payload["doc_name"] == "demo"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_pdf_smoke_with_stub(monkeypatch):
    temp_dir = make_temp_dir()
    try:
        pdf_path = temp_dir / "demo.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        output_dir = temp_dir / "out"

        monkeypatch.setattr(
            cli,
            "page_index_main",
            lambda path, opt: {"doc_name": "demo", "structure": [{"title": "Root"}]},
        )

        cli.main(["--pdf_path", str(pdf_path), "--output-dir", str(output_dir)])

        payload = json.loads((output_dir / "demo_structure.json").read_text(encoding="utf-8"))
        assert payload["structure"][0]["title"] == "Root"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_markdown_hybrid_writes_output(monkeypatch):
    temp_dir = make_temp_dir()
    try:
        md_path = temp_dir / "demo.md"
        json_path = temp_dir / "demo.json"
        md_path.write_text("# Chapter One\n\nbody\n\n# Chapter Two\n\nmore\n", encoding="utf-8")
        write_json_payload(
            json_path,
            total_pages=2,
            kids=[
                {"type": "heading", "page number": 1, "heading level": 1, "content": "Chapter One"},
                {"type": "paragraph", "page number": 1, "content": "body"},
                {"type": "heading", "page number": 2, "heading level": 1, "content": "Chapter Two"},
                {"type": "paragraph", "page number": 2, "content": "more"},
            ],
        )
        output_dir = temp_dir / "out"

        def fake_run_hybrid_pipeline_for_sources(source_path, md_path, json_path, opt, summary_token_threshold):
            assert Path(md_path).name == "demo.md"
            assert Path(json_path).name == "demo.json"
            return {
                "doc_name": "demo",
                "line_count": 6,
                "structure": [
                    {"title": "Chapter One", "start_page": 1},
                    {"title": "Chapter Two", "start_page": 2},
                ],
            }

        monkeypatch.setattr(cli, "_run_hybrid_pipeline_for_sources", fake_run_hybrid_pipeline_for_sources)

        cli.main(
            [
                "--md_path",
                str(md_path),
                "--md-hybrid",
                "--if-add-node-summary",
                "no",
                "--if-add-doc-description",
                "no",
                "--if-add-node-text",
                "yes",
                "--output-dir",
                str(output_dir),
            ]
        )

        payload = json.loads((output_dir / "demo_structure.json").read_text(encoding="utf-8"))
        assert payload["structure"][0]["start_page"] == 1
        assert payload["structure"][1]["start_page"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_pdf_hybrid_uses_adapter(monkeypatch):
    temp_dir = make_temp_dir()
    try:
        pdf_path = temp_dir / "demo.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        output_dir = temp_dir / "out"

        hybrid_md = temp_dir / "hybrid" / "demo.md"
        hybrid_json = temp_dir / "hybrid" / "demo.json"
        hybrid_md.parent.mkdir(parents=True, exist_ok=True)
        hybrid_md.write_text("# Demo\n", encoding="utf-8")
        hybrid_json.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(cli, "_prepare_hybrid_sources_from_pdf", lambda path, out: (hybrid_md, hybrid_json))
        monkeypatch.setattr(
            cli,
            "_run_hybrid_pipeline_for_sources",
            lambda source_path, md_path, json_path, opt, summary_token_threshold: {
                "doc_name": "demo",
                "structure": [{"title": "Hybrid Root", "start_page": 1}],
            },
        )

        cli.main(["--pdf_path", str(pdf_path), "--md-hybrid", "--output-dir", str(output_dir)])

        payload = json.loads((output_dir / "demo_structure.json").read_text(encoding="utf-8"))
        assert payload["structure"][0]["title"] == "Hybrid Root"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
