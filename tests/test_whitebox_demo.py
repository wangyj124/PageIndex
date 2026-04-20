import json
import shutil
import asyncio
import uuid
from pathlib import Path

from pageindex.identity import build_doc_id, compute_file_sha256
from pageindex.whitebox_demo import run_contract_extraction_whitebox_demo


def make_temp_dir():
    path = Path("artifacts/test-tmp") / f"whitebox-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class FakeClient:
    def __init__(self, doc_id, source_sha256):
        self.doc_id = doc_id
        self.tree_id = "tree_demo_001"
        self.source_sha256 = source_sha256
        self.documents = {
            doc_id: {
                "id": doc_id,
                "source_sha256": source_sha256,
                "index_strategy": "hybrid",
                "tree_id": self.tree_id,
            }
        }
        self.calls = []

    def index(self, file_path, strategy="standard", progress_callback=None, **kwargs):
        self.calls.append(("index", strategy))
        if progress_callback:
            progress_callback({"percent": 0, "stage": "starting", "message": "Starting index", "doc_name": Path(file_path).name, "extra": {}})
            progress_callback(
                {
                    "percent": 100,
                    "stage": "completed",
                    "message": "Cache hit. Reusing indexed document",
                    "doc_name": Path(file_path).name,
                    "extra": {"cache_hit": True},
                }
            )
        return self.doc_id

    def get_document(self, doc_id):
        self.calls.append(("get_document", doc_id))
        return json.dumps(
            {
                "doc_id": doc_id,
                "tree_id": self.tree_id,
                "source_sha256": self.source_sha256,
                "doc_name": "demo-contract",
                "index_strategy": "hybrid",
                "page_count": 6,
            },
            ensure_ascii=False,
        )

    def get_tree_id(self, doc_id):
        return self.tree_id

    def get_document_structure(self, doc_id):
        self.calls.append(("get_document_structure", doc_id))
        return json.dumps(
            [
                {
                    "title": "Pricing Clause",
                    "summary": "This section covers contract amount and payment terms.",
                    "start_page": 4,
                    "end_page": 5,
                    "nodes": [],
                }
            ],
            ensure_ascii=False,
        )

    def get_page_content(self, doc_id, pages):
        self.calls.append(("get_page_content", pages))
        return json.dumps([{"page": 4, "content": "The total contract amount is CNY 1,000,000."}], ensure_ascii=False)


async def fake_agent_executor(name, instructions, prompt, tools, verbose=False, printer=print):
    if name == "context":
        tools[0]()
        tools[1]()
        printer("[context] tool call: get_document")
        printer("[context] tool call: get_document_structure")
        return {"name": name, "reasoning": "reviewed cached metadata", "text": "context ready", "final_output": '{"ok": true}'}

    if name == "orchestrator":
        tools[0]()
        tools[1]()
        printer("[orchestrator] tool call: get_document")
        printer("[orchestrator] tool call: get_document_structure")
        return {
            "name": name,
            "reasoning": "split schema into field workers",
            "text": "dispatching workers",
            "final_output": json.dumps(
                {
                    "assignments": [
                        {"field": "contract_amount", "handoff_message": "Inspect pricing clauses first."},
                        {"field": "signing_date", "handoff_message": "Inspect signature pages."},
                    ]
                },
                ensure_ascii=False,
            ),
        }

    if name == "worker:contract_amount":
        tools[1]()
        tools[2]("4")
        printer("[worker:contract_amount] tool call: get_document_structure")
        printer("[worker:contract_amount] tool call: get_page_content {'pages': '4'}")
        if verbose:
            printer("[worker:contract_amount] tool output: The total contract amount is CNY 1,000,000.")
        return {
            "name": name,
            "reasoning": "pricing clause points to page 4",
            "text": "amount extracted",
            "final_output": json.dumps(
                {
                    "field": "contract_amount",
                    "candidate_pages": [4],
                    "status": "found",
                    "value": "CNY 1,000,000",
                    "evidence": "The total contract amount is CNY 1,000,000.",
                    "pages": [4],
                    "confidence": "High",
                    "reason": None,
                },
                ensure_ascii=False,
            ),
        }

    if name == "worker:signing_date":
        tools[1]()
        tools[2]("6")
        printer("[worker:signing_date] tool call: get_document_structure")
        printer("[worker:signing_date] tool call: get_page_content {'pages': '6'}")
        return {
            "name": name,
            "reasoning": "signature page is ambiguous",
            "text": "date not found",
            "final_output": json.dumps(
                {
                    "field": "signing_date",
                    "candidate_pages": [6],
                    "status": "not_found",
                    "value": "",
                    "evidence": "",
                    "pages": [6],
                    "confidence": "Low",
                    "reason": "No explicit signing date found on the inspected page.",
                },
                ensure_ascii=False,
            ),
        }

    if name == "merge":
        printer("[merge] no document tool calls")
        return {
            "name": name,
            "reasoning": "merged worker outputs",
            "text": "merge complete",
            "final_output": json.dumps(
                {
                    "summary": "1 found, 1 not_found",
                    "results": {
                        "contract_amount": {
                            "field": "contract_amount",
                            "candidate_pages": [4],
                            "status": "found",
                            "value": "CNY 1,000,000",
                            "evidence": "The total contract amount is CNY 1,000,000.",
                            "pages": [4],
                            "confidence": "High",
                            "reason": None,
                        },
                        "signing_date": {
                            "field": "signing_date",
                            "candidate_pages": [6],
                            "status": "not_found",
                            "value": "",
                            "evidence": "",
                            "pages": [6],
                            "confidence": "Low",
                            "reason": "No explicit signing date found on the inspected page.",
                        },
                    },
                },
                ensure_ascii=False,
            ),
        }

    raise AssertionError(f"unexpected agent name: {name}")


def test_whitebox_demo_prints_ids_cache_state_and_worker_details(capsys):
    temp_dir = make_temp_dir()
    try:
        pdf_path = temp_dir / "demo.pdf"
        schema_path = temp_dir / "schema.json"
        workspace = temp_dir / "workspace"
        pdf_path.write_bytes(b"%PDF-1.4\n%demo\n")
        schema_path.write_text(
            json.dumps(
                {
                    "fields": [
                        {"name": "contract_amount", "description": "Contract total amount"},
                        {"name": "signing_date", "description": "Contract signing date"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        source_sha256 = compute_file_sha256(pdf_path)
        doc_id = build_doc_id(source_sha256)
        client = FakeClient(doc_id, source_sha256)

        result = run_contract_extraction_whitebox_demo(
            pdf_path,
            schema_path,
            workspace,
            verbose=False,
            client=client,
            agent_executor=fake_agent_executor,
        )

        output = capsys.readouterr().out
        assert f"[indexing] doc_id={doc_id}" in output
        assert "[indexing] tree_id=tree_demo_001" in output
        assert "[indexing] cache_hit=true" in output
        assert "[orchestrator] assignments=" in output
        assert "[worker:contract_amount] candidate_pages=[4]" in output
        assert "[merge] summary=1 found, 1 not_found" in output
        assert result["results"]["contract_amount"]["confidence"] == "High"
        assert result["results"]["signing_date"]["confidence"] == "Low"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_whitebox_demo_verbose_mode_shows_extra_worker_details(capsys):
    temp_dir = make_temp_dir()
    try:
        pdf_path = temp_dir / "demo.pdf"
        schema_path = temp_dir / "schema.json"
        workspace = temp_dir / "workspace"
        pdf_path.write_bytes(b"%PDF-1.4\n%demo\n")
        schema_path.write_text(
            json.dumps({"fields": [{"name": "contract_amount", "description": "Contract total amount"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        source_sha256 = compute_file_sha256(pdf_path)
        doc_id = build_doc_id(source_sha256)
        client = FakeClient(doc_id, source_sha256)

        run_contract_extraction_whitebox_demo(
            pdf_path,
            schema_path,
            workspace,
            verbose=True,
            client=client,
            agent_executor=fake_agent_executor,
        )

        output = capsys.readouterr().out
        assert "[worker:contract_amount] tool output:" in output
        assert "[worker:contract_amount] normalized_result=" in output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_whitebox_demo_limits_worker_concurrency():
    temp_dir = make_temp_dir()
    try:
        pdf_path = temp_dir / "demo.pdf"
        schema_path = temp_dir / "schema.json"
        workspace = temp_dir / "workspace"
        pdf_path.write_bytes(b"%PDF-1.4\n%demo\n")
        schema_path.write_text(
            json.dumps(
                {
                    "fields": [
                        {"name": "contract_amount", "description": "Contract total amount"},
                        {"name": "signing_date", "description": "Contract signing date"},
                        {"name": "effective_date", "description": "Contract effective date"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        source_sha256 = compute_file_sha256(pdf_path)
        doc_id = build_doc_id(source_sha256)
        client = FakeClient(doc_id, source_sha256)
        state = {"active": 0, "peak": 0}

        async def tracking_agent_executor(name, instructions, prompt, tools, verbose=False, printer=print):
            forwarded_instructions = instructions
            forwarded_prompt = prompt
            del verbose
            if name.startswith("worker:"):
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                try:
                    await asyncio.sleep(0.01)
                    tools[1]()
                    tools[2]("4")
                    printer(f"[{name}] tool call: get_document_structure")
                    printer(f"[{name}] tool call: get_page_content {{'pages': '4'}}")
                    return {
                        "name": name,
                        "reasoning": "checked a tight page range",
                        "text": "worker complete",
                        "final_output": json.dumps(
                            {
                                "field": name.split(":", 1)[1],
                                "candidate_pages": [4],
                                "status": "found",
                                "value": "demo",
                                "evidence": "demo evidence",
                                "pages": [4],
                                "confidence": "High",
                                "reason": None,
                            },
                            ensure_ascii=False,
                        ),
                    }
                finally:
                    state["active"] -= 1

            return await fake_agent_executor(
                name,
                forwarded_instructions,
                forwarded_prompt,
                tools,
                printer=printer,
            )

        run_contract_extraction_whitebox_demo(
            pdf_path,
            schema_path,
            workspace,
            client=client,
            agent_executor=tracking_agent_executor,
            max_worker_concurrency=2,
        )

        assert state["peak"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
