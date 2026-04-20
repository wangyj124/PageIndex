"""
Contract key information extraction demo.

Run inside the PageIndex conda environment:

    conda activate PageIndex
    python examples/contract_extraction_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex import PageIndexClient, extract_contract_fields
from pageindex.logging_utils import JsonLogger


_EXAMPLES_DIR = Path(__file__).parent
_ROOT_DIR = _EXAMPLES_DIR.parent
PDF_PATH = next((_ROOT_DIR / "pdf").glob("*.pdf"))
SCHEMA_PATH = _ROOT_DIR / "sample_data" / "schemas" / "contract_fields_basic.json"
WORKSPACE = _ROOT_DIR / "artifacts" / "contract_workspace"


def main():
    client = PageIndexClient(workspace=WORKSPACE)
    progress_logger = JsonLogger(str(PDF_PATH))
    doc_id = client.index(
        str(PDF_PATH),
        strategy="hybrid",
        progress_logger=progress_logger,
    )
    print(f"Using doc_id={doc_id}, tree_id={client.get_tree_id(doc_id)}")

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    result = extract_contract_fields(client, doc_id, schema, max_concurrency=4)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
