"""
Whitebox multi-agent contract extraction demo.

Run inside the PageIndex conda environment:

    conda activate PageIndex
    pip install openai-agents
    python examples/contract_extraction_whitebox_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex import run_contract_extraction_whitebox_demo


_EXAMPLES_DIR = Path(__file__).parent
_ROOT_DIR = _EXAMPLES_DIR.parent
PDF_PATH = next((_ROOT_DIR / "pdf").glob("*.pdf"))
SCHEMA_PATH = _ROOT_DIR / "sample_data" / "schemas" / "contract_fields_xt_full.json"
WORKSPACE = _ROOT_DIR / "artifacts" / "whitebox_contract_workspace"


def main():
    run_contract_extraction_whitebox_demo(
        PDF_PATH,
        SCHEMA_PATH,
        WORKSPACE,
        verbose=False,
    )


if __name__ == "__main__":
    main()
