import importlib
import subprocess
import sys

import pageindex
legacy_page_index = importlib.import_module("pageindex.page_index")
legacy_markdown = importlib.import_module("pageindex.page_index_md")
legacy_utils = importlib.import_module("pageindex.utils")


def test_legacy_exports_still_available():
    assert callable(pageindex.page_index)
    assert callable(legacy_markdown.md_to_tree)
    assert callable(pageindex.md_to_tree_hybrid)
    assert callable(legacy_utils.print_tree)
    assert callable(legacy_page_index.page_index_main)


def test_cli_help_compatibility():
    result = subprocess.run([sys.executable, "run_pageindex.py", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--pdf_path" in result.stdout


def test_module_cli_help():
    result = subprocess.run([sys.executable, "-m", "pageindex.cli", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--output-dir" in result.stdout
