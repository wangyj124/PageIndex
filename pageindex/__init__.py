from .client import PageIndexClient
from .contract_extraction import ConfidenceLevel, ExtractionStatus, FieldSpec, extract_contract_fields, normalize_schema
from .hybrid_index import prepare_hybrid_sources_from_pdf, require_opendataloader_pdf, run_hybrid_pipeline_for_sources
from .identity import build_doc_id, build_tree_id, compute_file_sha256
from .hybrid_pipeline import add_preface_node_if_needed, build_hybrid_tree_pipeline, build_initial_flat_nodes
from .hybrid_pipeline import collapse_demoted_nodes
from .markdown import extract_hybrid_toc_with_fallback, md_to_tree, md_to_tree_hybrid
from .page_index import page_index, page_index_main
from .retrieve import get_document, get_document_structure, get_page_content
from .tree_reconstruction import (
    RECONSTRUCTION_SYSTEM_PROMPT,
    RECONSTRUCTION_USER_PROMPT_TEMPLATE,
    TreeReconstructionError,
    build_context_payload,
    reconstruct_tree_structure,
    validate_tree_logic,
)
from .tree_optimization import (
    generate_summaries,
    optimize_and_summarize_tree,
    refine_large_nodes,
    thin_small_nodes,
)
from .tree_utils import build_tree_and_intervals
from .whitebox_demo import run_contract_extraction_whitebox_demo

__all__ = [
    "PageIndexClient",
    "ConfidenceLevel",
    "ExtractionStatus",
    "FieldSpec",
    "add_preface_node_if_needed",
    "build_doc_id",
    "build_tree_id",
    "build_initial_flat_nodes",
    "build_hybrid_tree_pipeline",
    "collapse_demoted_nodes",
    "compute_file_sha256",
    "extract_contract_fields",
    "extract_hybrid_toc_with_fallback",
    "md_to_tree",
    "md_to_tree_hybrid",
    "normalize_schema",
    "page_index",
    "page_index_main",
    "prepare_hybrid_sources_from_pdf",
    "RECONSTRUCTION_SYSTEM_PROMPT",
    "RECONSTRUCTION_USER_PROMPT_TEMPLATE",
    "TreeReconstructionError",
    "build_context_payload",
    "reconstruct_tree_structure",
    "validate_tree_logic",
    "thin_small_nodes",
    "refine_large_nodes",
    "generate_summaries",
    "optimize_and_summarize_tree",
    "build_tree_and_intervals",
    "run_contract_extraction_whitebox_demo",
    "get_document",
    "get_document_structure",
    "get_page_content",
    "require_opendataloader_pdf",
    "run_hybrid_pipeline_for_sources",
]
