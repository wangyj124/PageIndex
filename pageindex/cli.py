import argparse
import asyncio
import json
from pathlib import Path

from .config import ConfigLoader
from .hybrid_index import (
    finalize_hybrid_payload as _finalize_hybrid_payload_impl,
    prepare_hybrid_sources_from_pdf as _prepare_hybrid_sources_from_pdf_impl,
    rename_hybrid_intervals_to_pages as _rename_hybrid_intervals_to_pages_impl,
    run_hybrid_pipeline_for_sources as _run_hybrid_pipeline_for_sources_impl,
)
from .logging_utils import JsonLogger
from .markdown import (
    md_to_tree,
    resolve_hybrid_json_path,
)
from .page_index import page_index_main


def build_parser():
    parser = argparse.ArgumentParser(description="Process PDF or Markdown document and generate structure")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--md_path", type=str, help="Path to the Markdown file")
    parser.add_argument("--json_path", type=str, help="Optional path to the JSON produced alongside the Markdown file")
    parser.add_argument("--model", type=str, default=None, help="Model to use (overrides config.yaml)")
    parser.add_argument("--toc-check-pages", type=int, default=None, help="Number of pages to check for table of contents (PDF only)")
    parser.add_argument("--max-pages-per-node", type=int, default=None, help="Maximum number of pages per node (PDF only)")
    parser.add_argument("--max-tokens-per-node", type=int, default=None, help="Maximum number of tokens per node (PDF only)")
    parser.add_argument("--if-add-node-id", type=str, default=None, help="Whether to add node id to the node")
    parser.add_argument("--if-add-node-summary", type=str, default=None, help="Whether to add summary to the node")
    parser.add_argument("--if-add-doc-description", type=str, default=None, help="Whether to add doc description to the doc")
    parser.add_argument("--if-add-node-text", type=str, default=None, help="Whether to add text to the node")
    parser.add_argument("--if-thinning", type=str, default="no", help="Whether to apply tree thinning for markdown (markdown only)")
    parser.add_argument("--thinning-threshold", type=int, default=5000, help="Minimum token threshold for thinning (markdown only)")
    parser.add_argument("--summary-token-threshold", type=int, default=200, help="Token threshold for generating summaries (markdown only)")
    parser.add_argument("--md-hybrid", action="store_true", help="Enable hybrid Markdown+JSON tree construction")
    parser.add_argument("--output-dir", type=str, default="artifacts/results", help="Directory to write generated JSON output")
    return parser


def _validate_input(args):
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    if args.json_path and not args.md_path:
        raise ValueError("--json_path can only be used together with --md_path")
    if args.json_path and not args.md_hybrid:
        raise ValueError("--json_path requires --md-hybrid")


def _write_output(output_dir, source_path, payload):
    source_name = Path(source_path).stem
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"{source_name}_structure.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Tree structure saved to: {output_file}")
    return output_file


def _prepare_hybrid_sources_from_pdf(pdf_path, output_dir):
    return _prepare_hybrid_sources_from_pdf_impl(pdf_path, output_dir)


def _rename_hybrid_intervals_to_pages(data):
    return _rename_hybrid_intervals_to_pages_impl(data)


def _finalize_hybrid_payload(tree_result, source_path, line_count, opt, summary_token_threshold):
    return _finalize_hybrid_payload_impl(tree_result, source_path, line_count, opt, summary_token_threshold)


def _run_hybrid_pipeline_for_sources(source_path, md_path, json_path, opt, summary_token_threshold):
    payload, _ = _run_hybrid_pipeline_for_sources_impl(
        source_path,
        md_path,
        json_path,
        opt,
        summary_token_threshold,
        progress_logger=JsonLogger(str(source_path)),
    )
    return payload


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_input(args)

    user_opt = {
        "model": args.model,
        "if_add_node_summary": args.if_add_node_summary,
        "if_add_doc_description": args.if_add_doc_description,
        "if_add_node_text": args.if_add_node_text,
        "if_add_node_id": args.if_add_node_id,
    }

    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError("PDF file must have .pdf extension")
        if not pdf_path.is_file():
            raise ValueError(f"PDF file not found: {args.pdf_path}")

        if args.md_hybrid:
            opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})
            md_path, json_path = _prepare_hybrid_sources_from_pdf(pdf_path, args.output_dir)
            result = _run_hybrid_pipeline_for_sources(
                source_path=pdf_path,
                md_path=md_path,
                json_path=json_path,
                opt=opt,
                summary_token_threshold=args.summary_token_threshold,
            )
        else:
            pdf_opt = ConfigLoader().load(
                {
                    **{k: v for k, v in user_opt.items() if v is not None},
                    **{
                        key: value
                        for key, value in {
                            "toc_check_page_num": args.toc_check_pages,
                            "max_page_num_each_node": args.max_pages_per_node,
                            "max_token_num_each_node": args.max_tokens_per_node,
                        }.items()
                        if value is not None
                    },
                }
            )
            result = page_index_main(str(pdf_path), pdf_opt)

        print("Parsing done, saving to file...")
        _write_output(args.output_dir, pdf_path, result)
        return

    md_path = Path(args.md_path)
    if md_path.suffix.lower() not in (".md", ".markdown"):
        raise ValueError("Markdown file must have .md or .markdown extension")
    if not md_path.is_file():
        raise ValueError(f"Markdown file not found: {args.md_path}")

    opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})
    print("Processing markdown file...")
    if args.md_hybrid:
        resolved_json_path = Path(resolve_hybrid_json_path(str(md_path), json_path=args.json_path))
        result = _run_hybrid_pipeline_for_sources(
            source_path=md_path,
            md_path=md_path,
            json_path=resolved_json_path,
            opt=opt,
            summary_token_threshold=args.summary_token_threshold,
        )
    else:
        result = asyncio.run(
            md_to_tree(
                md_path=str(md_path),
                if_thinning=args.if_thinning.lower() == "yes",
                min_token_threshold=args.thinning_threshold,
                if_add_node_summary=opt.if_add_node_summary,
                summary_token_threshold=args.summary_token_threshold,
                model=opt.model,
                if_add_doc_description=opt.if_add_doc_description,
                if_add_node_text=opt.if_add_node_text,
                if_add_node_id=opt.if_add_node_id,
            )
        )
    print("Parsing done, saving to file...")
    _write_output(args.output_dir, md_path, result)


if __name__ == "__main__":
    main()
