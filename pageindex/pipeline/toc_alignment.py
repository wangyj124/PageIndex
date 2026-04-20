from .tree_builder import (
    add_page_number_to_toc,
    add_page_offset_to_toc_json,
    calculate_page_offset,
    extract_matching_page_pairs,
    page_list_to_group_text,
    remove_first_physical_index_section,
    remove_page_number,
    toc_index_extractor,
)

__all__ = [
    "remove_page_number",
    "extract_matching_page_pairs",
    "calculate_page_offset",
    "add_page_offset_to_toc_json",
    "page_list_to_group_text",
    "add_page_number_to_toc",
    "remove_first_physical_index_section",
    "toc_index_extractor",
]
