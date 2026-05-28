from pageindex.reference_resolution import filter_valid_reference_pages, normalize_reference_text


def test_normalize_reference_text_strips_separators():
    assert normalize_reference_text("附件 甲-第1页") == "附件甲第1页"


def test_filter_valid_reference_pages_rejects_invented_pages_and_deduplicates():
    assert filter_valid_reference_pages([4, 99, 4, 5], {1, 4, 5}, limit=3) == [4, 5]
