from pageindex.config import ConfigLoader


def test_config_loader_merges_defaults():
    opt = ConfigLoader().load({"model": "openai/test-model"})
    assert opt.model == "openai/test-model"
    assert hasattr(opt, "toc_check_page_num")
    assert opt.long_context_model == "openai/deepseek-v4-flash"
    assert opt.bm25_primary_rrf_weight == 3.0
    assert opt.embedding_model == "openai/bge-m3:latest"
    assert opt.embedding_batch_size == 64
    assert opt.embedding_request_token_budget == 8192


def test_config_loader_accepts_long_context_model_override():
    opt = ConfigLoader().load({"long_context_model": "openai/long-context-test"})

    assert opt.long_context_model == "openai/long-context-test"


def test_config_loader_rejects_unknown_keys():
    loader = ConfigLoader()
    try:
        loader.load({"unknown_key": True})
    except ValueError as exc:
        assert "Unknown config keys" in str(exc)
    else:
        raise AssertionError("ConfigLoader should reject unknown keys")
