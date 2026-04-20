from pageindex.config import ConfigLoader


def test_config_loader_merges_defaults():
    opt = ConfigLoader().load({"model": "openai/test-model"})
    assert opt.model == "openai/test-model"
    assert hasattr(opt, "toc_check_page_num")


def test_config_loader_rejects_unknown_keys():
    loader = ConfigLoader()
    try:
        loader.load({"unknown_key": True})
    except ValueError as exc:
        assert "Unknown config keys" in str(exc)
    else:
        raise AssertionError("ConfigLoader should reject unknown keys")
