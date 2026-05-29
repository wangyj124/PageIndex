import pytest

import pageindex.utils as utils


class DummyLogger:
    def __init__(self, file_path, base_dir="artifacts/logs"):
        self.file_path = file_path
        self.base_dir = base_dir
        self.events = []

    def info(self, message, *args, **kwargs):
        self.events.append(("info", message))

    def error(self, message, *args, **kwargs):
        self.events.append(("error", message))

    def exception(self, message, *args, **kwargs):
        self.events.append(("exception", message))


class DummyResponse:
    status = 200

    def __init__(self, payload=b"%PDF-1.4\nremote\n"):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload

    def getcode(self):
        return self.status


def test_convert_word_to_pdf_uses_remote_api_on_linux(monkeypatch, tmp_path):
    word_path = tmp_path / "contract.docx"
    word_path.write_bytes(b"word-content")
    output_dir = tmp_path / "output"
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["content_type"] = request.headers["Content-type"]
        captured["body"] = request.data
        return DummyResponse()

    monkeypatch.setattr(utils, "JsonLogger", DummyLogger)
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setenv(utils.WORD_TO_PDF_CONVERT_URL_ENV, "http://convert.example/convert")
    monkeypatch.setenv(utils.WORD_TO_PDF_CONVERT_TIMEOUT_ENV, "12.5")
    monkeypatch.setattr(utils.urlrequest, "urlopen", fake_urlopen)

    result = utils.convert_word_to_pdf(str(word_path), str(output_dir))

    assert result == str((output_dir / "contract.pdf").resolve())
    assert (output_dir / "contract.pdf").read_bytes() == b"%PDF-1.4\nremote\n"
    assert captured["url"] == "http://convert.example/convert"
    assert captured["timeout"] == 12.5
    assert captured["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="contract.docx"' in captured["body"]
    assert b"word-content" in captured["body"]


def test_convert_word_to_pdf_raises_friendly_error_when_remote_api_returns_non_pdf(monkeypatch, tmp_path):
    word_path = tmp_path / "contract.doc"
    word_path.write_bytes(b"word-content")

    monkeypatch.setattr(utils, "JsonLogger", DummyLogger)
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils.urlrequest, "urlopen", lambda request, timeout: DummyResponse(b'{"error":"failed"}'))

    with pytest.raises(RuntimeError, match="未返回 PDF"):
        utils.convert_word_to_pdf(str(word_path), str(tmp_path / "output"))


def test_convert_word_to_pdf_raises_friendly_error_when_pywin32_missing(monkeypatch, tmp_path):
    word_path = tmp_path / "contract.docx"
    word_path.write_bytes(b"word-content")

    def fake_import_module(module_name):
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(utils, "JsonLogger", DummyLogger)
    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(utils.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="pywin32"):
        utils.convert_word_to_pdf(str(word_path), str(tmp_path / "output"))
