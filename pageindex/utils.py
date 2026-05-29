from __future__ import annotations

import importlib
import os
import platform
import uuid
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import *
from .llm import *
from .logging_utils import *
from .pdf import *
from .tree_utils import *


WORD_FILE_SUFFIXES = {".doc", ".docx"}
WORD_TO_PDF_FORMAT = 17
DEFAULT_WORD_TO_PDF_CONVERT_URL = "http://10.8.2.63:8000/convert"
WORD_TO_PDF_CONVERT_URL_ENV = "PAGEINDEX_WORD_TO_PDF_CONVERT_URL"
WORD_TO_PDF_CONVERT_TIMEOUT_ENV = "PAGEINDEX_WORD_TO_PDF_CONVERT_TIMEOUT"
DEFAULT_WORD_TO_PDF_CONVERT_TIMEOUT = 120


def _convert_word_to_pdf_windows(word_file: Path, output_pdf_path: Path) -> Path:
    try:
        win32_client = importlib.import_module("win32com.client")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前为 Windows 环境，但未安装 pywin32，无法调用 Microsoft Word 转换 DOC/DOCX。"
        ) from exc

    word_app = None
    document = None

    try:
        word_app = win32_client.DispatchEx("Word.Application")
        word_app.Visible = False
        if hasattr(word_app, "DisplayAlerts"):
            word_app.DisplayAlerts = 0

        document = word_app.Documents.Open(str(word_file))
        if output_pdf_path.exists():
            output_pdf_path.unlink()
        document.SaveAs(str(output_pdf_path), FileFormat=WORD_TO_PDF_FORMAT)
    except Exception as exc:
        raise RuntimeError(f"使用 Microsoft Word 转换 PDF 失败: {exc}") from exc
    finally:
        if document is not None:
            try:
                document.Close(False)
            except Exception:
                pass
        if word_app is not None:
            try:
                word_app.Quit()
            except Exception:
                pass

    if not output_pdf_path.is_file():
        raise RuntimeError(f"Microsoft Word 未生成预期的 PDF 文件: {output_pdf_path}")

    return output_pdf_path.resolve()


def _convert_word_to_pdf_linux(word_file: Path, output_dir_path: Path, output_pdf_path: Path) -> Path:
    endpoint = os.getenv(WORD_TO_PDF_CONVERT_URL_ENV, DEFAULT_WORD_TO_PDF_CONVERT_URL).strip()
    if not endpoint:
        raise RuntimeError(f"Linux Word 转 PDF 远程服务地址为空，请配置 {WORD_TO_PDF_CONVERT_URL_ENV}")

    try:
        timeout = float(os.getenv(WORD_TO_PDF_CONVERT_TIMEOUT_ENV, str(DEFAULT_WORD_TO_PDF_CONVERT_TIMEOUT)))
    except ValueError as exc:
        raise RuntimeError(f"{WORD_TO_PDF_CONVERT_TIMEOUT_ENV} 必须是数字秒数") from exc

    boundary = f"----PageIndexWordToPdf{uuid.uuid4().hex}"
    file_bytes = word_file.read_bytes()
    content_type = (
        "application/msword"
        if word_file.suffix.lower() == ".doc"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{word_file.name}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    request = urlrequest.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )

    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            response_body = response.read()
    except urlerror.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore").strip()
        message = f"远程 Word 转 PDF 服务返回 HTTP {exc.code}"
        if details:
            message = f"{message}: {details}"
        raise RuntimeError(message) from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"无法连接远程 Word 转 PDF 服务 {endpoint}: {exc.reason}") from exc

    if status < 200 or status >= 300:
        raise RuntimeError(f"远程 Word 转 PDF 服务返回异常状态码: {status}")

    if not response_body.startswith(b"%PDF"):
        preview = response_body[:200].decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"远程 Word 转 PDF 服务未返回 PDF 内容: {preview}")

    if output_pdf_path.exists():
        output_pdf_path.unlink()
    output_pdf_path.write_bytes(response_body)
    if not output_pdf_path.is_file():
        raise RuntimeError(f"远程 Word 转 PDF 服务已返回内容，但未写入预期 PDF 文件: {output_pdf_path}")

    return output_pdf_path.resolve()


def convert_word_to_pdf(word_path: str, output_dir: str) -> str:
    """将 DOC/DOCX 文件转换为 PDF，并返回生成后的 PDF 绝对路径。"""
    word_file = Path(word_path).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger = JsonLogger(str(word_file), base_dir=str(output_dir_path / "logs"))
    system_name = platform.system()
    file_suffix = word_file.suffix.lower()

    if not word_file.is_file():
        raise FileNotFoundError(f"Word 文件不存在: {word_file}")
    if file_suffix not in WORD_FILE_SUFFIXES:
        raise ValueError(f"仅支持转换 .doc 或 .docx 文件，当前文件为: {word_file.name}")

    output_pdf_path = (output_dir_path / f"{word_file.stem}.pdf").resolve()

    logger.info(
        {
            "event": "word_to_pdf_started",
            "source_file": str(word_file),
            "target_file": str(output_pdf_path),
            "platform": system_name,
        }
    )

    try:
        if system_name == "Windows":
            converted_path = _convert_word_to_pdf_windows(word_file, output_pdf_path)
            backend = "microsoft_word"
        elif system_name == "Linux":
            converted_path = _convert_word_to_pdf_linux(word_file, output_dir_path, output_pdf_path)
            backend = "remote_word_api"
        else:
            raise RuntimeError(
                f"当前操作系统 {system_name} 暂不支持 Word 转 PDF，仅支持 Windows 和 Linux。"
            )
    except Exception as exc:
        logger.exception(
            {
                "event": "word_to_pdf_failed",
                "source_file": str(word_file),
                "target_file": str(output_pdf_path),
                "platform": system_name,
                "error": str(exc),
            }
        )
        raise

    logger.info(
        {
            "event": "word_to_pdf_completed",
            "source_file": str(word_file),
            "target_file": str(converted_path),
            "platform": system_name,
            "backend": backend,
        }
    )
    return str(converted_path)
