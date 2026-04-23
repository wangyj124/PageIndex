import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .pdf import get_pdf_name


PROGRESS_STAGE_PERCENTS = {
    "starting": 0,
    "preparing_sources": 10,
    "converting_pdf": 25,
    "loading_hybrid_sources": 40,
    "aligning_headings": 55,
    "reconstructing_tree": 70,
    "generating_summaries": 85,
    "caching_pages": 95,
    "saving_workspace": 95,
    "completed": 100,
    "failed": 100,
}


class JsonLogger:
    def __init__(self, file_path, base_dir="artifacts/logs"):
        pdf_name = get_pdf_name(file_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_data = []

    def _coerce_message(self, message: Any, args: tuple[Any, ...]) -> Any:
        if isinstance(message, dict) or not args:
            return message
        if isinstance(message, str):
            try:
                return message % args
            except Exception:
                return " ".join([message, *[str(arg) for arg in args]])
        return " ".join([str(message), *[str(arg) for arg in args]])

    def log(self, level, message, *args, **kwargs):
        message = self._coerce_message(message, args)
        if isinstance(message, dict):
            payload = dict(message)
        else:
            payload = {"message": message}
        payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        if level:
            payload.setdefault("level", level)
        if kwargs:
            payload.update(kwargs)
        self.log_data.append(payload)
        with open(self._filepath(), "w", encoding="utf-8") as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)

    def info(self, message, *args, **kwargs):
        self.log("INFO", message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.log("ERROR", message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self.log("DEBUG", message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, *args, **kwargs)

    def _filepath(self):
        return os.path.join(self.base_dir, self.filename)


def build_progress_event(stage, message, doc_name="", percent=None, extra=None):
    event = {
        "stage": stage,
        "message": message,
        "doc_name": doc_name or "",
        "percent": PROGRESS_STAGE_PERCENTS.get(stage, percent if percent is not None else 0),
        "extra": extra or {},
    }
    return event


def emit_progress_event(stage, message, doc_name="", percent=None, extra=None, progress_callback=None, progress_logger=None):
    event = build_progress_event(stage, message, doc_name=doc_name, percent=percent, extra=extra)
    if progress_logger is not None:
        progress_logger.info(event)
    if progress_callback is not None:
        progress_callback(dict(event))
    else:
        print(f"[{event['percent']:>3}%] [{stage}] {message}")
    return event


__all__ = ["JsonLogger", "PROGRESS_STAGE_PERCENTS", "build_progress_event", "emit_progress_event"]
