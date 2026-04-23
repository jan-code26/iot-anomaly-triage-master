"""
Structured JSON logging for the IoT Anomaly Triage backend.

Every log record is emitted as a single JSON line — compatible with Render's
log aggregation, Datadog, and any log sink that expects NDJSON.

Usage:
    from backend.logging_config import get_logger
    log = get_logger(__name__)
    log.info("ingest", engine_id=1, cycle=42, combined_score=0.72)

Call setup_logging() once at application startup (in main.py lifespan or at
module load). After that, all loggers in the backend emit JSON lines.
"""
from __future__ import annotations

import json
import logging
import sys
import time


class _JsonFormatter(logging.Formatter):
    """Formats every LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Fields passed via extra={} land in record.__dict__ — include them all
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                payload[key] = val

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger to emit JSON lines to stdout.
    Call once at startup — safe to call multiple times (idempotent).
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and
           isinstance(h.formatter, _JsonFormatter) for h in root.handlers):
        return  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "httpcore", "httpx", "groq", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first."""
    return logging.getLogger(name)
