"""
Structured logging utilities for the LLM Evals Lab.

Provides a thin wrapper around Python's standard logging that emits
structured JSON events for pipeline steps. Useful for integration with
log aggregation systems (Elasticsearch, Splunk, Datadog, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    JSON-based log formatter that emits structured log records.

    Each log record becomes a JSON object with standard fields:
    timestamp, level, logger, message, plus any extra keyword arguments
    passed via the 'extra' dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include any extra fields passed as record attributes
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName",
            ):
                try:
                    json.dumps(val)
                    log_obj[key] = val
                except (TypeError, ValueError):
                    log_obj[key] = str(val)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def get_structured_logger(
    name: str, level: int = logging.INFO
) -> logging.Logger:
    """Return a logger that emits structured JSON to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_run_event(
    logger: logging.Logger,
    event: str,
    run_id: str,
    **kwargs: Any,
) -> None:
    """
    Emit a structured log event for a pipeline run step.

    Parameters
    ----------
    event : str
        Event name (e.g., "retrieval_complete", "answer_generated").
    run_id : str
        Current run ID.
    **kwargs : Any
        Additional key-value pairs to include in the log record.
    """
    logger.info(
        event,
        extra={"run_id": run_id, "event": event, **kwargs},
    )
