from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict

from pythonjsonlogger.json import JsonFormatter


def configure_logging(log_file: Path, level: str = "INFO") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
                "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "json",
                "filename": str(log_file),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "json",
            },
        },
        "root": {"level": level, "handlers": ["file", "console"]},
    }

    logging.config.dictConfig(logging_config)
