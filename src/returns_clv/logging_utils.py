"""Console logging and small report-writing helpers shared by every stage."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install a single stdout handler, idempotently."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
    # Chatty third parties.
    for noisy in ("matplotlib", "PIL", "numexpr", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def banner(logger: logging.Logger, title: str) -> None:
    """Log a visually separated stage heading."""
    logger.info("=" * 74)
    logger.info(title.upper())
    logger.info("=" * 74)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _json_default(obj: Any) -> Any:
    """Make numpy scalars, paths and datetimes JSON-serialisable."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
