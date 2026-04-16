"""
sms_spam/logs/logger.py — Centralised Logging
===============================================
Provides a single ``get_logger()`` factory used across the entire project.

Features
--------
* Writes to BOTH the console (coloured) and a rotating log file under ``logs/``.
* Console handler uses colour-coded levels (DEBUG=cyan, INFO=green,
  WARNING=yellow, ERROR=red, CRITICAL=magenta).
* File handler stores plain-text entries with full timestamps.
* Log files rotate at 5 MB; up to 3 backup files are kept.

Usage
-----
    from sms_spam.logs.logger import get_logger

    log = get_logger(__name__)          # module-level logger
    log.info("Pipeline started")
    log.warning("Low memory")
    log.error("File not found: %s", path)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # d:/SMS-Spam-Detection
_LOG_DIR      = _PROJECT_ROOT / "logs"
_LOG_FILE     = _LOG_DIR / "sms_spam.log"

# ---------------------------------------------------------------------------
# ANSI colour codes for the console handler
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_COLOURS = {
    "DEBUG"    : "\033[36m",   # cyan
    "INFO"     : "\033[32m",   # green
    "WARNING"  : "\033[33m",   # yellow
    "ERROR"    : "\033[31m",   # red
    "CRITICAL" : "\033[35m",   # magenta
}


class _ColouredFormatter(logging.Formatter):
    """Console formatter that prepends a colour code to the level name."""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    _DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, _RESET)
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        formatter = logging.Formatter(self._FMT, datefmt=self._DATE)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain-text formatter for the rotating file handler."""

    _FMT  = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


# ---------------------------------------------------------------------------
# Internal registry — keep one logger per name (avoid duplicate handlers)
# ---------------------------------------------------------------------------
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "sms_spam",
    *,
    level: int = logging.DEBUG,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Return (or create) a named logger with console + file handlers.

    Parameters
    ----------
    name : str
        Logger name — use ``__name__`` for module-level loggers.
    level : int
        Minimum log level (default: ``logging.DEBUG``).
    log_to_file : bool
        When *True* (default) messages are written to ``logs/sms_spam.log``.
    log_to_console : bool
        When *True* (default) messages are printed to stdout.

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # prevent double-logging if root logger is set

    # ── Console handler ───────────────────────────────────────────────────────
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_ColouredFormatter())
        logger.addHandler(ch)

    # ── Rotating file handler ──────────────────────────────────────────────────
    if log_to_file:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(_PlainFormatter())
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
