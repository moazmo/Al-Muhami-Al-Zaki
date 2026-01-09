"""
Structured logging for Al-Muhami Al-Zaki.

Uses loguru for rich, structured logging with rotation.
"""

import sys
from typing import Optional

from loguru import logger

from src.utils.config import get_settings


def setup_logger(level: Optional[str] = None, json_format: bool = False) -> None:
    """
    Configure the application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to settings.log_level.
        json_format: If True, output logs as JSON for production.
    """
    settings = get_settings()
    log_level = level or settings.log_level

    # Remove default handler
    logger.remove()

    # Console format
    if json_format:
        log_format = "{message}"
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            serialize=True,  # JSON output
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            level=log_level,
            format=log_format,
            colorize=True,
        )

    # File logging with rotation
    logger.add(
        "logs/app.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


def get_logger(name: str):
    """
    Get a contextualized logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance bound to the module name
    """
    return logger.bind(name=name)


# Export logger instance for direct imports
__all__ = ["logger", "setup_logger", "get_logger"]
