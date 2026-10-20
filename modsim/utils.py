"""Utility functions for modsim."""

from __future__ import annotations

from pathlib import Path


def validate_file_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate and convert a file path.

    Args:
        path: File path string or Path object
        must_exist: If True, raise error if file doesn't exist

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If path is invalid
    """
    try:
        p = Path(path)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid path '{path}': {e}")

    if must_exist and not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return p


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1.5s", "2m 30s", "1h 5m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_number(n: int | float, precision: int = 0) -> str:
    """Format number with thousands separators.

    Args:
        n: Number to format
        precision: Decimal places (0 for integers)

    Returns:
        Formatted string like "1,234" or "1,234.56"
    """
    if precision == 0:
        return f"{int(n):,}"
    return f"{n:,.{precision}f}"
