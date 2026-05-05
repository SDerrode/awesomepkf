"""Rich-based pretty display helpers for dicts and dataclasses."""

from __future__ import annotations

import math
import sys as _sys
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

__all__ = ["console", "rich_show_fields"]

# FIX: force_terminal=True removed — avoids spurious ANSI sequences in file/pipe logs
console = Console(color_system="truecolor" if _sys.stdout.isatty() else None)


def rich_show_fields(
    d: dict | Any,
    fields: list[str] | None = None,
    title: str = "Data selection",
    decimals: int = 4,
    max_items: int = 10,
) -> None:
    """
    Display a dictionary or dataclass in a readable Rich table.

    Floats are rounded to ``decimals`` digits. NumPy booleans are cast to
    Python bools. Arrays longer than ``max_items`` are truncated. Nested
    dicts and lists are supported.

    Parameters
    ----------
    d : dict or dataclass
        Data to display. Dataclasses are converted via ``asdict``.
    fields : list of str, optional
        Subset of keys to display. If ``None``, all keys are shown.
    title : str, optional
        Title of the Rich table (default ``"Data selection"``).
    decimals : int, optional
        Number of decimal places for float formatting (default ``4``).
    max_items : int, optional
        Maximum number of items shown for arrays and lists (default ``10``).
    """
    if is_dataclass(d):
        d = asdict(d)

    if fields is None:
        fields = list(d.keys())

    table = Table(title=title)
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", justify="left")

    def format_value(obj) -> str:
        """Recursive formatter for scientific display."""
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return str(bool(obj))
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return str(obj)
            return f"{obj:.{decimals}f}"
        if isinstance(obj, np.ndarray):
            return format_value(obj.tolist())
        if isinstance(obj, (list, tuple)):
            if len(obj) > max_items:
                displayed = [format_value(v) for v in obj[:max_items]]
                return "[" + ", ".join(displayed) + ", ...]"
            return "[" + ", ".join(format_value(v) for v in obj) + "]"
        if isinstance(obj, dict):
            items = [f"{k}: {format_value(v)}" for k, v in obj.items()]
            return "{ " + ", ".join(items) + " }"
        return str(obj)

    for key in fields:
        if key in d:
            table.add_row(
                Text(key, style="cyan"),
                Text(format_value(d[key]), style="magenta"),
            )

    console.print(table)
