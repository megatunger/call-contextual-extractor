from __future__ import annotations


def format_progress(label: str, done: int, total: int) -> str:
    pct = (100.0 * done / total) if total else 0.0
    return f"{label}: {done}/{total} ({pct:.1f}%)"


def print_progress(label: str, done: int, total: int) -> None:
    print(format_progress(label, done, total))
