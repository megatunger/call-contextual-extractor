from __future__ import annotations

import json
from pathlib import Path


def call_dir(segment_output_dir: Path, file_stem: str) -> Path:
    """Path to one call's output folder (does not create it)."""
    return segment_output_dir / file_stem


def ensure_call_dir(segment_output_dir: Path, file_stem: str) -> Path:
    path = call_dir(segment_output_dir, file_stem)
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_call_dirs(segment_output_dir: Path):
    """Yield call output directories only (skip files like .DS_Store)."""
    if not segment_output_dir.is_dir():
        return
    for path in sorted(segment_output_dir.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def vad_done_path(segment_output_dir: Path, file_stem: str) -> Path:
    return call_dir(segment_output_dir, file_stem) / "vad.done"


def segments_manifest_path(segment_output_dir: Path, file_stem: str) -> Path:
    return call_dir(segment_output_dir, file_stem) / "segments_manifest.jsonl"


def asr_results_path(segment_output_dir: Path, file_stem: str) -> Path:
    return call_dir(segment_output_dir, file_stem) / "asr_results.jsonl"


def final_dialogue_path(segment_output_dir: Path, file_stem: str) -> Path:
    return call_dir(segment_output_dir, file_stem) / "final_dialogue.json"


def load_jsonl_rows(jsonl_path: Path) -> dict[str, dict]:
    rows_by_key: dict[str, dict] = {}
    if not jsonl_path.is_file():
        return rows_by_key
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("segment_id")
        if key:
            rows_by_key[key] = row
    return rows_by_key


def load_jsonl_list(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not jsonl_path.is_file():
        return rows
    for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl_row(jsonl_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_rows(jsonl_path: Path, rows: list[dict]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
