from __future__ import annotations

import re

from pipeline.checkpoints import (
    asr_results_path,
    final_dialogue_path,
    load_jsonl_rows,
    save_json,
)
from pipeline.config import PipelineConfig
from pipeline.progress import print_progress
from pipeline.asr_recognize import list_calls_with_segments
from tqdm.auto import tqdm


def _segment_sort_key(row: dict) -> tuple:
    match = re.search(r"_(\d{4})$", row.get("segment_id", ""))
    segment_idx = int(match.group(1)) if match else 0
    return (float(row.get("start", 0.0)), row.get("speaker", ""), segment_idx)


def merge_transcript_lines(rows: list[dict]) -> list[str]:
    sorted_rows = sorted(rows, key=_segment_sort_key)
    lines: list[str] = []
    last_speaker = None

    for row in sorted_rows:
        if not row.get("success") or not row.get("transcription"):
            continue

        speaker = row["speaker"]
        text = row["transcription"]
        if lines and speaker == last_speaker:
            lines[-1] = f"{lines[-1]} {text}".strip()
        else:
            lines.append(f"{speaker}: {text}")
            last_speaker = speaker

    return lines


def merge_one_call(call_id: str, cfg: PipelineConfig) -> list[str]:
    asr_path = asr_results_path(cfg.segment_output_dir, call_id)
    rows_map = load_jsonl_rows(asr_path)
    rows = list(rows_map.values())
    dialogue = merge_transcript_lines(rows)
    save_json(final_dialogue_path(cfg.segment_output_dir, call_id), dialogue)
    return dialogue


def run_merge_stage(cfg: PipelineConfig) -> dict:
    call_ids = list_calls_with_segments(cfg)
    if not call_ids:
        return {"total_calls": 0, "merged_calls": 0}

    merged = 0
    for call_id in tqdm(call_ids, desc="Stage 3 · Merge transcripts", unit="call"):
        merge_one_call(call_id, cfg)
        merged += 1

    print_progress("Merge transcripts completed", merged, len(call_ids))
    return {"total_calls": len(call_ids), "merged_calls": merged}
