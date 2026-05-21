from __future__ import annotations

from pathlib import Path

import requests
from tqdm.auto import tqdm

from pipeline.checkpoints import (
    append_jsonl_row,
    asr_results_path,
    iter_call_dirs,
    load_jsonl_list,
    load_jsonl_rows,
    segments_manifest_path,
)
from pipeline.config import PipelineConfig
from pipeline.progress import print_progress


def recognize_segment(segment_path: str, cfg: PipelineConfig) -> dict:
    if not cfg.speech_api_key:
        return {
            "success": False,
            "transcription": "",
            "word_levels": [],
            "error": "Missing SPEECH_API_KEY (.env or environment)",
        }

    headers = {"x-api-key": cfg.speech_api_key}
    data = {"word_level": "1"}

    try:
        with open(segment_path, "rb") as handle:
            files = {"audio_file": (Path(segment_path).name, handle, "audio/wav")}
            response = requests.post(
                cfg.recognize_url,
                headers=headers,
                data=data,
                files=files,
                timeout=cfg.request_timeout_sec,
            )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "transcription": "",
            "word_levels": [],
            "error": str(exc),
        }

    return {
        "success": bool(payload.get("success", False)),
        "transcription": (payload.get("transcription") or "").strip(),
        "word_levels": payload.get("word_levels") or [],
        "error": None,
    }


def list_calls_with_segments(cfg: PipelineConfig) -> list[str]:
    if not cfg.segment_output_dir.is_dir():
        return []
    call_ids: list[str] = []
    for call_dir in iter_call_dirs(cfg.segment_output_dir):
        manifest = call_dir / "segments_manifest.jsonl"
        has_wavs = any(call_dir.rglob("*.wav"))
        if manifest.is_file() or has_wavs:
            call_ids.append(call_dir.name)
    return call_ids


def load_segments_for_call(call_id: str, cfg: PipelineConfig) -> list[dict]:
    manifest_path = segments_manifest_path(cfg.segment_output_dir, call_id)
    if manifest_path.is_file():
        return load_jsonl_list(manifest_path)

    # Backward compatibility: older runs may only have WAV files on disk.
    call_dir = cfg.segment_output_dir / call_id
    rows: list[dict] = []
    for wav_path in sorted(call_dir.rglob("*.wav")):
        speaker_dir = wav_path.parent.name.lower()
        speaker = "Agent" if speaker_dir == "agent" else "Customer" if speaker_dir == "customer" else speaker_dir.title()
        rows.append(
            {
                "segment_id": wav_path.stem,
                "call_id": call_id,
                "speaker": speaker,
                "start": 0.0,
                "end": 0.0,
                "duration": 0.0,
                "path": str(wav_path),
            }
        )
    return rows


def iter_segments_for_asr(cfg: PipelineConfig) -> list[dict]:
    pending: list[dict] = []
    for call_id in list_calls_with_segments(cfg):
        manifest_rows = load_segments_for_call(call_id, cfg)
        cached = (
            load_jsonl_rows(asr_results_path(cfg.segment_output_dir, call_id))
            if cfg.resume_from_checkpoint
            else {}
        )
        for row in manifest_rows:
            seg_id = row["segment_id"]
            cached_row = cached.get(seg_id)
            if cached_row is not None:
                if cached_row.get("success") or (not cfg.retry_failed_segments):
                    continue
            pending.append(row)
    return pending


def run_asr_stage(cfg: PipelineConfig) -> dict:
    pending = iter_segments_for_asr(cfg)
    total_manifest = sum(
        len(load_segments_for_call(call_id, cfg))
        for call_id in list_calls_with_segments(cfg)
    )
    done_already = total_manifest - len(pending)
    print_progress("ASR segments already done", done_already, total_manifest)

    if not pending:
        return {
            "total_segments": total_manifest,
            "processed_segments": 0,
            "skipped_segments": done_already,
        }

    success_count = 0
    for row in tqdm(pending, desc="Stage 2 · ASR recognize segments", unit="seg"):
        asr = recognize_segment(row["path"], cfg)
        out = {**row, **asr}
        call_id = row["call_id"]
        append_jsonl_row(asr_results_path(cfg.segment_output_dir, call_id), out)
        if out.get("success"):
            success_count += 1

    print_progress("ASR segments completed", total_manifest, total_manifest)
    return {
        "total_segments": total_manifest,
        "processed_segments": len(pending),
        "skipped_segments": done_already,
        "successful_in_run": success_count,
    }
