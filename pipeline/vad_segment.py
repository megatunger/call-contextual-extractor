from __future__ import annotations

from pathlib import Path

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm.auto import tqdm

_worker_model = None

def _init_worker():
    global _worker_model
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _worker_model = load_silero_vad()

def _process_file_wrapper(path, cfg):
    global _worker_model
    return segment_one_call(path, _worker_model, cfg)

from pipeline.checkpoints import (
    ensure_call_dir,
    segments_manifest_path,
    vad_done_path,
    write_jsonl_rows,
)
from pipeline.config import PipelineConfig
from pipeline.progress import print_progress


def list_raw_audio_files(cfg: PipelineConfig) -> list[Path]:
    files = sorted(cfg.raw_audio_dir.glob("*.mp3")) + sorted(cfg.raw_audio_dir.glob("*.wav"))
    files = [p for p in files if p.is_file()]
    if cfg.max_calls is not None:
        files = files[: cfg.max_calls]
    return files


def pick_channel(y: np.ndarray, channel_index: int) -> np.ndarray:
    if y.ndim == 1:
        return y
    channel_index = min(channel_index, y.shape[0] - 1)
    return y[channel_index]


def segment_speaker(
    y_full: np.ndarray,
    speaker: str,
    channel_index: int,
    model,
    sr: int,
    file_stem: str,
    cfg: PipelineConfig,
) -> list[dict]:
    wav_np = pick_channel(y_full, channel_index)
    wav = torch.from_numpy(wav_np).float()
    stamps = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)

    segments: list[dict] = []
    for i, stamp in enumerate(stamps):
        start_s = float(stamp["start"])
        end_s = float(stamp["end"])
        start_sample = int(max(0, np.floor(start_s * sr)))
        end_sample = int(min(len(wav_np), np.ceil(end_s * sr)))
        if end_sample <= start_sample:
            continue

        segment_wav = wav_np[start_sample:end_sample]
        segment_id = f"{file_stem}_{speaker.lower()}_{i:04d}"
        segment_path = (
            cfg.segment_output_dir / file_stem / speaker.lower() / f"{segment_id}.wav"
        )
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(segment_path), segment_wav, sr)

        segments.append(
            {
                "segment_id": segment_id,
                "call_id": file_stem,
                "speaker": speaker,
                "start": start_s,
                "end": end_s,
                "duration": end_s - start_s,
                "path": str(segment_path),
            }
        )
    return segments


def segment_one_call(path: Path, model, cfg: PipelineConfig) -> list[dict]:
    y, sr = librosa.load(str(path), sr=cfg.sampling_rate, mono=False)
    file_stem = path.stem
    all_segments: list[dict] = []

    for speaker, channel_index in cfg.speaker_channels.items():
        all_segments.extend(
            segment_speaker(y, speaker, channel_index, model, sr, file_stem, cfg)
        )

    ensure_call_dir(cfg.segment_output_dir, file_stem)
    manifest_path = segments_manifest_path(cfg.segment_output_dir, file_stem)
    write_jsonl_rows(manifest_path, all_segments)
    vad_done_path(cfg.segment_output_dir, file_stem).write_text("ok\n", encoding="utf-8")
    return all_segments


def is_vad_done(file_stem: str, cfg: PipelineConfig) -> bool:
    return vad_done_path(cfg.segment_output_dir, file_stem).is_file()


def run_vad_stage(cfg: PipelineConfig, model=None) -> dict:
    raw_files = list_raw_audio_files(cfg)
    if not raw_files:
        return {"total_calls": 0, "processed_calls": 0, "skipped_calls": 0, "total_segments": 0}

    torch.set_num_threads(1)
    if model is None:
        model = load_silero_vad()

    pending = [
        p
        for p in raw_files
        if not (cfg.skip_vad_if_done and is_vad_done(p.stem, cfg))
    ]
    skipped = len(raw_files) - len(pending)
    print_progress("VAD calls already done", skipped, len(raw_files))

    total_segments = 0
    max_workers = min(16, (os.cpu_count() or 4))
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_file_wrapper, path, cfg): path for path in pending}
        
        for future in tqdm(as_completed(futures), total=len(pending), desc="Stage 1 · VAD segment calls", unit="call"):
            try:
                segments = future.result()
                total_segments += len(segments)
            except Exception as e:
                print(f"Error processing {futures[future].name}: {e}")

    print_progress("VAD calls completed", len(raw_files), len(raw_files))
    return {
        "total_calls": len(raw_files),
        "processed_calls": len(pending),
        "skipped_calls": skipped,
        "total_segments": total_segments,
    }
