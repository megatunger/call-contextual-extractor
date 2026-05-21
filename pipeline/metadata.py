from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl
import soundfile as sf


def ffprobe_audio(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_sample",
        "-show_entries",
        "format=duration",
        "-select_streams",
        "a:0",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    streams = data.get("streams") or [{}]
    st0 = streams[0]
    fmt = data.get("format") or {}
    bps = st0.get("bits_per_sample")
    if bps == 0:
        bps = None
    dur = fmt.get("duration")
    return {
        "codec_name": st0.get("codec_name"),
        "sample_rate_hz_probe": int(st0["sample_rate"]) if st0.get("sample_rate") else None,
        "channels_probe": int(st0["channels"]) if st0.get("channels") is not None else None,
        "bits_per_sample": bps,
        "duration_sec_probe": float(dur) if dur is not None else None,
    }


def soundfile_wav_row(path: Path) -> dict | None:
    if path.suffix.lower() != ".wav":
        return None
    info = sf.info(str(path))
    return {
        "wav_subtype": info.subtype,
        "sample_rate_hz_sf": info.samplerate,
        "channels_sf": info.channels,
        "duration_sec_sf": info.duration,
    }


def collect_metadata_rows(paths: list[Path]) -> pl.DataFrame:
    rows: list[dict] = []
    for p in paths:
        row: dict = {"file": p.name}
        try:
            row.update(ffprobe_audio(p))
        except Exception as exc:  # noqa: BLE001
            row["probe_error"] = repr(exc)
        sf_row = soundfile_wav_row(p)
        if sf_row:
            row.update(sf_row)
        rows.append(row)
    return pl.DataFrame(rows)
