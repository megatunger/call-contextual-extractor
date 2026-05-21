from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class PipelineConfig:
    repo_root: Path
    raw_audio_dir: Path
    segment_output_dir: Path
    speech_api_key: str = ""
    sampling_rate: int = 16_000
    max_calls: int | None = None  # None = process all files in raw_audio
    speaker_channels: dict[str, int] | None = None
    recognize_url: str = "http://103.140.249.39:8000/recognize/"
    request_timeout_sec: int = 60
    resume_from_checkpoint: bool = True
    retry_failed_segments: bool = True
    skip_vad_if_done: bool = True

    def __post_init__(self) -> None:
        if self.speaker_channels is None:
            self.speaker_channels = {"Agent": 0, "Customer": 1}

    @classmethod
    def from_repo(cls, repo_root: Path | None = None, **overrides) -> "PipelineConfig":
        root = repo_root or Path.cwd()
        dotenv = load_dotenv(root / ".env")
        api_key = os.environ.get("SPEECH_API_KEY") or dotenv.get("SPEECH_API_KEY", "")
        cfg = cls(
            repo_root=root,
            raw_audio_dir=root / "data" / "raw_audio",
            segment_output_dir=root / "data" / "segmented_audio",
            speech_api_key=api_key,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg
