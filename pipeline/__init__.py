"""Call audio pipeline: VAD segmentation, ASR recognition, transcript merge."""

__all__ = [
    "PipelineConfig",
    "iter_call_dirs",
    "run_vad_stage",
    "run_asr_stage",
    "run_merge_stage",
]


def __getattr__(name: str):
    if name == "PipelineConfig":
        from pipeline.config import PipelineConfig

        return PipelineConfig
    if name == "iter_call_dirs":
        from pipeline.checkpoints import iter_call_dirs

        return iter_call_dirs
    if name == "run_vad_stage":
        from pipeline.vad_segment import run_vad_stage

        return run_vad_stage
    if name == "run_asr_stage":
        from pipeline.asr_recognize import run_asr_stage

        return run_asr_stage
    if name == "run_merge_stage":
        from pipeline.transcript import run_merge_stage

        return run_merge_stage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
