"""Merge Colab/CUDA LoRA adapters into a full model folder for Apple Silicon eval.

CUDA Unsloth saves PEFT adapters (adapter_model.safetensors). MLX evaluation
cannot load those directly; it needs either MLX-native adapters or a merged
full model. Run this once on Colab (or any CUDA machine) after fine-tuning:

    python pipeline/export_merged.py --lora data/models/Qwen3.5-2B_lora
    python pipeline/export_merged.py --lora data/models/Qwen3.5-0.8B_lora

Then on Mac, the evaluator uses data/models/<name>_merged automatically.
"""

import os
import argparse
from pathlib import Path

from unsloth import FastLanguageModel


def export_merged(lora_path: str, output_path: str | None = None) -> str:
    lora_path = str(Path(lora_path).resolve())
    if not Path(lora_path).is_dir():
        raise FileNotFoundError(f"LoRA folder not found: {lora_path}")

    output_path = output_path or lora_path.replace("_lora", "_merged")
    output_path = str(Path(output_path).resolve())
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print(f"Loading LoRA adapters from {lora_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained_merged(output_path, tokenizer)
    tokenizer.save_pretrained(output_path)
    print(f"Done. Use {output_path} for local MLX evaluation.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export merged full model from CUDA/PEFT LoRA for Mac MLX eval"
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=True,
        help="Path to LoRA folder (e.g. data/models/Qwen3.5-2B_lora)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder (default: <lora_path> with _lora -> _merged)",
    )
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    export_merged(args.lora, args.output)
