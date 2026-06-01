import os
import json
import time
import argparse
import hashlib
import pandas as pd
import difflib
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel

def parse_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            text = text[start:end]
        return json.loads(text)
    except:
        return None

def soft_match(pred, gt):
    if not pred or not gt: return pred == gt
    pred = str(pred).strip().lower()
    gt = str(gt).strip().lower()
    if pred == gt: return True
    if pred in gt or gt in pred: return True
    if difflib.SequenceMatcher(None, pred, gt).ratio() > 0.8: return True
    return False

def safe_mae(pred, target):
    if pred is None or target is None:
        return None
    try:
        return abs(float(pred) - float(target))
    except:
        return None

def _checkpoint_path(checkpoint_dir: Path, model_path: str) -> Path:
    key = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(model_path))
    return checkpoint_dir / f"{safe_name}_{key}.json"

def _load_checkpoint(path: Path, model_path: str, dataset_path: str, num_test: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        state.get("model_path") != model_path
        or state.get("dataset_path") != dataset_path
        or state.get("num_test") != num_test
    ):
        return None
    return state

def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _format_eval_progress(model_label: str, done: int, total: int, elapsed_s: float) -> str:
    pct = (100.0 * done / total) if total else 0.0
    base = f"{model_label}: {done}/{total} ({pct:.1f}%)"
    if done <= 0 or elapsed_s <= 0:
        return f"{base} | elapsed {elapsed_s:.0f}s"
    rate = done / elapsed_s
    eta_s = (total - done) / rate if rate > 0 else 0.0
    return f"{base} | elapsed {elapsed_s:.0f}s | ETA {eta_s:.0f}s"

def _model_display_info(model_path: str) -> tuple[str, str]:
    name = os.path.basename(model_path.rstrip("/"))
    is_finetuned = name.endswith("_lora")
    model_label = name.replace("_lora", "")
    variant = "fine-tuned" if is_finetuned else "base"
    return model_label, variant

def _metrics_from_counts(
    model_label: str,
    variant: str,
    num_test: int,
    parse_success: int,
    busy_match: int,
    sector_match: int,
    scheduled_match: int,
    interested_diffs: list,
    rating_diffs: list,
) -> dict:
    return {
        "Model": model_label,
        "Variant": variant,
        "Valid_JSON_%": round((parse_success / num_test) * 100, 2),
        "Busy_Accuracy_%": round((busy_match / parse_success) * 100, 2) if parse_success > 0 else 0,
        "Sector_Match_%": round((sector_match / parse_success) * 100, 2) if parse_success > 0 else 0,
        "Schedule_Match_%": round((scheduled_match / parse_success) * 100, 2) if parse_success > 0 else 0,
        "Interested_MAE": round(sum(interested_diffs) / len(interested_diffs), 2) if interested_diffs else None,
        "Interested_MSE": round(sum(d**2 for d in interested_diffs) / len(interested_diffs), 2) if interested_diffs else None,
        "Interested_Acc_±1_%": round(sum(1 for d in interested_diffs if d <= 1) / len(interested_diffs) * 100, 2) if interested_diffs else None,
        "Rating_MAE": round(sum(rating_diffs) / len(rating_diffs), 2) if rating_diffs else None,
        "Rating_MSE": round(sum(d**2 for d in rating_diffs) / len(rating_diffs), 2) if rating_diffs else None,
        "Rating_Acc_±1_%": round(sum(1 for d in rating_diffs if d <= 1) / len(rating_diffs) * 100, 2) if rating_diffs else None,
    }

def evaluate_models(
    models,
    dataset_path="data/finetuning_dataset_test.jsonl",
    checkpoint_dir="data/evaluation_checkpoints",
    resume=True,
):
    print("Loading test dataset...")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found!")
        print("Run dataset_builder.py first to create the 80/20 train/test split.")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    test_dataset = dataset
    num_test = len(test_dataset)
    dataset_path = os.path.abspath(dataset_path)
    ckpt_dir = Path(checkpoint_dir)
    print(f"Evaluating on {num_test} held-out test examples from {dataset_path}.")
    if resume:
        print(f"Checkpoints: {ckpt_dir} (use --no-resume to start fresh)")
    else:
        print("Resume disabled; ignoring any saved checkpoints.")
    
    results = []
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    
    for model_path in models:
        print(f"\n======================================")
        print(f"Evaluating model: {model_path}")
        print(f"======================================")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True,
            )
            FastLanguageModel.for_inference(model)
        except Exception as e:
            print(f"Skipping {model_path} - could not load model. Error: {e}")
            continue
        
        model_label, variant = _model_display_info(model_path)
        progress_label = f"{model_label} ({variant})"
        ckpt_path = _checkpoint_path(ckpt_dir, model_path)
        if not resume and ckpt_path.is_file():
            ckpt_path.unlink()

        parse_success = 0
        busy_match = 0
        sector_match = 0
        scheduled_match = 0
        interested_diffs = []
        rating_diffs = []
        start_index = 0

        if resume:
            saved = _load_checkpoint(ckpt_path, model_path, dataset_path, num_test)
            if saved:
                start_index = saved.get("next_index", 0)
                parse_success = saved.get("parse_success", 0)
                busy_match = saved.get("busy_match", 0)
                sector_match = saved.get("sector_match", 0)
                scheduled_match = saved.get("scheduled_match", 0)
                interested_diffs = list(saved.get("interested_diffs", []))
                rating_diffs = list(saved.get("rating_diffs", []))
                print(
                    f"Resuming from sample {start_index + 1}/{num_test} "
                    f"(checkpoint: {ckpt_path.name})"
                )

        loop_start = time.monotonic()
        
        for i, example in enumerate(test_dataset):
            if i < start_index:
                continue

            done = i + 1
            elapsed = time.monotonic() - loop_start
            print(_format_eval_progress(progress_label, done, num_test, elapsed), flush=True)
            
            instruction = example["instruction"]
            input_text = example["input"]
            ground_truth_text = example["response"]
            
            ground_truth = parse_json(ground_truth_text)
            if ground_truth is not None:
                inputs = tokenizer([alpaca_prompt.format(instruction, input_text)], return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                response_marker = "### Response:\n"
                if response_marker in decoded:
                    pred_text = decoded.split(response_marker)[1].strip()
                else:
                    pred_text = decoded

                pred_json = parse_json(pred_text)

                if pred_json is not None:
                    parse_success += 1

                    if pred_json.get("customer_busy") == ground_truth.get("customer_busy"):
                        busy_match += 1

                    pred_sector = pred_json.get("customer_sector")
                    gt_sector = ground_truth.get("customer_sector")
                    if soft_match(pred_sector, gt_sector):
                        sector_match += 1

                    pred_sched = pred_json.get("customer_scheduled_at")
                    gt_sched = ground_truth.get("customer_scheduled_at")
                    if soft_match(pred_sched, gt_sched):
                        scheduled_match += 1

                    diff_int = safe_mae(pred_json.get("customer_interested"), ground_truth.get("customer_interested"))
                    if diff_int is not None:
                        interested_diffs.append(diff_int)

                    diff_rat = safe_mae(pred_json.get("customer_rating"), ground_truth.get("customer_rating"))
                    if diff_rat is not None:
                        rating_diffs.append(diff_rat)

            _save_checkpoint(
                ckpt_path,
                {
                    "model_path": model_path,
                    "dataset_path": dataset_path,
                    "num_test": num_test,
                    "next_index": i + 1,
                    "parse_success": parse_success,
                    "busy_match": busy_match,
                    "sector_match": sector_match,
                    "scheduled_match": scheduled_match,
                    "interested_diffs": interested_diffs,
                    "rating_diffs": rating_diffs,
                },
            )
                    
        print()
        if ckpt_path.is_file():
            ckpt_path.unlink()

        metrics = _metrics_from_counts(
            model_label,
            variant,
            num_test,
            parse_success,
            busy_match,
            sector_match,
            scheduled_match,
            interested_diffs,
            rating_diffs,
        )
        
        print(f"Results for {metrics['Model']}:")
        print(metrics)
        results.append(metrics)
        
        # Free memory before loading next model
        import torch
        del model, tokenizer
        torch.cuda.empty_cache()
        
    if results:
        # Save results
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv("data/evaluation_results.csv", index=False)
        print("\nEvaluation complete! Results saved to data/evaluation_results.csv")
    else:
        print("\nNo models were successfully evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple fine-tuned models")
    parser.add_argument("--models", nargs="+", default=[
        "data/models/Qwen3.5-2B_lora", 
        "data/models/Qwen3.5-0.8B_lora", 
        "data/models/gemma-4-E4B_lora"
    ], help="List of model paths to evaluate")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/finetuning_dataset_test.jsonl",
        help="Path to the test JSONL (20%% split from dataset_builder.py)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/evaluation_checkpoints",
        help="Directory for per-model progress checkpoints (enables resume)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete checkpoints and evaluate every sample from scratch",
    )
    args = parser.parse_args()

    evaluate_models(
        args.models,
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )
