import os
import json
import time
import argparse
import hashlib
from datetime import datetime, timezone
import pandas as pd
import difflib
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel

LIVE_RESULTS_CSV = "data/evaluation_results_live.csv"
PROGRESS_JSON = "data/evaluation_progress.json"

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

def _run_id(model_path: str) -> tuple[str, str]:
    key = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(model_path))
    return safe_name, key

def _checkpoint_path(checkpoint_dir: Path, model_path: str) -> Path:
    safe_name, key = _run_id(model_path)
    return checkpoint_dir / f"{safe_name}_{key}.json"

def _samples_path(checkpoint_dir: Path, model_path: str) -> Path:
    safe_name, key = _run_id(model_path)
    return checkpoint_dir / f"{safe_name}_{key}.samples.jsonl"

def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def _save_live_results(live_path: Path, completed: list[dict], current: dict | None) -> None:
    rows = list(completed)
    if current:
        rows.append(current)
    if not rows:
        return
    live_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(live_path, index=False)

def _save_progress(progress_path: Path, payload: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(progress_path)

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

def _format_eval_progress(
    model_label: str,
    done: int,
    total: int,
    elapsed_s: float,
    running: dict | None = None,
) -> str:
    pct = (100.0 * done / total) if total else 0.0
    base = f"{model_label}: {done}/{total} ({pct:.1f}%)"
    if running:
        base += (
            f" | JSON {running.get('Valid_JSON_%', 0)}%"
            f" Busy {running.get('Busy_Accuracy_%', 0)}%"
            f" Sector {running.get('Sector_Match_%', 0)}%"
        )
    if done <= 0 or elapsed_s <= 0:
        return f"{base} | elapsed {elapsed_s:.0f}s"
    rate = done / elapsed_s
    eta_s = (total - done) / rate if rate > 0 else 0.0
    return f"{base} | elapsed {elapsed_s:.0f}s | ETA {eta_s:.0f}s"

def _score_sample(pred_json: dict | None, ground_truth: dict | None) -> dict:
    if ground_truth is None:
        return {"valid_json": False, "skipped": True, "reason": "invalid_ground_truth"}
    if pred_json is None:
        return {
            "valid_json": False,
            "skipped": False,
            "busy_match": False,
            "sector_match": False,
            "schedule_match": False,
            "interested_abs_err": None,
            "rating_abs_err": None,
        }
    scores = {
        "valid_json": True,
        "skipped": False,
        "busy_match": pred_json.get("customer_busy") == ground_truth.get("customer_busy"),
        "sector_match": soft_match(pred_json.get("customer_sector"), ground_truth.get("customer_sector")),
        "schedule_match": soft_match(
            pred_json.get("customer_scheduled_at"), ground_truth.get("customer_scheduled_at")
        ),
        "interested_abs_err": safe_mae(
            pred_json.get("customer_interested"), ground_truth.get("customer_interested")
        ),
        "rating_abs_err": safe_mae(pred_json.get("customer_rating"), ground_truth.get("customer_rating")),
    }
    return scores

def _apply_sample_scores(
    scores: dict,
    parse_success: int,
    busy_match: int,
    sector_match: int,
    scheduled_match: int,
    interested_diffs: list,
    rating_diffs: list,
) -> tuple[int, int, int, int, list, list]:
    if not scores.get("valid_json"):
        return parse_success, busy_match, sector_match, scheduled_match, interested_diffs, rating_diffs
    parse_success += 1
    if scores.get("busy_match"):
        busy_match += 1
    if scores.get("sector_match"):
        sector_match += 1
    if scores.get("schedule_match"):
        scheduled_match += 1
    if scores.get("interested_abs_err") is not None:
        interested_diffs.append(scores["interested_abs_err"])
    if scores.get("rating_abs_err") is not None:
        rating_diffs.append(scores["rating_abs_err"])
    return parse_success, busy_match, sector_match, scheduled_match, interested_diffs, rating_diffs

def _running_metrics_row(
    model_label: str,
    variant: str,
    model_path: str,
    samples_done: int,
    num_test: int,
    parse_success: int,
    busy_match: int,
    sector_match: int,
    scheduled_match: int,
    interested_diffs: list,
    rating_diffs: list,
    status: str,
) -> dict:
    row = _metrics_from_counts(
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
    row["Model_Path"] = model_path
    row["Samples_Done"] = samples_done
    row["Num_Test"] = num_test
    row["Status"] = status
    return row

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
    live_path = Path(LIVE_RESULTS_CSV)
    progress_path = Path(PROGRESS_JSON)

    if resume:
        print(f"Checkpoints: {ckpt_dir} (use --no-resume to start fresh)")
    else:
        print("Resume disabled; ignoring any saved checkpoints.")
    print(f"Live summary: {live_path}")
    print(f"Per-sample logs: {ckpt_dir}/*.samples.jsonl")
    print(f"Progress file: {progress_path}")

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
        samples_path = _samples_path(ckpt_dir, model_path)
        if not resume:
            if ckpt_path.is_file():
                ckpt_path.unlink()
            if samples_path.is_file():
                samples_path.unlink()

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

            instruction = example["instruction"]
            input_text = example["input"]
            ground_truth_text = example["response"]

            ground_truth = parse_json(ground_truth_text)
            pred_text = None
            pred_json = None

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

            scores = _score_sample(pred_json, ground_truth)
            (
                parse_success,
                busy_match,
                sector_match,
                scheduled_match,
                interested_diffs,
                rating_diffs,
            ) = _apply_sample_scores(
                scores,
                parse_success,
                busy_match,
                sector_match,
                scheduled_match,
                interested_diffs,
                rating_diffs,
            )

            running = _running_metrics_row(
                model_label,
                variant,
                model_path,
                done,
                num_test,
                parse_success,
                busy_match,
                sector_match,
                scheduled_match,
                interested_diffs,
                rating_diffs,
                status="in_progress",
            )
            print(_format_eval_progress(progress_label, done, num_test, elapsed, running), flush=True)

            sample_record = {
                "index": i,
                "model": model_label,
                "variant": variant,
                "model_path": model_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scores": scores,
                "prediction": pred_json,
                "prediction_text": pred_text,
                "ground_truth": ground_truth,
            }
            _append_jsonl(samples_path, sample_record)

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
                    "samples_path": str(samples_path),
                },
            )

            _save_live_results(live_path, results, running)
            _save_progress(
                progress_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_path": dataset_path,
                    "current_model_path": model_path,
                    "current_model": model_label,
                    "current_variant": variant,
                    "sample_index": i,
                    "samples_done": done,
                    "num_test": num_test,
                    "running_metrics": running,
                    "completed_models": results,
                    "samples_log": str(samples_path),
                    "live_results_csv": str(live_path),
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
        
        metrics["Model_Path"] = model_path
        metrics["Samples_Done"] = num_test
        metrics["Num_Test"] = num_test
        metrics["Status"] = "complete"
        metrics["Samples_Log"] = str(samples_path)

        print(f"Results for {metrics['Model']} ({variant}):")
        print(metrics)
        results.append(metrics)

        _save_live_results(live_path, results, current=None)
        _save_progress(
            progress_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_path": dataset_path,
                "current_model_path": None,
                "samples_done": num_test,
                "num_test": num_test,
                "completed_models": results,
                "live_results_csv": str(live_path),
            },
        )

        # Free memory before loading next model
        import torch
        del model, tokenizer
        torch.cuda.empty_cache()
        
    if results:
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(results)
        final_cols = [c for c in df.columns if c not in ("Status", "Samples_Log")]
        df[final_cols].to_csv("data/evaluation_results.csv", index=False)
        df.to_csv(LIVE_RESULTS_CSV, index=False)
        print("\nEvaluation complete!")
        print("  Final metrics: data/evaluation_results.csv")
        print(f"  Live summary: {LIVE_RESULTS_CSV}")
        print(f"  Per-sample logs: {ckpt_dir}/*.samples.jsonl")
        print(f"  Progress: {PROGRESS_JSON}")
    else:
        print("\nNo models were successfully evaluated.")

if __name__ == "__main__":
    # Allow `python pipeline/evaluator.py` from any cwd (e.g. subprocess / notebooks)
    os.chdir(Path(__file__).resolve().parent.parent)

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
