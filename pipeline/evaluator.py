import os
import json
import argparse
import pandas as pd
import difflib
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

def evaluate_models(models, dataset_path="data/finetuning_dataset_test.jsonl"):
    print("Loading test dataset...")
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found!")
        print("Run dataset_builder.py first to create the 80/20 train/test split.")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    test_dataset = dataset
    num_test = len(test_dataset)
    print(f"Evaluating on {num_test} held-out test examples from {dataset_path}.")
    
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
        
        parse_success = 0
        busy_match = 0
        sector_match = 0
        scheduled_match = 0
        interested_diffs = []
        rating_diffs = []
        
        for i, example in enumerate(test_dataset):
            print(f"  Testing sample {i+1}/{num_test}...", end="\r")
            
            instruction = example["instruction"]
            input_text = example["input"]
            ground_truth_text = example["response"]
            
            ground_truth = parse_json(ground_truth_text)
            if ground_truth is None:
                continue # Skip invalid ground truths
                
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
                
                # Boolean Accuracy: Busy
                if pred_json.get("customer_busy") == ground_truth.get("customer_busy"):
                    busy_match += 1
                    
                # Soft Match: Sector
                pred_sector = pred_json.get("customer_sector")
                gt_sector = ground_truth.get("customer_sector")
                if soft_match(pred_sector, gt_sector):
                    sector_match += 1
                    
                # Soft Match: Scheduled At
                pred_sched = pred_json.get("customer_scheduled_at")
                gt_sched = ground_truth.get("customer_scheduled_at")
                if soft_match(pred_sched, gt_sched):
                    scheduled_match += 1
                    
                # MAE: Interested
                diff_int = safe_mae(pred_json.get("customer_interested"), ground_truth.get("customer_interested"))
                if diff_int is not None:
                    interested_diffs.append(diff_int)
                    
                # MAE: Rating
                diff_rat = safe_mae(pred_json.get("customer_rating"), ground_truth.get("customer_rating"))
                if diff_rat is not None:
                    rating_diffs.append(diff_rat)
                    
        print("\n")
        # Calculate final metrics
        metrics = {
            "Model": os.path.basename(model_path).replace("_lora", ""),
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
    args = parser.parse_args()

    evaluate_models(args.models, dataset_path=args.dataset)
