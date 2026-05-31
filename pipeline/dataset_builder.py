import argparse
import json
import glob
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env. Please add it to run this script.")
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.1-flash-lite"

TRAIN_RATIO = 0.8
SPLIT_SEED = 42

DEFAULT_POOL_FILE = "data/finetuning_dataset.jsonl"
TRAIN_FILE = "data/finetuning_dataset_train.jsonl"
TEST_FILE = "data/finetuning_dataset_test.jsonl"
SPLIT_MANIFEST_FILE = "data/finetuning_dataset_split.json"

PROMPT_INSTRUCTION = """You are a structured-field extractor for telesales CRM QA.
Read the following transcript (agent and customer) and extract the customer information.
Output valid JSON matching this schema exactly:
{
    "customer_sector": "string (industry or sector, or null if unknown)",
    "customer_needs": "string (brief summary of what the customer needs)",
    "customer_interested": "number (1-10 scale of interest)",
    "customer_busy": "boolean (true if they said they are busy, false otherwise)",
    "customer_scheduled_at": "string (time or date they want to be called back, or null)",
    "customer_rating": "number (optional 1-10 rating, if available)"
}
Return only JSON. Do not include markdown formatting or extra text.
"""


def load_records(jsonl_path: str) -> list[dict]:
    """Load all JSONL records from a file."""
    records = []
    path = Path(jsonl_path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(jsonl_path: str, records: list[dict]) -> None:
    """Write records to a JSONL file (overwrites existing file)."""
    path = Path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_train_test(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split records into train and test sets (default 80/20)."""
    if not records:
        return [], []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []
    n_train = int(len(shuffled) * train_ratio)
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return shuffled[:n_train], shuffled[n_train:]


def write_train_test_splits(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SPLIT_SEED,
    train_path: str = TRAIN_FILE,
    test_path: str = TEST_FILE,
    manifest_path: str = SPLIT_MANIFEST_FILE,
) -> tuple[list[dict], list[dict]]:
    """Write train/test JSONL files and a small manifest with split metadata."""
    train_records, test_records = split_train_test(records, train_ratio=train_ratio, seed=seed)
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    manifest = {
        "pool_size": len(records),
        "train_size": len(train_records),
        "test_size": len(test_records),
        "train_ratio": train_ratio,
        "seed": seed,
        "train_file": train_path,
        "test_file": test_path,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Split {len(records)} examples -> "
        f"train={len(train_records)} ({train_ratio:.0%}), "
        f"test={len(test_records)} (seed={seed})"
    )
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    return train_records, test_records


def extract_fields_from_transcript(transcript_text):
    """Call Gemini to extract fields from the transcript."""
    if not client:
        return None
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{PROMPT_INSTRUCTION}\n\nTranscript:\n{transcript_text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return None


def build_dataset(base_dir="data", output_file=DEFAULT_POOL_FILE):
    """Label dialogues with Gemini, append to the pool file, then write 80/20 train/test splits."""
    search_pattern = os.path.join(base_dir, "segmented_audio", "*", "final_dialogue.json")
    dialogue_files = glob.glob(search_pattern)

    processed_transcripts = set()
    if os.path.exists(output_file):
        for record in load_records(output_file):
            processed_transcripts.add(record.get("input", ""))

    print(f"Found {len(dialogue_files)} dialogue files. {len(processed_transcripts)} already processed.")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    added_count = 0
    import concurrent.futures
    from tqdm import tqdm
    import threading

    write_lock = threading.Lock()

    def process_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            dialogue_lines = json.load(f)

        transcript_text = "\n".join(dialogue_lines)
        call_id = Path(file_path).parent.name
        if not transcript_text.strip():
            return None, transcript_text, "empty"

        if transcript_text in processed_transcripts:
            return None, transcript_text, "processed"

        time.sleep(4.1)

        extracted_json = extract_fields_from_transcript(transcript_text)
        if extracted_json:
            record = {
                "call_id": call_id,
                "instruction": PROMPT_INSTRUCTION.strip(),
                "input": transcript_text,
                "response": json.dumps(extracted_json, ensure_ascii=False),
            }
            return record, transcript_text, "success"
        return None, transcript_text, "failed"

    with open(output_file, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_file = {
                executor.submit(process_file, fp): fp for fp in dialogue_files
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(dialogue_files),
                desc="Building Dataset",
            ):
                try:
                    record, transcript_text, status = future.result()
                    if status == "success" and record:
                        with write_lock:
                            if transcript_text not in processed_transcripts:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                out_f.flush()
                                processed_transcripts.add(transcript_text)
                                added_count += 1
                except Exception as e:
                    fp = future_to_file[future]
                    print(f"Error processing {fp}: {e}")

    print(f"Dataset building complete. Added {added_count} new records to {output_file}")

    pool_records = load_records(output_file)
    if pool_records:
        write_train_test_splits(pool_records)
    else:
        print("No labeled records in pool; skipping train/test split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build finetuning JSONL and 80/20 train/test splits.")
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Re-split an existing pool JSONL into train/test without calling Gemini.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_POOL_FILE,
        help="Pool JSONL to split (used with --split-only).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of examples for training (default 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="Random seed for the train/test split.",
    )
    args = parser.parse_args()

    if args.split_only:
        records = load_records(args.input)
        if not records:
            print(f"No records found at {args.input}")
        else:
            write_train_test_splits(
                records,
                train_ratio=args.train_ratio,
                seed=args.seed,
            )
    else:
        build_dataset(output_file=args.input)
