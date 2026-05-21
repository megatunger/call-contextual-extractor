import os
import json
import glob
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

# Use gemini-3.1-flash-lite as requested
MODEL_NAME = "gemini-3.1-flash-lite"

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

def build_dataset(base_dir="data", output_file="data/finetuning_dataset.jsonl"):
    """Iterate over all final_dialogue.json and build the dataset."""
    search_pattern = os.path.join(base_dir, "segmented_audio", "*", "final_dialogue.json")
    dialogue_files = glob.glob(search_pattern)
    
    # Load already processed transcripts to support pause/resume
    processed_transcripts = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                processed_transcripts.add(data.get("input", ""))

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
        if not transcript_text.strip():
            return None, transcript_text, "empty"
            
        if transcript_text in processed_transcripts:
            return None, transcript_text, "processed"
            
        # Rate limit: max 15 requests per minute (1 every 4 seconds)
        time.sleep(4.1)
            
        extracted_json = extract_fields_from_transcript(transcript_text)
        if extracted_json:
            record = {
                "instruction": PROMPT_INSTRUCTION.strip(),
                "input": transcript_text,
                "response": json.dumps(extracted_json, ensure_ascii=False)
            }
            return record, transcript_text, "success"
        return None, transcript_text, "failed"

    with open(output_file, "a", encoding="utf-8") as out_f:
        # Rate limit: 1 worker ensures strict sequential timing
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_file = {executor.submit(process_file, fp): fp for fp in dialogue_files}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(dialogue_files), desc="Building Dataset"):
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

if __name__ == "__main__":
    build_dataset()
