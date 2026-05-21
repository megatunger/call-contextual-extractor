import os
import argparse
import json
from unsloth import FastLanguageModel

def load_model(model_name="data/finetuned_model_lora"):
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Error: Model directory '{model_name}' not found. Please run the fine-tuning script to generate the LoRA adapters first.")
    
    print(f"Loading finetuned model from: {model_name}...")
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_extraction(model, tokenizer, transcript: str) -> dict:
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

    instruction = """You are a structured-field extractor for telesales CRM QA.
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
Return only JSON. Do not include markdown formatting or extra text."""

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction.strip(), # instruction
                transcript.strip(),  # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors="pt").to(model.device)

    # Generate the output tokens
    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    
    # Decode and extract just the response
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Parse the output to get everything after '### Response:'
    response_marker = "### Response:\n"
    if response_marker in decoded:
        json_output = decoded.split(response_marker)[1].strip()
        try:
            return json.loads(json_output)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON generated", "raw": json_output}
    else:
        return {"error": "Response marker not found", "raw": decoded}

def run_inference(model_name="data/finetuned_model_lora", transcript=""):
    try:
        model, tokenizer = load_model(model_name)
    except FileNotFoundError as e:
        print(e)
        return
        
    print("\nGenerating structured JSON output...")
    result = generate_extraction(model, tokenizer, transcript)
    
    print("\n--- Output ---\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the finetuned model.")
    parser.add_argument("--model", type=str, default="data/finetuned_model_lora", 
                        help="Path to the trained LoRA adapters or base model")
    parser.add_argument("--transcript", type=str, default="Agent: Alo chị ơi, em bên dự án BDS X.\nCustomer: Chị bận lắm em ơi, cuối tuần gọi lại nhé.",
                        help="The transcript text to parse")
    
    args = parser.parse_args()
    run_inference(model_name=args.model, transcript=args.transcript)
