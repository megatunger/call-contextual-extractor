import os
import argparse
from unsloth import FastLanguageModel

def run_inference(model_name="data/finetuned_model_lora", transcript=""):
    if not os.path.exists(model_name):
        print(f"Error: Model directory '{model_name}' not found.")
        print("Please run the fine-tuning script to generate the LoRA adapters first.")
        return

    print(f"Loading finetuned model from: {model_name}...")
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

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

    print("\nPreparing input transcript...")
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction.strip(), # instruction
                transcript.strip(),  # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to(model.device)

    print("Generating structured JSON output...")
    # Generate the output tokens
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    
    # Decode and extract just the response
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    
    # Parse the output to get everything after '### Response:'
    response_marker = "### Response:\n"
    if response_marker in decoded:
        json_output = decoded.split(response_marker)[1].strip()
        print("\n--- Extracted JSON ---\n")
        print(json_output)
    else:
        print("\n--- Full Output ---\n")
        print(decoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using the finetuned model.")
    parser.add_argument("--model", type=str, default="data/finetuned_model_lora", 
                        help="Path to the trained LoRA adapters or base model")
    parser.add_argument("--transcript", type=str, default="Agent: Alo chị ơi, em bên dự án BDS X.\nCustomer: Chị bận lắm em ơi, cuối tuần gọi lại nhé.",
                        help="The transcript text to parse")
    
    args = parser.parse_args()
    run_inference(model_name=args.model, transcript=args.transcript)
