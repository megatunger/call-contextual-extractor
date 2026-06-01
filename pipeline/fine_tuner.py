import os
os.environ["TORCHDYNAMO_DISABLE"] = "1" # Disable torch.compile to prevent Gemma-4 graph break crashes
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

def train_model(model_name="Qwen/Qwen3.5-0.8B", dataset_path="data/finetuning_dataset_train.jsonl", save_name="Qwen3.5-0.8B", epochs=1):
    output_dir = f"data/checkpoints_{save_name}"
    print(f"Loading model: {model_name}")
    
    max_seq_length = 1024 # Reduced from 2048 to save memory for 4B+ models
    dtype = None # None for auto detection
    load_in_4bit = True # Use 4bit quantization to reduce memory usage
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    
    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Define a chat template prompt format
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["response"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run dataset_builder.py first (writes train/test splits from the labeled pool)."
        )
        
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset successfully loaded. Found {len(dataset)} examples. Formatting prompts...")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    print("Formatting complete! Initializing SFTTrainer...")
    
    os.makedirs(output_dir, exist_ok=True)
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            max_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            num_train_epochs = epochs,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "paged_adamw_8bit", # Use paged optimizer to offload states to CPU RAM
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            save_steps = 20, # Save checkpoints every 20 steps
        ),
    )
    
    # Check for existing checkpoints to resume
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        print(f"Found {len(checkpoints)} existing checkpoints in {output_dir}. Resuming training from the latest checkpoint...")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training from scratch...")
        trainer_stats = trainer.train()
        
    print(f"Training finished! Final metrics: {trainer_stats.metrics}")
        
    # Save final model
    final_output_path = f"data/models/{save_name}_lora"
    os.makedirs(final_output_path, exist_ok=True)
    print(f"Training complete. Saving LoRA adapters to {final_output_path}")
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM using LoRA.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B", 
                        help="HuggingFace model ID to fine-tune")
    parser.add_argument("--dataset", type=str, default="data/finetuning_dataset_train.jsonl",
                        help="Path to the training JSONL (80%% split from dataset_builder.py)")
    
    parser.add_argument("--save_name", type=str, default="Qwen3.5-0.8B",
                        help="Name of the folder to save the trained model into")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of full dataset passes (epochs) to train for")
    
    args = parser.parse_args()
    train_model(model_name=args.model, dataset_path=args.dataset, save_name=args.save_name, epochs=args.epochs)
