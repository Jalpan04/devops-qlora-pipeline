import os
import sys
import time
import logging
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("outputs/cpt_training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Helper function to log GPU memory usage
def log_vram_usage(step_description: str):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        free, total = torch.cuda.mem_get_info(device)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        logging.info(
            f"VRAM at [{step_description}] - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free: {free_gb:.2f} GB / {total_gb:.2f} GB"
        )
    else:
        logging.info(f"VRAM at [{step_description}] - CUDA not available (using CPU)")

# Custom callback to print VRAM during training steps
class VRAMLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0 or state.global_step <= 3:
            log_vram_usage(f"Step {state.global_step}")

def main():
    parser = argparse.ArgumentParser(description="QLoRA Continued Pre-Training (CPT) pipeline.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Hugging Face model identifier.")
    parser.add_argument("--dataset_path", type=str, default="03_ready_for_qlora/pretraining_clean.txt", help="Path to raw pretraining text dataset.")
    parser.add_argument("--output_dir", type=str, default="outputs/cpt_qlora_adapter", help="Directory to save the trained adapter.")
    parser.add_argument("--dry_run", action="store_true", help="Perform a quick 3-step training test to verify pipeline compatibility.")
    args = parser.parse_args()

    overall_start = time.time()
    os.makedirs("outputs", exist_ok=True)
    
    if args.dry_run:
        logging.info("!!! DRY-RUN MODE ENABLED (Will only train for 3 steps to test environment) !!!")
    
    logging.info(f"Starting QLoRA CPT Pipeline initialization for {args.model_id}...")
    log_vram_usage("Script Startup")

    # 2. Load Tokenizer
    tok_start = time.time()
    logging.info(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Tokenizer loaded in {time.time() - tok_start:.2f} seconds.")

    # 3. Load Dataset
    if not os.path.exists(args.dataset_path):
        logging.error(f"Dataset path '{args.dataset_path}' does not exist! Cannot proceed.")
        sys.exit(1)

    dataset_start = time.time()
    logging.info(f"Loading dataset file: {args.dataset_path}...")
    dataset = load_dataset("text", data_files={"train": args.dataset_path})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    block_size = 1024  # Sequence block length
    logging.info("Tokenizing text content (with progress bars)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing lines"
    )

    def group_texts(examples):
        concatenated_ids = []
        for ids in examples["input_ids"]:
            concatenated_ids.extend(ids)
        
        total_length = len(concatenated_ids)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        result = {
            "input_ids": [
                concatenated_ids[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logging.info(f"Grouping tokens into constant block sizes of {block_size}...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc="Creating sequence blocks"
    )
    
    total_blocks = len(lm_dataset["train"])
    logging.info(f"Dataset preparation complete in {time.time() - dataset_start:.2f} seconds. Total context blocks: {total_blocks}")
    if total_blocks == 0:
        logging.error("No context blocks were created. Check if your dataset file is empty or too small.")
        sys.exit(1)

    # 4. Load Quantized Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_start = time.time()
    logging.info(f"Loading base model '{args.model_id}' in 4-bit precision...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logging.info(f"Model loaded successfully in {time.time() - model_start:.2f} seconds.")
        log_vram_usage("Model Loaded")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Enable gradient checkpointing for VRAM savings
    logging.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()

    # Prepare model for kbit training
    logging.info("Preparing model for kbit training...")
    model = prepare_model_for_kbit_training(model)

    # 5. Apply LoRA Config
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    logging.info("Applying LoRA config adapter...")
    model = get_peft_model(model, peft_config)
    
    # Log trainable parameter details
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logging.info(
        f"Trainable Parameters: {trainable_params:,} | All Parameters: {all_param:,} | Ratio: {100 * trainable_params / all_param:.4f}%"
    )
    log_vram_usage("LoRA Applied")

    # 6. Configure Training Arguments
    # Accelerate training using TF32 if GPU supports it
    has_tf32 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if has_tf32:
        logging.info("Ampere/newer GPU detected. Enabling TensorFloat-32 (TF32) for fast QLoRA training.")
    
    # Hyperparameters
    batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    epochs = 1
    logging_steps = 1 if args.dry_run else 10
    save_steps = 99999 if args.dry_run else 100
    max_steps = 3 if args.dry_run else -1
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs if max_steps == -1 else 1,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="paged_adamw_32bit",
        logging_dir="./logs",
        report_to="none",
        remove_unused_columns=False,
        tf32=has_tf32,
        dataloader_num_workers=2,
    )

    # 7. Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[VRAMLoggingCallback()],
    )

    # 8. Execution
    logging.info("Starting training loop...")
    train_loop_start = time.time()
    try:
        trainer.train()
        logging.info(f"Training loop completed in {(time.time() - train_loop_start) / 60:.2f} minutes.")
    except Exception as e:
        logging.error(f"Training loop failed: {e}")
        log_vram_usage("Crash State")
        sys.exit(1)

    # 9. Save Results
    save_start = time.time()
    logging.info(f"Saving checkpoints and tokenizers to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info(f"Model saved successfully in {time.time() - save_start:.2f} seconds.")
    
    elapsed_time = (time.time() - overall_start) / 60
    logging.info(f"CPT Pipeline completed successfully in {elapsed_time:.2f} minutes!")
    log_vram_usage("Finished State")

if __name__ == "__main__":
    main()
