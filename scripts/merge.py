import os
import sys
import time
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_vram_usage(step_name: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logging.info(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base Hugging Face models.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B", help="Base model identifier or path.")
    parser.add_argument("--adapter_dir", type=str, default="outputs/cpt_qlora_adapter", help="Directory where the trained LoRA adapter is saved.")
    parser.add_argument("--output_dir", type=str, default="outputs/merged_devops_model", help="Directory to save the merged model.")
    args = parser.parse_args()

    overall_start = time.time()
    
    # Validation checks
    if not os.path.exists(args.adapter_dir):
        logging.error(f"LoRA Adapter directory '{args.adapter_dir}' not found. Please run training first.")
        sys.exit(1)

    logging.info(f"Starting model merge process...")
    logging.info(f" - Base Model: {args.base_model}")
    logging.info(f" - Adapter Directory: {args.adapter_dir}")
    logging.info(f" - Output Directory: {args.output_dir}")
    get_vram_usage("Start")

    # 1. Load Base Model
    logging.info("Step 1/4: Loading base model (this may take a few minutes)...")
    load_base_start = time.time()
    try:
        # Load in 16-bit to preserve precision for merging (quantized models cannot be merged directly)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        load_base_time = time.time() - load_base_start
        logging.info(f"Base model loaded in {load_base_time:.2f} seconds.")
        get_vram_usage("Base Model Loaded")
    except Exception as e:
        logging.error(f"Failed to load base model: {e}")
        sys.exit(1)

    # 2. Load Tokenizer
    logging.info("Step 2/4: Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # 3. Load Peft/Adapter model
    logging.info("Step 3/4: Loading LoRA adapter weights...")
    load_adapter_start = time.time()
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
        load_adapter_time = time.time() - load_adapter_start
        logging.info(f"LoRA adapter weights loaded in {load_adapter_time:.2f} seconds.")
        get_vram_usage("Adapter Loaded")
    except Exception as e:
        logging.error(f"Failed to load LoRA adapter: {e}")
        sys.exit(1)

    # 4. Merge Adapter weights and Save
    logging.info("Step 4/4: Merging adapter weights into base model & unloading Peft wrapper...")
    merge_start = time.time()
    try:
        merged_model = model.merge_and_unload()
        merge_time = time.time() - merge_start
        logging.info(f"Model merged successfully in {merge_time:.2f} seconds.")
        get_vram_usage("Merged Model")
    except Exception as e:
        logging.error(f"Failed to merge model: {e}")
        sys.exit(1)

    # 5. Save Merged Model & Tokenizer
    logging.info(f"Saving merged model to '{args.output_dir}' (this may take a few minutes)...")
    save_start = time.time()
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        save_time = time.time() - save_start
        logging.info(f"Merged model saved in {save_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Failed to save merged model: {e}")
        sys.exit(1)

    overall_elapsed = (time.time() - overall_start) / 60
    logging.info(f"Merge pipeline completed successfully in {overall_elapsed:.2f} minutes!")

if __name__ == "__main__":
    main()
