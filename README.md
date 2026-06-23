# DevOps QLoRA Fine-Tuning Pipeline

Fine-tune Qwen2.5-Coder-7B on DevOps domain data using QLoRA (4-bit quantized LoRA).
Designed to run on a single consumer GPU (16 GB VRAM recommended, 8 GB minimum for dry runs).

## What Is This?

This repo contains:
- A **467 MB DevOps text corpus** (documentation, code, manpages, research papers) for Continued Pre-Training (CPT)
- A **modular training script** with full CLI control over hyperparameters
- A **merge script** to combine the trained LoRA adapter back into the base model
- Automated setup scripts for both **Linux** and **Windows**

The trained model is intended to serve as the LLM reasoning engine in [DocRAFT](https://github.com/ayoitssmit/DocRAFT).

---

## Quick Start (3 Commands)

### Prerequisites
- Python 3.11 (3.10-3.12 supported)
- NVIDIA GPU with CUDA drivers installed
- Git with Git LFS (`git lfs install`)

### Step 1: Clone
```bash
git clone https://github.com/Jalpan04/devops-qlora-pipeline.git
cd devops-qlora-pipeline
git lfs pull
```

### Step 2: Setup Environment
The setup script creates a virtual environment and installs PyTorch (CUDA 12.1) and the entire Hugging Face training stack automatically.

**Linux:**
```bash
bash setup.sh
```

**Windows:**
```bat
setup.bat
```

### Step 3: Train

**Dry run first** (verifies model loading, VRAM fit, and training loop in 3 steps):
```bash
# Linux
source .venv/bin/activate
python scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --dry_run --batch_size 1

# Windows
.venv\Scripts\activate.bat
python scripts\train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --dry_run --batch_size 1
```

**Full training** (runs 1 epoch over the entire 467 MB corpus):
```bash
python scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --batch_size 2
```

Training logs are saved to `outputs/cpt_training.log`. The trained LoRA adapter is saved to `outputs/cpt_qlora_adapter/`.

---

## After Training: Merge and Deploy

### Merge LoRA Adapter into Base Model
```bash
python scripts/merge.py \
  --base_model Qwen/Qwen2.5-Coder-7B \
  --adapter_dir outputs/cpt_qlora_adapter \
  --output_dir outputs/merged_devops_model
```

### Convert to GGUF for Ollama
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Convert to GGUF
python convert_hf_to_gguf.py ../outputs/merged_devops_model/ --outfile ../outputs/devops_model.gguf

# Quantize to 4-bit (optional, reduces size)
./llama-quantize ../outputs/devops_model.gguf ../outputs/devops_model_Q4_K_M.gguf Q4_K_M
```

### Load into Ollama
Create a file named `Modelfile`:
```dockerfile
FROM ./outputs/devops_model_Q4_K_M.gguf
SYSTEM You are an expert DevOps assistant.
```

```bash
ollama create devops-assistant -f Modelfile
ollama run devops-assistant
```

---

## Training Parameters

All hyperparameters are configurable via CLI flags. No code changes needed to scale up or down.

| Flag | Default | Description |
|------|---------|-------------|
| `--model_id` | `Qwen/Qwen2.5-Coder-7B` | Hugging Face model ID or local path |
| `--dataset_path` | `data/pretraining_clean.txt` | Path to the CPT text corpus |
| `--output_dir` | `outputs/cpt_qlora_adapter` | Where to save the trained adapter |
| `--dry_run` | off | Run only 3 steps to verify setup |
| `--batch_size` | `2` | Per-GPU batch size |
| `--gradient_accumulation_steps` | `4` | Steps before weight update |
| `--learning_rate` | `2e-4` | Training learning rate |
| `--epochs` | `1` | Number of full passes over the data |
| `--block_size` | `1024` | Context window length in tokens |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha scaling |
| `--lora_dropout` | `0.05` | LoRA dropout rate |
| `--max_steps` | `-1` | Limit total steps (-1 = unlimited) |
| `--save_steps` | `100` | Save checkpoint every N steps |
| `--logging_steps` | `10` | Log metrics every N steps |

**Example: Custom training run**
```bash
python scripts/train_cpt.py \
  --model_id Qwen/Qwen2.5-Coder-7B \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --epochs 2 \
  --save_steps 200 \
  --logging_steps 20
```

---

## Repo Structure

```
devops-qlora-pipeline/
├── data/
│   └── pretraining_clean.txt    # 467 MB DevOps CPT corpus (Git LFS)
├── scripts/
│   ├── train_cpt.py             # CPT training script (QLoRA)
│   └── merge.py                 # LoRA adapter merge script
├── setup.sh                     # Linux automated setup
├── setup.bat                    # Windows automated setup
├── requirements.txt             # Python dependencies
├── .gitattributes               # Git LFS tracking
├── .gitignore
└── README.md
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (dry run only) | 16 GB (full 7B training) |
| System RAM | 16 GB | 64 GB |
| Disk Space | 20 GB | 50 GB (for model weights + checkpoints) |
| CUDA | 12.x | 12.1+ |
| Python | 3.10 | 3.11 |

---

## Authors

Built by Smit Shah & Jalpan Vyas
