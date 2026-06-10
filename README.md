# DevOps QLoRA Fine-Tuning Pipeline Instructions

**Repository**: [github.com/Jalpan04/devops-qlora-pipeline](https://github.com/Jalpan04/devops-qlora-pipeline)

Follow these exact steps to set up the environment, generate datasets, train the model, and convert the output to GGUF format on your 16GB VRAM PC.

---

## 1. System Setup (From Zero)

### Step 1: Install Git
* Download and run the installer from [git-scm.com](https://git-scm.com/).

### Step 2: Install Python (v3.11)
* Download the installer from [python.org](https://www.python.org/downloads/).
* **Important**: Check the box **"Add Python to PATH"** during installation.

### Step 3: Install NVIDIA CUDA Toolkit
* Download and run the CUDA 12.1 installer from [NVIDIA Developer Portal](https://developer.nvidia.com/cuda-downloads).

### Step 4: Clone the Repository
Open PowerShell and run:
```powershell
git clone <your_github_repo_url>
cd <your_repo_folder_name>
```

### Step 5: Install uv
uv is a fast Python package manager that replaces pip and venv.
```powershell
pip install uv
```

### Step 6: Setup Virtual Environment & Install Dependencies
Run the following commands in PowerShell:
```powershell
# Create environment and activate it
uv venv .venv
.venv\Scripts\Activate.ps1

# Install CUDA-enabled PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training libraries
uv pip install transformers peft bitsandbytes datasets accelerate trl google-genai
```

---

## 2. Download the Model

Run this once on the training PC to download all model weights locally (~15 GB):

```powershell
uv pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-7B --local-dir ./models/Qwen2.5-Coder-7B
```

Expected output:
```
Fetching 20 files: 100%|████████████████████| 20/20
./models/Qwen2.5-Coder-7B/config.json
./models/Qwen2.5-Coder-7B/tokenizer.json
./models/Qwen2.5-Coder-7B/model-00001-of-00004.safetensors
...
```
If the download is interrupted, just re-run the same command — it resumes automatically.

---

## 3. Continued Pre-Training (CPT)

> **Before running any training script, make sure your virtual environment is activated:**
> ```powershell
> .venv\Scripts\Activate.ps1
> ```
> Alternatively, prefix any command with `uv run` and it will use the project venv automatically without needing to activate it.

### Step 1: Verification Dry Run (3 Steps Only)
Verify that the model loads correctly and fits in VRAM before committing to a full run:
```powershell
uv run scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --dry_run
```

**Expected output (success looks like this):**
```
Starting QLoRA CPT Pipeline initialization for Qwen/Qwen2.5-Coder-7B...
VRAM at [Script Startup] - Allocated: 0.00 GB | Free: X.XX GB / 16.00 GB
Tokenizer loaded in X.XX seconds.
Dataset preparation complete. Total context blocks: XXXX
Loading base model 'Qwen/Qwen2.5-Coder-7B' in 4-bit precision...
Model loaded successfully in XX.XX seconds.
VRAM at [Model Loaded] - Allocated: X.XX GB | ...
Trainable Parameters: X,XXX,XXX | All Parameters: X,XXX,XXX | Ratio: X.XX%
Starting training loop...
{'loss': X.XXXX, 'learning_rate': X.Xe-XX, 'epoch': X.XX}
Training loop completed in X.XX minutes.
CPT Pipeline completed successfully in X.XX minutes!
```

If you see these lines and no `Error` or `OOM` messages, the environment is correctly configured. VRAM allocation at the `[Model Loaded]` step should be approximately 5–7 GB. If it exceeds 14 GB, reduce `batch_size` to `1` inside `scripts/train_cpt.py`.

### Step 2: Run the Full CPT Loop
Start the full training run after the dry run passes:
```powershell
uv run scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B
```

*All console output is also saved to `outputs/cpt_training.log`.*


## 3. Supervised Fine-Tuning (SFT) Data Generation

### Setup Gemini API Key
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Run in PowerShell:
   ```powershell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```

### Run SFT Data Generation
```powershell
uv run scripts/generate_sft_data.py
```
*Output will be appended directly to `03_ready_for_qlora/instructions.jsonl`.*

---

## 4. Convert to GGUF for Ollama

### Step 1: Merge the LoRA Adapter
Merge your trained weights back into the base model:
```powershell
uv run scripts/merge.py --base_model Qwen/Qwen2.5-Coder-7B --adapter_dir outputs/cpt_qlora_adapter --output_dir outputs/merged_devops_model
```

### Step 2: Convert to GGUF using llama.cpp
Run in your terminal:
```powershell
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install conversion requirements
uv pip install -r requirements.txt

# Convert to 16-bit GGUF
uv run convert_hf_to_gguf.py ../outputs/merged_devops_model/ --outfile ../outputs/devops_model.gguf

# Quantize to 4-bit (Optional)
./llama-quantize ../outputs/devops_model.gguf ../outputs/devops_model_Q4_K_M.gguf Q4_K_M
```

### Step 3: Load into Ollama
1. Create a `Modelfile` containing:
   ```dockerfile
   FROM ./outputs/devops_model_Q4_K_M.gguf
   SYSTEM You are an expert DevOps assistant.
   ```
2. Build and run:
   ```powershell
   ollama create devops-assistant -f Modelfile
   ollama run devops-assistant
   ```
