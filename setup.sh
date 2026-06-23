#!/usr/bin/env bash
# setup.sh -- Automated environment setup for DevOps QLoRA CPT pipeline.
# Usage: bash setup.sh
# Tested on Ubuntu 22.04 / 24.04 with NVIDIA GPU and CUDA 12.x drivers.

set -e

echo "============================================"
echo "  DevOps QLoRA Pipeline - Environment Setup"
echo "============================================"
echo ""

# 1. Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3 python; do
    if command -v "$cmd" &> /dev/null; then
        PY_VERSION=$("$cmd" --version 2>&1 | awk '{print $2}')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.10-3.12 is required but not found."
    echo "        Install Python 3.11 and try again."
    echo "        Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "[OK] Found $($PYTHON_CMD --version)"

# 2. Check NVIDIA GPU
echo ""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[OK] GPU detected: $GPU_NAME ($GPU_MEM)"
else
    echo "[WARNING] nvidia-smi not found. Training will be very slow on CPU."
fi

# 3. Create virtual environment
echo ""
if [ -d ".venv" ]; then
    echo "[OK] Virtual environment already exists at .venv/"
else
    echo "[SETUP] Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "[OK] Virtual environment created."
fi

# 4. Activate and install packages
echo ""
echo "[SETUP] Installing dependencies (this may take a few minutes)..."
source .venv/bin/activate

pip install --upgrade pip --quiet

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

# Install Hugging Face training stack
pip install transformers peft bitsandbytes datasets accelerate trl --quiet

echo ""
echo "[OK] All dependencies installed."

# 5. Verify installation
echo ""
echo "[VERIFY] Checking PyTorch CUDA..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'[OK] CUDA available: {name} ({mem:.1f} GB VRAM)')
else:
    print('[WARNING] CUDA not available. Training will run on CPU (very slow).')
"

echo ""
echo "============================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run a dry run to verify everything works:"
echo "     python scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --dry_run --batch_size 1"
echo ""
echo "  3. Start full training:"
echo "     python scripts/train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --batch_size 2"
echo ""
echo "============================================"
