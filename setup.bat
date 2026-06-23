@echo off
REM setup.bat -- Automated environment setup for DevOps QLoRA CPT pipeline (Windows).
REM Usage: setup.bat
REM Tested on Windows 10/11 with NVIDIA GPU and CUDA 12.x drivers.

echo ============================================
echo   DevOps QLoRA Pipeline - Environment Setup
echo ============================================
echo.

REM 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11 from python.org and add to PATH.
    exit /b 1
)

python --version
echo [OK] Python found.
echo.

REM 2. Check NVIDIA GPU
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo [WARNING] nvidia-smi not found. Training will be very slow on CPU.
)
echo.

REM 3. Create virtual environment
if exist ".venv" (
    echo [OK] Virtual environment already exists at .venv\
) else (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    echo [OK] Virtual environment created.
)
echo.

REM 4. Activate and install packages
echo [SETUP] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat

pip install --upgrade pip --quiet

REM Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

REM Install Hugging Face training stack
pip install transformers peft bitsandbytes datasets accelerate trl --quiet

echo.
echo [OK] All dependencies installed.
echo.

REM 5. Verify installation
echo [VERIFY] Checking PyTorch CUDA...
python -c "import torch; print(f'[OK] CUDA available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB VRAM)') if torch.cuda.is_available() else print('[WARNING] CUDA not available.')"

echo.
echo ============================================
echo   Setup complete. Next steps:
echo.
echo   1. Activate the environment:
echo      .venv\Scripts\activate.bat
echo.
echo   2. Run a dry run to verify everything works:
echo      python scripts\train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --dry_run --batch_size 1
echo.
echo   3. Start full training:
echo      python scripts\train_cpt.py --model_id Qwen/Qwen2.5-Coder-7B --batch_size 2
echo.
echo ============================================
