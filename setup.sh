#!/bin/bash
# =============================================================================
# setup.sh — install GNN-AD-Navigator dependencies
# =============================================================================
#
# Creates a Python virtual environment at ./venv, installs PyTorch and
# torch-geometric appropriate for the host system (CPU or CUDA), and
# verifies everything imports correctly.
#
# Disk usage:
#   CPU build  : ~700 MB
#   CUDA build : ~2.1 GB
#
# Usage:
#   ./setup.sh           # auto-detect CPU/CUDA
#   ./setup.sh --cpu     # force CPU-only install (smaller, no GPU)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

FORCE_CPU=0
[[ "${1:-}" == "--cpu" ]] && FORCE_CPU=1

echo "============================================================"
echo "GNN-AD-Navigator setup"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo ""

# ── [1/5] Python version check ─────────────────────────────────────────────
echo "[1/5] Checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "  ✗ python3 not found. Install Python 3.10+ first."
    exit 1
fi

py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=$(echo "$py_version" | cut -d. -f1)
py_minor=$(echo "$py_version" | cut -d. -f2)

if [[ $py_major -lt 3 || ($py_major -eq 3 && $py_minor -lt 10) ]]; then
    echo "  ✗ Python $py_version detected. Requires 3.10+."
    exit 1
fi
echo "  ✓ Python $py_version"

# ── [2/5] virtualenv ───────────────────────────────────────────────────────
echo ""
echo "[2/5] Creating virtual environment..."
if [[ -d "$VENV_DIR" ]]; then
    echo "  (existing venv detected at $VENV_DIR — reusing)"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ venv created at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── [3/5] detect CPU vs CUDA ───────────────────────────────────────────────
echo ""
echo "[3/5] Detecting compute backend..."

USE_CUDA=0
if [[ $FORCE_CPU -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        USE_CUDA=1
    fi
fi

if [[ $USE_CUDA -eq 1 ]]; then
    echo "  ✓ CUDA detected — installing GPU build (~2.1 GB)"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    PYG_INDEX="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
else
    echo "  ✓ CPU-only install (~700 MB)"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    PYG_INDEX="https://data.pyg.org/whl/torch-2.1.0+cpu.html"
fi

# ── [4/5] install dependencies ─────────────────────────────────────────────
echo ""
echo "[4/5] Installing dependencies (this can take 3-5 minutes)..."

# torch first — pinned to a stable version PyG ships wheels for
echo "  → torch"
pip install --quiet --index-url "$TORCH_INDEX" "torch==2.1.0"

# numpy
echo "  → numpy"
pip install --quiet "numpy>=1.24"

# torch-geometric with native extensions from PyG's wheel index
echo "  → torch-geometric (+ scatter, sparse extensions)"
pip install --quiet torch-geometric
pip install --quiet -f "$PYG_INDEX" torch-scatter torch-sparse || {
    echo "  ⚠ optional native extensions failed — torch-geometric will use"
    echo "    slower fallback Python implementations. This is fine for inference."
}

echo "  ✓ dependencies installed"

# ── [5/5] permissions + verification ───────────────────────────────────────
echo ""
echo "[5/5] Finalising..."
chmod +x "$PROJECT_DIR/pipeline.sh"
echo "  ✓ pipeline.sh executable"

# verification — import everything and report versions
python - <<'PY'
import torch
import torch_geometric
import numpy as np

print(f"  ✓ torch             {torch.__version__}")
print(f"  ✓ torch_geometric   {torch_geometric.__version__}")
print(f"  ✓ numpy             {np.__version__}")
print(f"  ✓ CUDA available    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    Device: {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "============================================================"
echo "Setup complete"
echo "============================================================"
echo ""
echo "Activate the environment before using the tool:"
echo "    source venv/bin/activate"
echo ""
echo "Then run the pipeline:"
echo "    ./pipeline.sh ./input ./output \\"
echo "        --start  <node> \\"
echo "        --target <node>"
echo ""
echo "See README.md for details."