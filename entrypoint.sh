#!/bin/bash
set -e

# Definisikan direktori target untuk model
MODEL_DIR="/app/pretrained_models/FireRedTTS2"

# Cek apakah direktori model ada dan tidak kosong
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Model directory not found or is empty. Starting download..."
    
    # Membuat direktori induk jika belum ada
    mkdir -p /app/pretrained_models

    # Mengunduh model FireRedTTS2 menggunakan Git LFS
    echo "Downloading FireRedTTS2 model (step 1: clone without smudge)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/FireRedTeam/FireRedTTS2 "$MODEL_DIR"

    echo "Downloading FireRedTTS2 LFS files (step 2: pull)..."
    cd "$MODEL_DIR"
    git lfs pull
    cd /app # Kembali ke direktori utama
    
    echo "Model download complete."
else
    echo "Model directory already exists and is not empty. Skipping download."
fi

echo "Starting Gradio application..."
# Menjalankan aplikasi Gradio dan meneruskan path model yang benar
# `exec` menggantikan proses shell dengan proses python, ini adalah praktik yang baik.
exec python gradio_demo.py \
    --pretrained-dir "$MODEL_DIR" \
    --server-name "0.0.0.0" \
    --server-port "7860"
