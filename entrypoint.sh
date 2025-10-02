#!/bin/bash
set -e

if [ -z "$(ls -A /app/pretrained_models 2>/dev/null)" ]; then
    echo "Folder checkpoints not found"
    
    ### BLOK PERBAIKAN UNTUK TTS MODEL ###
    echo "Downloading FireRedTTS2 model (step 1: clone without smudge)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/FireRedTeam/FireRedTTS2 /app/pretrained_models/FireRedTTS2

    echo "Downloading FireRedTTS2 LFS files (step 2: pull)..."
    cd /app/pretrained_models/FireRedTTS2
    git lfs pull
    cd /app # Kembali ke direktori utama
    ### AKHIR BLOK PERBAIKAN ###
    
    echo "All model downloaded.."
else
    echo "Folder checkpoints already exists and is not empty. Skipping download."
fi

echo "Starting main application..."
exec uvicorn main:app --host 0.0.0.0 --port 8010
