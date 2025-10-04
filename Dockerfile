FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Atur direktori kerja di dalam container
WORKDIR /app

# Install dependensi sistem, termasuk Python, Pip, dan SOX
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    git-lfs \
    build-essential \
    curl \
    libsndfile1 \
    sox && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Salin file requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Gunakan versi PyTorch yang direkomendasikan
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Salin semua file aplikasi, termasuk entrypoint.sh
COPY . .
# Berikan izin eksekusi pada entrypoint script
RUN chmod +x /app/entrypoint.sh

# Buka port yang digunakan oleh Gradio
EXPOSE 7860

# Atur entrypoint untuk container
ENTRYPOINT ["/app/entrypoint.sh"]
