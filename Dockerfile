FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Atur direktori kerja di dalam container
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install dependensi
RUN apt-get update && apt-get install -y git git-lfs \
    python3.10 python3-pip git build-essential wget curl \
    libsndfile1-dev \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Salin file requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0

# Salin semua file aplikasi, termasuk entrypoint.sh
COPY . .

# Berikan izin eksekusi pada entrypoint script
RUN chmod +x /app/entrypoint.sh

# Buka port yang digunakan oleh Gradio
EXPOSE 7860

# Atur entrypoint untuk container
ENTRYPOINT ["/app/entrypoint.sh"]
