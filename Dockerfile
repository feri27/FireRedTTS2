FROM nvcr.io/nvidia/tensorrt:23.12-py3
ENV NVIDIA_DISABLE_REQUIRE=1

# Instalasi dependensi sistem yang diperlukan: git dan git-lfs
RUN apt-get update && \
    build-essential \
    python3-dev python3.10-dev \
    apt-get install -y git git-lfs && \
    ffmpeg \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# Salin semua file aplikasi, termasuk entrypoint.sh
COPY . .

# Berikan izin eksekusi pada entrypoint script
RUN chmod +x /app/entrypoint.sh

# Buka port yang digunakan oleh Gradio
EXPOSE 7860

# Atur entrypoint untuk container
ENTRYPOINT ["/app/entrypoint.sh"]
