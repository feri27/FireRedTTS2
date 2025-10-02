# Gunakan image python slim sebagai dasar
FROM python:3.10-slim

# Instalasi dependensi sistem yang diperlukan: git dan git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Salin semua file aplikasi, termasuk entrypoint.sh
COPY . .

# Berikan izin eksekusi pada entrypoint script
RUN chmod +x /app/entrypoint.sh

# Buka port yang digunakan oleh Gradio
EXPOSE 7860

# Atur entrypoint untuk container
ENTRYPOINT ["/app/entrypoint.sh"]
