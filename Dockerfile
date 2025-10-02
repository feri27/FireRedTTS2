# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
# Installing PyTorch separately for CUDA 11.8 support
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy the rest of the application's code into the container
# Assuming your project structure is flat or fireredtts2 is a subdir
COPY . .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable for pretrained model directory
# You should mount your pretrained models to this directory
# e.g., using docker run -v /path/to/your/models:/app/pretrained_models
ENV PRETRAINED_DIR=/app/pretrained_models

# Run gradio_demo.py when the container launches
# The application will be available at http://<container-ip>:7860
CMD ["python", "gradio_demo.py", "--pretrained-dir", "/app/pretrained_models", "--server-name", "0.0.0.0"]
