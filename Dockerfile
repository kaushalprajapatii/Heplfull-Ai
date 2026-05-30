# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (build-essential, git, wget, and standard runtimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user home app folder
WORKDIR $HOME/app

# Copy the application code into the container with correct ownership
COPY --chown=user . $HOME/app

# Pre-download all required AI models during the build phase so they are baked into the image
RUN mkdir -p $HOME/app/backend/models && \
    mkdir -p $HOME/.u2net && \
    wget -q -O $HOME/app/backend/models/depth_anything.onnx "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" && \
    wget -q -O $HOME/app/backend/models/gfpgan.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx" && \
    wget -q -O $HOME/app/backend/models/realesrgan.onnx "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/Real-ESRGAN_x2plus.onnx" && \
    wget -q -O $HOME/.u2net/u2net.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" && \
    wget -q -O $HOME/.u2net/birefnet-general.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/birefnet-general.onnx"

# Expose port 7860, which Hugging Face expects
EXPOSE 7860

# Run the Streamlit application on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
