# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (build-essential, git, wget, ca-certificates, and standard runtimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create the non-root user and home directory (still as ROOT)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user home app folder
WORKDIR /home/user/app

# Copy the application code into the container
COPY . /home/user/app

# Pre-download all required AI models (as ROOT, guaranteeing absolute write permissions)
RUN mkdir -p /home/user/app/backend/models && \
    mkdir -p /home/user/.u2net && \
    wget -q -O /home/user/app/backend/models/depth_anything.onnx "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx" && \
    wget -q -O /home/user/app/backend/models/gfpgan.onnx "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx" && \
    wget -q -O /home/user/app/backend/models/realesrgan.onnx "https://huggingface.co/tidus2102/Real-ESRGAN/resolve/main/Real-ESRGAN_x2plus.onnx" && \
    wget -q -O /home/user/.u2net/u2net.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" && \
    wget -q -O /home/user/.u2net/BiRefNet-general-epoch_244.onnx "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"

# Recursively give user (1000) full ownership of their files and cached weights
RUN chown -R user:user /home/user

# Switch to the non-root user for execution safety and Hugging Face requirements
USER user

# Expose port 7860, which Hugging Face expects
EXPOSE 7860

# Run the Streamlit application on port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
