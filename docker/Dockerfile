# Hebrew Idiom Detection - Docker Configuration
# Base image: PyTorch with CUDA support for GPU training
# Last Updated: 2025-12-08

# Use official PyTorch image with CUDA 11.8 and Python 3.10
# Note: Update to pytorch/pytorch:2.6.0-cuda11.8-cudnn8-runtime when available
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set maintainer information
LABEL maintainer="igor.nazarenko@post.runi.ac.il"
LABEL description="Docker image for Hebrew Idiom Detection project"
LABEL version="1.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies
# Includes: git, curl, wget, vim, build tools, rclone, and Hebrew fonts
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    nano \
    build-essential \
    fonts-dejavu-core \
    fonts-noto \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install rclone for Google Drive sync
RUN curl https://rclone.org/install.sh | bash

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file
COPY ../requirements.txt /workspace/requirements.txt

# Install Python dependencies
# Note: PyTorch is already installed in base image, but requirements.txt specifies >=2.6.0
# If base image has older version, this will upgrade it
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories matching volume structure
RUN mkdir -p /workspace/data \
    /workspace/data/splits \
    /workspace/src \
    /workspace/experiments/configs \
    /workspace/experiments/results \
    /workspace/experiments/results/zero_shot \
    /workspace/experiments/results/full_finetune \
    /workspace/experiments/results/hpo \
    /workspace/experiments/results/evaluation \
    /workspace/experiments/results/optuna_studies \
    /workspace/experiments/results/best_hyperparameters \
    /workspace/experiments/logs \
    /workspace/models \
    /workspace/notebooks \
    /workspace/scripts \
    /workspace/tests \
    /workspace/cache \
    /workspace/cache/huggingface \
    /workspace/config

# Expose Jupyter notebook port
EXPOSE 8888

# Expose TensorBoard port
EXPOSE 6006

# Expose Optuna Dashboard port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command: start bash
CMD ["/bin/bash"]

# Alternative commands (uncomment as needed):
# Start Jupyter notebook:
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
#
# Start TensorBoard:
# CMD ["tensorboard", "--logdir=/workspace/experiments/results", "--host=0.0.0.0", "--port=6006"]
#
# Run training:
# CMD ["python", "/workspace/src/idiom_experiment.py", "--mode", "full_finetune", "--task", "cls", "--device", "cuda"]
