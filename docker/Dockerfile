# Hebrew Idiom Detection - Docker Configuration
# Base image: PyTorch with CUDA support for GPU training

# Use official PyTorch image with CUDA 11.8 and Python 3.10
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set maintainer information
LABEL maintainer="igor.nazarenko@post.runi.ac.il"
LABEL description="Docker image for Hebrew Idiom Detection project"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/src \
    /workspace/experiments/configs \
    /workspace/experiments/results \
    /workspace/experiments/logs \
    /workspace/models \
    /workspace/notebooks \
    /workspace/scripts \
    /workspace/tests \
    /workspace/paper/figures \
    /workspace/paper/tables

# Expose Jupyter notebook port
EXPOSE 8888

# Expose TensorBoard port (if needed)
EXPOSE 6006

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: start bash
CMD ["/bin/bash"]

# Alternative: Start Jupyter notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
