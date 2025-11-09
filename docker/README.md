# Docker Configuration for Hebrew Idiom Detection

This directory contains Docker configurations for reproducible development and training environments.

## Overview

The Docker setup provides:
- **PyTorch 2.0.1** with CUDA 11.7 support for GPU training
- **Python 3.10** environment
- All project dependencies pre-installed
- Jupyter notebook support
- TensorBoard integration

## Files

- `Dockerfile` - Main Docker image definition
- `../docker-compose.yml` - Docker Compose configuration for easy orchestration
- `../.dockerignore` - Files excluded from Docker build context

## Prerequisites

### Local Development
- Docker Desktop installed (https://www.docker.com/products/docker-desktop)
- For GPU support: NVIDIA Docker runtime (Linux only)

### VAST.ai Deployment
- VAST.ai account with credit
- No local Docker installation needed

## Building the Image

### Option 1: Using Docker Compose (Recommended)

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up -d

# Access the container
docker-compose exec hebrew-idiom-detection bash

# Stop the container
docker-compose down
```

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t hebrew-idiom-detection:latest -f docker/Dockerfile .

# Run the container
docker run -it --name hebrew-idiom \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/experiments:/workspace/experiments \
  -v $(pwd)/models:/workspace/models \
  -p 8888:8888 \
  hebrew-idiom-detection:latest
```

### With GPU Support (NVIDIA GPUs on Linux)

```bash
# Using Docker Compose (uncomment GPU section in docker-compose.yml first)
docker-compose up -d

# Using Docker directly
docker run -it --gpus all \
  --name hebrew-idiom \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/src:/workspace/src \
  -v $(pwd)/experiments:/workspace/experiments \
  -v $(pwd)/models:/workspace/models \
  -p 8888:8888 \
  hebrew-idiom-detection:latest
```

## Running Jupyter Notebook

### Method 1: Start Jupyter in running container

```bash
# Access container
docker-compose exec hebrew-idiom-detection bash

# Start Jupyter inside container
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then access at: `http://localhost:8888`

### Method 2: Modify docker-compose.yml

Uncomment the Jupyter command line in `docker-compose.yml`:

```yaml
command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then start with:
```bash
docker-compose up -d
```

## Deploying to VAST.ai

### Step 1: Build and Push to Docker Hub (Optional)

```bash
# Login to Docker Hub
docker login

# Tag the image
docker tag hebrew-idiom-detection:latest yourusername/hebrew-idiom-detection:latest

# Push to Docker Hub
docker push yourusername/hebrew-idiom-detection:latest
```

### Step 2: Launch on VAST.ai

1. Go to https://vast.ai and search for instances
2. Filter by:
   - GPU: RTX 3090, RTX 4090, or A5000
   - VRAM: >= 24GB
   - Disk space: >= 50GB
3. Click "RENT" on desired instance
4. Configure:
   - **Image:** `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` (or your Docker Hub image)
   - **On-start script:**
     ```bash
     cd /workspace
     git clone https://github.com/igornazarenko434/hebrew-idiom-detection.git
     cd hebrew-idiom-detection
     pip install -r requirements.txt
     ```
   - **Ports:** Open port 8888 for Jupyter

### Step 3: Connect to VAST.ai Instance

```bash
# SSH connection (provided by VAST.ai)
ssh -p [PORT] root@[HOST]

# Or use Jupyter (if configured)
# Access at: http://[HOST]:[PORT]
```

### Step 4: Setup on VAST.ai Instance

```bash
# Navigate to workspace
cd /workspace/hebrew-idiom-detection

# Download dataset from Google Drive
gdown [YOUR_GOOGLE_DRIVE_FILE_ID]

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Start training
python src/idiom_experiment.py --config experiments/configs/train_config.yaml
```

## Common Commands

### Container Management

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Start stopped container
docker start hebrew-idiom-detection

# Stop running container
docker stop hebrew-idiom-detection

# Remove container
docker rm hebrew-idiom-detection

# View container logs
docker logs hebrew-idiom-detection
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi hebrew-idiom-detection:latest

# Prune unused images
docker image prune
```

### Data Synchronization

```bash
# Copy files from container to host
docker cp hebrew-idiom-detection:/workspace/experiments/results ./experiments/results

# Copy files from host to container
docker cp ./data/dataset.csv hebrew-idiom-detection:/workspace/data/
```

## Volume Mounts

The following directories are mounted as volumes (changes persist):

- `./data` → `/workspace/data` - Dataset files
- `./src` → `/workspace/src` - Source code
- `./experiments` → `/workspace/experiments` - Experiment configs and results
- `./models` → `/workspace/models` - Model checkpoints
- `./notebooks` → `/workspace/notebooks` - Jupyter notebooks

## Environment Variables

Set in `docker-compose.yml` or pass with `-e` flag:

- `PYTHONUNBUFFERED=1` - Disable Python output buffering
- `CUDA_VISIBLE_DEVICES=0` - GPU device to use

## Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run container as current user
docker run -it --user $(id -u):$(id -g) ...
```

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi

# If fails, install nvidia-docker2
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

```bash
# Limit container memory
docker run -it --memory="16g" ...
```

### Port Already in Use

```bash
# Use different port
docker run -p 8889:8888 ...  # Access at localhost:8889
```

## Best Practices

1. **Code Development**: Edit code on host machine (mounted as volume)
2. **Data Storage**: Keep large datasets in mounted volumes, not in image
3. **Model Checkpoints**: Save to mounted `/workspace/models` directory
4. **Results Backup**: Regularly sync `/workspace/experiments/results` to Google Drive
5. **Clean Up**: Remove unused containers and images regularly

## Notes

- This Docker setup is optimized for CUDA 11.7 GPUs
- For newer GPUs, consider updating CUDA version in Dockerfile
- Mac users: GPU training not supported in Docker (use VAST.ai)
- For production deployment, consider using Kubernetes or similar orchestration

## Support

For issues or questions:
- Check Docker documentation: https://docs.docker.com
- VAST.ai documentation: https://vast.ai/docs/
- Project issues: https://github.com/igornazarenko434/hebrew-idiom-detection/issues
