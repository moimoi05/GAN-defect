# ==============================================================================
# DEFECT-GAN TRAINING & DEPLOYMENT DOCKERFILE
# ==============================================================================
# Base Image: Official PyTorch with CUDA 12.1 runtime and cuDNN 8
# Compatible with NVIDIA Drivers >= 530.30.02 (Linux)
# Fallback Note: For older NVIDIA drivers (>= 520), switch to pytorch:2.1.2-cuda11.8-cudnn8-runtime
#
# Build command:
#   docker build -t defectgan:latest .
#
# GPU Training command:
#   docker run --gpus all --ipc=host \
#     -v /path/to/GAN_Project:/workspace/data:ro \
#     -v /path/to/outputs:/workspace/outputs \
#     defectgan:latest train --config configs/defectgan_1024.yaml
#
# CPU Smoke test command:
#   docker run --rm \
#     -v /path/to/GAN_Project:/workspace/data:ro \
#     -v /path/to/outputs:/workspace/outputs \
#     defectgan:latest smoke-test
# ==============================================================================

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="RTC R&D Computer Vision Team"
LABEL project="Defect-GAN Industrial Scratch Synthesis"
LABEL description="Bidirectional RGB Defect Synthesis & Restoration with Layer-Wise Composition"

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKSPACE=/workspace
ENV DATA_DIR=/workspace/data
ENV OUTPUT_DIR=/workspace/outputs

WORKDIR ${WORKSPACE}

# Install essential system runtime libraries (OpenCV, PIL support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, training manifests, and scripts
# NOTE: Datasets (F971, S942, patches) are NOT copied into the image to ensure
# minimal image size and data privacy. They are mounted at runtime.
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY training/manifests/ training/manifests/
COPY tests/ tests/
COPY docker/entrypoint.sh docker/entrypoint.sh

# Ensure entrypoint is executable
RUN chmod +x docker/entrypoint.sh

# Create mount point directories
RUN mkdir -p /workspace/data /workspace/outputs

# Set entrypoint
ENTRYPOINT ["docker/entrypoint.sh"]

# Default command displays usage
CMD ["--help"]
