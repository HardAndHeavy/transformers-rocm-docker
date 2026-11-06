# ROCm runtime layout and installation sources:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/rocm_runtime.Dockerfile
# https://github.com/ROCm/TheRock/blob/main/RELEASES.md
# Pin both the release tag and the manifest digest for reproducible builds.
FROM rocm/dev-ubuntu-26.04:10.1.0-full@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce

ARG ROCM_VERSION=10.1.0
ARG ROCM_PYTHON_INDEX_URL=https://stable.repo.amd.com/rocm/whl-next/
ARG PYTORCH_DEVICE=device-all
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0a0
ARG TORCHAUDIO_VERSION=2.11.0.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Ubuntu 26.04 provides Python 3.14, which is supported by the ROCm 10.1
# PyTorch 2.14 wheels. A venv avoids changing distribution-managed Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        make \
        python3.14 \
        python3.14-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3.14 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Keep the ROCm wheels as a matched set. The device extra installs the GPU
# kernels and the matching Triton/ROCm Python dependencies automatically.
# For a smaller single-GPU image use, for example:
#   --build-arg PYTORCH_DEVICE=device-gfx1201
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --index-url "${ROCM_PYTHON_INDEX_URL}" \
        "torch[${PYTORCH_DEVICE}]==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
        "torchvision[${PYTORCH_DEVICE}]==${TORCHVISION_VERSION}+rocm${ROCM_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}+rocm${ROCM_VERSION}" \
    && python -m pip install --no-cache-dir \
        "transformers>=5.12.1" \
        accelerate \
        peft \
        sentencepiece \
        scipy \
        protobuf

# Fail during the build if a CPU-only or otherwise incompatible torch was
# resolved. GPU access itself is checked by `make test` on the target host.
RUN python - <<'PY'
import torch
import transformers

assert torch.version.hip is not None, "A ROCm-enabled PyTorch wheel is required"
assert torch.__version__.startswith("2.14.0+rocm10.1.0"), torch.__version__
print(f"Python/PyTorch: {torch.__version__}; HIP: {torch.version.hip}")
print(f"Transformers: {transformers.__version__}")
PY

WORKDIR /app
CMD ["bash"]
