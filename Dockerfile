# ROCm runtime layout and installation sources:
# https://github.com/ROCm/TheRock/blob/main/dockerfiles/rocm_runtime.Dockerfile
# https://github.com/ROCm/TheRock/blob/main/RELEASES.md
# Pin both the release tag and the manifest digest for reproducible builds.
FROM rocm/dev-ubuntu-26.04:10.1.0-full@sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce

ARG IMAGE_VERSION=3.0.0
ARG IMAGE_REVISION=unknown
ARG IMAGE_CREATED=1970-01-01T00:00:00Z
LABEL org.opencontainers.image.title="transformers-rocm" \
      org.opencontainers.image.description="Transformers, PyTorch and ROCm for AMD Radeon GPUs" \
      org.opencontainers.image.source="https://github.com/HardAndHeavy/transformers-rocm-docker" \
      org.opencontainers.image.url="https://hub.docker.com/r/hardandheavy/transformers-rocm" \
      org.opencontainers.image.documentation="https://github.com/HardAndHeavy/transformers-rocm-docker#readme" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.base.name="rocm/dev-ubuntu-26.04:10.1.0-full" \
      org.opencontainers.image.base.digest="sha256:4f5ed1bf6532a4b9920400401b0ad0f706af356dae61e40afefbfd6c8073f0ce"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Versions come from the Ubuntu 26.04 resolute/{updates,security} repositories.
# Update these pins deliberately when moving the digested base image forward.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=20260816ubuntu0.26.04.2 \
        curl=8.18.0-1ubuntu2.8 \
        git=1:2.53.0-1ubuntu1.1 \
        git-lfs=3.7.1-1 \
        make=4.4.1-3 \
        python3.14=3.14.4-1ubuntu0.2 \
        python3.14-venv=3.14.4-1ubuntu0.2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.14 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# The locks cover Python 3.14 on Linux x86_64 and every transitive dependency.
# Hash checking prevents changed or substituted artifacts from being installed.
# The second command disables build isolation so the ROCm sdist uses only the
# pinned setuptools/wheel installed by the bootstrap lock.
COPY requirements-bootstrap.lock requirements.lock /opt/requirements/
RUN python -m pip install --no-cache-dir --require-hashes \
        --requirement /opt/requirements/requirements-bootstrap.lock \
    && python -m pip install --no-cache-dir --require-hashes --no-build-isolation \
        --requirement /opt/requirements/requirements.lock \
    && python -m pip check

# Fail during the build for incompatible wheels. GPU access itself is checked
# by the runtime healthcheck and by `make test` on the target host.
RUN python - <<'PY'
import torch
import torchaudio
import torchvision
import transformers

expected_versions = {
    "torch": "2.13.0+rocm10.1.0",
    "torchaudio": "2.11.0.3+rocm10.1.0",
    "torchvision": "0.28.0+rocm10.1.0",
    "transformers": "5.22.1",
}
for module in (torch, torchaudio, torchvision, transformers):
    name = module.__name__
    assert module.__version__ == expected_versions[name], (name, module.__version__)
assert torch.version.hip is not None, "A ROCm-enabled PyTorch wheel is required"
print(f"PyTorch: {torch.__version__}; HIP: {torch.version.hip}")
print(f"torchvision: {torchvision.__version__}; torchaudio: {torchaudio.__version__}")
print(f"Transformers: {transformers.__version__}")
PY

# This is a GPU runtime image, not an HTTP service. The check validates imports
# and GPU visibility without downloading a model or running generation.
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python -c "import torch, transformers; assert torch.version.hip is not None; assert torch.cuda.is_available(), 'ROCm GPU is not available'"

WORKDIR /app
CMD ["bash"]
