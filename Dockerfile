# https://github.com/ROCm/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
# https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/pai/rocm-ci-pipeline-env.Dockerfile
FROM ubuntu:24.04

ARG ROCM_VERSION=6.2.4
ARG APT_PREF='Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600'
RUN echo "$APT_PREF" > /etc/apt/preferences.d/rocm-pin-600

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    ninja-build \
    build-essential \
    pkg-config \
    curl \
    wget \
    git \
    make

RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ noble main" | tee /etc/apt/sources.list.d/rocm.list && \
    printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/${ROCM_VERSION}/ubuntu noble main" | tee /etc/apt/sources.list.d/amdgpu.list
RUN apt-get update && apt-get install -y \
    rocm-dev \
    rocm-libs

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Install Conda
ENV PATH /opt/miniconda/bin:${PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    conda init bash && \
    conda config --set auto_activate_base false && \
    conda update --all && \
    rm ~/miniconda.sh && conda clean -ya
ENV PYTHON_VERSION=3.11
RUN conda install python=${PYTHON_VERSION} pip

ENV INDEX_URL=https://download.pytorch.org/whl/rocm6.2
RUN pip install torch torchvision torchaudio --index-url ${INDEX_URL}
RUN pip install transformers \
    peft \
    sentencepiece \
    scipy \
    protobuf --extra-index-url ${INDEX_URL}
