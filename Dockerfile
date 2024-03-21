# https://github.com/ROCm/ROCm-docker/blob/master/dev/Dockerfile-ubuntu-22.04-complete
# https://github.com/microsoft/onnxruntime/blob/main/tools/ci_build/github/pai/rocm-ci-pipeline-env.Dockerfile
FROM ubuntu:22.04

ARG ROCM_VERSION=6.0.2
ARG AMDGPU_VERSION=${ROCM_VERSION}
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
    printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main" | tee /etc/apt/sources.list.d/rocm.list && \
    printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main" | tee /etc/apt/sources.list.d/amdgpu.list
RUN apt-get update && apt-get install -y \
    rocm-dev \
    rocm-libs

# Install Conda
ENV PATH /opt/miniconda/bin:${PATH}
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --no-check-certificate && /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    conda init bash && \
    conda config --set auto_activate_base false && \
    conda update --all && \
    rm ~/miniconda.sh && conda clean -ya
ENV PYTHON_VERSION=3.11
RUN conda install python=${PYTHON_VERSION} pip

# https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/install-pytorch.html
RUN wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.0/torch-2.1.1%2Brocm6.0-cp311-cp311-linux_x86_64.whl
RUN pip3 install --force-reinstall torch-2.1.1+rocm6.0-cp311-cp311-linux_x86_64.whl

RUN pip install transformers \
    peft \
    sentencepiece \
    scipy

# https://github.com/agrocylo/bitsandbytes-rocm
# or
# https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6
ENV PYTORCH_ROCM_ARCH=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100,gfx1101,gfx940,gfx941,gfx942
ENV BITSANDBYTES_VERSION=62353b0200b8557026c176e74ac48b84b953a854
RUN git clone https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6 /bitsandbytes && \
    cd /bitsandbytes && \
    git checkout ${BITSANDBYTES_VERSION} && \
    make hip ROCM_TARGET=${PYTORCH_ROCM_ARCH} ROCM_HOME=/opt/rocm/ && \
    pip install . --extra-index-url https://download.pytorch.org/whl/nightly
