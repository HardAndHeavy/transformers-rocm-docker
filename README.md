# Transformers + PyTorch + ROCm in Docker

A reproducible Linux/amd64 development and inference image for running
[Hugging Face Transformers](https://huggingface.co/docs/transformers) on AMD
Radeon GPUs. Published as
[`hardandheavy/transformers-rocm`](https://hub.docker.com/r/hardandheavy/transformers-rocm).

The repository includes an interactive LFM3 chat, a real model-generation
smoke test, and a fast ROCm runtime test that executes a BF16 workload on the
GPU.

## Version 3.0.0 software stack

See the [changelog](CHANGELOG.md) for the release highlights. The image version
is independent of the ROCm and PyTorch versions.

| Component | Version |
| --- | --- |
| Base OS | Ubuntu 26.04 |
| ROCm | 10.1.0 |
| Python | 3.14 |
| PyTorch | 2.13.0+rocm10.1.0 |
| torchvision | 0.28.0+rocm10.1.0 |
| torchaudio | 2.11.0.3+rocm10.1.0 |
| Transformers | 5.22.1 |
| Accelerate | 1.15.0 |
| PEFT | 0.21.2 |

The image contains ROCm kernels for all GPU architectures provided by the
locked `device-all` wheels. See the dependency section below if you need a
smaller, architecture-specific build.

## Requirements

- Linux x86_64 with an AMD GPU supported by ROCm 10.1.
- A compatible AMDGPU kernel driver exposing `/dev/kfd` and `/dev/dri`.
- Docker Engine.
- Git and `make` for the repository commands.
- [`uv`](https://docs.astral.sh/uv/) only when intentionally regenerating the
  dependency locks.

The complete ROCm user-space stack is installed inside the image; installing
it on the host is not required. For host driver installation, follow the
[AMD ROCm Linux installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).

## Quick start with Docker Hub

Start a shell and pass the AMD devices through to the container:

```bash
mkdir -p huggingface
docker run --rm -it \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  -v "$PWD/huggingface:/root/.cache/huggingface" \
  hardandheavy/transformers-rocm:3.0.0
```

`latest` points to the same release after publication, but the numbered tag is
recommended for reproducible deployments. Mount your own project at `/app` to
work with it inside the container, for example `-v "$PWD:/app"`.

## Build and test from source

```bash
git clone https://github.com/HardAndHeavy/transformers-rocm-docker.git
cd transformers-rocm-docker

make check                 # Dockerfile static checks and Python syntax
make build tag=3.0.0       # creates transformers-rocm:3.0.0 and :latest
make smoke-test tag=3.0.0  # imports, GPU discovery, BF16 GPU computation
make test tag=3.0.0        # loads LFM3 onto the GPU and generates a response
```

`3.0.0` is the default `tag`, so the explicit `tag=3.0.0` arguments can be
omitted. Run `make bash` for an interactive shell in the locally built image.
The repository is mounted at `/app`, and models are cached in `./huggingface`.

The full test downloads about 3.5 GB on its first run. Both test targets
require access to a real AMD GPU; `docker build` verifies the installed stack
but cannot verify host GPU access.

## Interactive LFM3 chat

After building the image:

```bash
make bash
python test/lfm3.py
```

The model is loaded once, and conversation history is preserved between
messages. Use `/clear` to reset the history, `/help` to list commands, and
`/exit`, `/quit`, or `Ctrl+D` to leave the chat.

To override the model or generation limit in either interactive or automatic
mode:

```bash
MODEL_ID=LiquidAI/LFM3-1.7B MAX_NEW_TOKENS=256 make test
# Or inside `make bash`:
python test/lfm3.py --model-id LiquidAI/LFM3-1.7B --max-new-tokens 256
```

Only use `trust_remote_code` with model repositories you trust. The included
example enables it for its configured model.

## Reproducible builds and dependencies

The base image is pinned by digest, direct APT packages have exact Ubuntu
versions, and every Python package—including transitive dependencies and the
installer/build backend—is locked with SHA256 hashes. Installation uses
`pip --require-hashes`; normal builds never select newer dependency versions.

OCI labels record the image version, Git revision, build timestamp, source,
license, and base-image digest. `BUILD_DATE` and `GIT_REVISION` can be
overridden by release automation.

Run this command **only for an intentional dependency update**, review both
lock files, and then rebuild and run both GPU tests:

```bash
make lock
```

The locks target CPython 3.14 on Linux x86_64. The default `device-all` extras
provide broad GPU compatibility but produce a large download and image: the
validated 3.0.0 image occupies **59.5 GiB uncompressed** locally, and a build
needs additional temporary space for exporting and unpacking layers. For a
smaller single-architecture image, replace both `device-all` extras in
`requirements.in` with the appropriate device extra (for example,
`device-gfx1201` for the Radeon AI PRO R9700), regenerate the locks, and test
on that GPU.

This ROCm 10.1 image uses the stable PyTorch 2.13/torchvision 0.28 wheel set.
The torchvision 0.29.0a0 wheel published for PyTorch 2.14 is a prerelease and
is deliberately excluded; see
[AMD's compatibility issue](https://github.com/ROCm/TheRock/issues/8802).
AMD's **stable** aggregate Python index is
`https://stable.repo.amd.com/rocm/whl-next/`: despite its path, it is distinct
from `nightly.repo.amd.com`. AMD does not currently publish a versioned
aggregate index for this release, so the reviewed lock files and mandatory
hashes make builds independent of subsequent index changes.

## Health checks

The Dockerfile healthcheck verifies Python imports, a ROCm-enabled PyTorch
build, and GPU visibility. It does not download a model or perform generation.
Containers must receive `/dev/kfd` and `/dev/dri`, as in all repository run
commands.

Kubernetes does not use the Dockerfile `HEALTHCHECK`; configure an equivalent
readiness probe and use the model test for release validation:

```yaml
readinessProbe:
  exec:
    command:
      - python
      - -c
      - "import torch, transformers; assert torch.version.hip is not None; assert torch.cuda.is_available()"
  periodSeconds: 30
  timeoutSeconds: 15
  failureThreshold: 3
```

## Publishing a release

After logging in with `docker login` and passing all checks on an AMD GPU host:

```bash
make publish tag=3.0.0
```

This pushes both `hardandheavy/transformers-rocm:3.0.0` and `:latest`.
`make publish tag=latest` is deliberately rejected to avoid an unversioned
release.

## License

The original code and documentation in this repository are licensed under the
[Apache License 2.0](LICENSE). Third-party software and models remain subject
to their respective licenses.
