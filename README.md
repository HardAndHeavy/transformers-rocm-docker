# Transformers on GPU AMD Radeon in Docker

Basic configuration of the docker container [hardandheavy/transformers-rocm](https://hub.docker.com/r/hardandheavy/transformers-rocm) for working with [transformer models](https://huggingface.co) on GPU from Radeon.

### Requirements

* Ubuntu
* make
* Docker
* ROCm

### Install ROCm

```bash
sudo apt install -y rocm
```

### Testing

Tested on AMD Radeon AI PRO R9700 Creator 32 Gb.

```bash
make bash
make test
```

### License

The original code and documentation in this repository are licensed under
the [Apache License 2.0](LICENSE).
Third-party software and models remain subject to their respective licenses.
