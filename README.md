# Transformers on GPU AMD Radeon in Docker
Basic configuration of the docker container [hardandheavy/transformers-rocm](https://hub.docker.com/r/hardandheavy/transformers-rocm) for working with [transformer models](https://huggingface.co) on GPU from Radeon.

### Requirements
* Ubuntu
* make
* Docker
* ROCm

### Install ROCm
Taken from [quick-start install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html):
```bash
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
sudo reboot
```

### Testing
Tested on AMD Radeon RX 7900 XTX.
```bash
make bash
make test
```
