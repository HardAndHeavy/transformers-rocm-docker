# Transformers on GPU from Radeon in docker
Basic configuration of the docker container [hardandheavy/transformers-rocm](https://hub.docker.com/repository/docker/hardandheavy/transformers-rocm/general) for working with [transformer models](https://huggingface.co) on GPU from Radeon.

### Requirements
* Ubuntu
* make
* Docker
* ROCm

### Install ROCm
Taken from [quick-start install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html):
```
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb

# If an error occurs, you must run:
sudo chown -Rv _apt:root /var/cache/apt/archives/partial/
sudo chmod -Rv 700 /var/cache/apt/archives/partial/

sudo apt update
sudo apt install amdgpu-dkms
sudo apt install rocm-hip-libraries
sudo reboot
```

### Testing
Tested on AMD RadeonRX 7900 XTX.
```
make bash
make test
```
