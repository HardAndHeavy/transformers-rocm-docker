# Transformers on GPU AMD Radeon in Docker
Basic configuration of the docker container [hardandheavy/transformers-rocm](https://hub.docker.com/repository/docker/hardandheavy/transformers-rocm/general) for working with [transformer models](https://huggingface.co) on GPU from Radeon.

### Requirements
* Ubuntu
* make
* Docker
* ROCm

### Install ROCm
Taken from [quick-start install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html):
```bash
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.2.1/ubuntu/noble/amdgpu-install_6.2.60201-1_all.deb
sudo apt install ./amdgpu-install_6.2.60201-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm
sudo reboot
```

Install rocminfo:
```bash
sudo apt install rocminfo
# It is necessary to change the PATH in the /etc/environment file. Add ":/opt/rocm/bin"
```

In case of an error `gpu_process_host` took the advice from [SamuelMarks](https://github.com/signalapp/Signal-Desktop/issues/6855#issuecomment-2118305464):
```bash
# Deleting everything that contains "va-driver"
sudo apt remove mesa-va-drivers
sudo apt remove mesa-amdgpu-va-drivers:amd64
```

### Testing
Tested on AMD Radeon RX 7900 XTX.
```bash
make bash
make test
```
