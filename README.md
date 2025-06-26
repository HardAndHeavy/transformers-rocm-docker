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
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
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
