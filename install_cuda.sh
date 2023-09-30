#!/bin/bash
# # 为系统安装cuda 11.0
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# 3090ti 不能支持11.0，需要安装高版本的cuda
wget -c https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget -c https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.2-535.104.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
