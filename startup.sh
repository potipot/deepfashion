#!/bin/bash

sudo apt-get update
sudo apt install -y gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ unzip ninja-build

# create user
myusername=ppotrykus
adduser --disabled-password --shell /bin/bash --gecos "" $myusername
usermod -aG sudo $myusername

# copy ssh keys
cp -R ~/.ssh /home/$myusername/.
sudo chown -R $myusername /home/$myusername/.ssh

# set password
usermod --password $(echo nvidia | openssl passwd -1 -stdin) $myusername

# install anaconda
curl https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -o anaconda.sh
bash anaconda.sh -b -p /opt/anaconda3
sudo -Hu $myusername bash -c 'source /opt/anaconda3/bin/activate && conda init'
# source .bashrc when using ssh
printf "if [ -f ~/.bashrc ]; then\n    . ~/.bashrc\nfi\n" >> /home/$myusername/.profile

# clone repo
cd /home/$myusername/
git clone https://github.com/potipot/deepfashion --recurse-submodules

# download datasets
cd deepfashion/datasets
wget https://deepfashion2.blob.core.windows.net/deepfashion2/train.zip
wget https://deepfashion2.blob.core.windows.net/deepfashion2/validation.zip

#unzip datasets/validation.zip
#unzip datasets/train.zip

# create conda env
cd ../deepfashion
conda env create -n deepfashion -f environment.yml

# install nvidia apex
conda activate deepfashion
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex#egg=apex
pip install --upgrade git+https://github.com/potipot/icevision.git@confusion_matrix
pip install --upgrade git+https://github.com/potipot/pytorch-lightning@feature/5555_add_timm_support
sudo chown -R $myusername /home/$myusername/deepfashion
