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


