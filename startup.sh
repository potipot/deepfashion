pip install gpustat gdown

export DF2PW=<some_password>
mkdir datasets && cd datasets
gdown https://drive.google.com/uc?id=1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK  # train
gdown https://drive.google.com/uc?id=1O45YqhREBOoLudjA06HcTehcEebR0o9y  # validation
unzip -P $DF2PW validation.zip
unzip -P $DF2PW train.zip

cd ~
git clone https://github.com/NVIDIA/apex
git clone https://github.com/potipot/deepfashion --recurse-submodules
cd deepfashion
conda env create -n deepfashion -f environment.yml