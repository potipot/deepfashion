sudo apt-get install unzip ninja-build g++ -y
pip install gpustat gdown

export DF2PW=<some_password>

gdown https://drive.google.com/uc?id=1lQZOIkO-9L0QJuk_w1K8-tRuyno-KvLK  # train
gdown https://drive.google.com/uc?id=1O45YqhREBOoLudjA06HcTehcEebR0o9y  # validation
unzip -P $DF2PW validation.zip
unzip -P $DF2PW train.zip


cd ~ && git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ~ && git clone https://github.com/potipot/deepfashion --recurse-submodules
cd deepfashion
conda env create -n deepfashion -f environment.yml