sudo apt install -y unzip
cd ~/deepfashion/datasets

#wget https://deepfashion2.blob.core.windows.net/deepfashion2/train.zip
#wget https://deepfashion2.blob.core.windows.net/deepfashion2/validation.zip
curl https://nextcloud.toucan.systems/index.php/s/azdn3YDwxCGmMtK/download -OJ # train
curl https://nextcloud.toucan.systems/index.php/s/NgCgyLLmEieorrg/download -OJ # validation

unzip datasets/validation.zip
unzip datasets/train.zip