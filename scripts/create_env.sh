cd ~/deepfashion
conda env create -n deepfashion -f environment.yml

# install nvidia apex
conda activate deepfashion
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex#egg=apex
pip install --upgrade git+https://github.com/potipot/icevision.git@confusion_matrix
pip install --upgrade git+https://github.com/potipot/pytorch-lightning@feature/5555_add_timm_support