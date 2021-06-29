#!/usr/bin/env bash

git clone https://github.com/ym-han/for_gvm_gpt.git

sudo apt-get update --yes

sudo apt-get install gcc python3-dev python3-setuptools
sudo pip3 uninstall crcmod
sudo pip3 install --no-cache-dir -U crcmod

sudo apt install zstd --yes
gsutil cp gs://coref_gpt/model_zstd/slim_chkpt.tar.zstd . 
tar -I zstd -xf slim_chkpt.tar.zstd


sudo apt-get install python3-venv --yes
python3 -m venv env
source env/bin/activate

pip install cloud-tpu-client
pip install fastcore
pip install tqdm

git clone https://github.com/kingoflolz/mesh-transformer-jax.git
pip install -r mesh-transformer-jax/requirements.txt
pip install mesh-transformer-jax/ jax==0.2.12

pip uninstall jaxlib --yes
pip install jaxlib==0.1.67
# will error otherwise

# pip install --upgrade fabric dataclasses requests 
