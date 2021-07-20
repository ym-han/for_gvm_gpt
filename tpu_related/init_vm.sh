#!/usr/bin/env bash
set -e

sudo apt-get update --yes
sudo apt-get upgrade --yes
sudo apt install git-all --yes


sudo apt-get install screen
sudo apt-get install gcc python3-dev python3-setuptools
sudo pip3 uninstall crcmod --yes
sudo pip3 install --no-cache-dir -U crcmod

cd $HOME

if [ ! -d "$HOME/slim_chkpt" ]; then
  sudo apt install zstd --yes
  gsutil cp gs://coref_gpt/model_zstd/slim_chkpt.tar.zstd . 
  tar -I zstd -xf slim_chkpt.tar.zstd
  mv step_383500 slim_chkpt
fi

cd $HOME


if [ ! -d "$HOME/env/" ]; then
  sudo apt-get install python3-venv --yes
  python3.8 -m venv "$HOME/env/"
fi

. $HOME/env/bin/activate

pip3 install --upgrade setuptools

pip3 install cloud-tpu-client
pip3 install fastcore
pip3 install tqdm
pip3 install wandb
pip3 install notifiers

pip3 install smart_open[gcs]
pip3 install rich
pip3 install ujson
pip3 install dill

pip3 install "ray[default]"
pip3 install "ray[serve]"

git clone https://github.com/kingoflolz/mesh-transformer-jax.git
pip3 install -r mesh-transformer-jax/requirements.txt
pip3 install mesh-transformer-jax/ jax==0.2.12

pip3 uninstall jaxlib --yes
pip3 install jaxlib==0.1.67
# will error otherwise

# pip install --upgrade fabric dataclasses requests 
