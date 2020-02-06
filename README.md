# Encoder-Decoder-attention-cn

## Installation 

conda create -n tfTrain python=3.6

pip install tf-nightly-gpu

pip install -r requirements.txt

## Training

set args.evaluate=False, run:

python train_attention.py 

## Test

set args.evaluate=True, run:

python train_attention.py

