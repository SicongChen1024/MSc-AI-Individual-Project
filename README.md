# MSc-AI-Individual-Project
This repositry contains the implementations of my MSc AI Individual project: Defending Against Adversarial Attacks on Perceptual Ad-blockers

## Installation
```
pip install -r requirements.txt
```
## Adversarial Training
Change the directory to Adversarial_training and run
```
python main.py
```
to derive a sample SqueezeNet ad-classifier with adversarial training.
## Jacobian Regularization
Change the directory to Jacobian_regularization and run
```
python main.py
```
to derive a sample SqueezeNet ad-classifier with Jacobian regularization.
## Ensemble Ad-blockers
Change the directory to Multiple_models and run
```
python main.py
```
to derive the results of conducting randomized and ensemble&randomized algorithms.
## Acknowledgement
Thanks to 

* [Florian Tramer](https://github.com/ftramer/ad-versarial) (MIT License Copyright (c) 2018 ftramer) 
* [Harry](https://github.com/Harry24k/adversarial-attacks-pytorch) (MIT License Copyright (c) 2020 Harry Kim)
* [Facebook Research](https://github.com/facebookresearch/jacobian_regularizer) (MIT License Copyright (c) Facebook, Inc. and its affiliates.)
* [YI-LIN SUNG](https://github.com/louis2889184/pytorch-adversarial-training)

since part of codes in this repositry are borrowed or modified (state explictly in the scripts) from their implementations.
