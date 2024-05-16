# Car Parts Perspective Scorer
## Overview

This repository contains code for training a Convolutional Neural Network (CNN) to score the perspective of car parts. The model is trained to score different viewpoints (score_hood, score_backdoor_left) perspectives of car parts.

## Dataset

The dataset used for training consists of simulated and labeled images of car parts captured from various angles. It includes images of car fronts, sides, rears, and tops. The dataset is divided into training, validation sets to evaluate the performance of the model.

## Training

To train the model, run the `train.py` script. You can customize the hyperparameters such as batch size, learning rate, and number of epochs in the corresponding yaml file (cfg/train.yaml). Make sure to add your training data and adjust the paths in the yaml files accordingly. During training, the model's performance on the validation set is monitored to prevent overfitting.
## Evaluation

After training, the trained model can be compared with the `compare_results.py` file.

## Inference

Once trained and evaluated, the model can be used for inference on new images to classify the perspective of car parts. Use the `infer.py` script to classify individual images.
## Requirements

Make sure you have the following dependencies installed:

    hydra
    PIL
    Python 3.x
    Pytorch
    NumPy
    Matplotlib

License

This project is licensed under the MIT License - see the LICENSE file for details.