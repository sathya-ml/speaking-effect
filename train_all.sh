#!/bin/bash

PYTHONPATH=./ python3 classify/train_fer_features_cnn.py -c ./config/
PYTHONPATH=./ python3 classify/train_fer_features_cnn_lip.py -c ./config/
PYTHONPATH=./ python3 classify/train_fer_features_rnn.py -c ./config/
PYTHONPATH=./ python3 classify/train_fer_features_rnn_lip.py -c ./config/
