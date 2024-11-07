#!/bin/bash
conda create -n qtrain python=3.10
eval "$(conda shell.bash hook)"
conda activate qtrain
if [ -d "/home/users/$USER/qxr_training" ]; then
    # If it exists, change to the 'qxr_training' directory
    cd /home/users/$USER/qxr_training
fi
pip install -e .
