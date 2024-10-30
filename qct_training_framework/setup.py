#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="Deep-learning training Framework for rapid experimentation for classification and segmentation",
    author="Vikash Challa, Souvik Mandal",
    author_email="",
    url="https://github.com/qureai/qct_training_framework",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
)
