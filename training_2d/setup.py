import os

from setuptools import find_packages, setup

README = open(os.path.join(os.path.dirname(__file__), "README.md")).read()

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

version = open("VERSION").read().strip()
requirements = open("requirements.txt").read().split("\n")

setup(
    name="cxr_training",
    version=version,
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    description="Refactored cxr training repo",
    long_description=README,
    url="http://qure.ai",
    author="Qure AI",
    author_email="support@qure.ai",
    install_requires=requirements,
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
