# detection

## Setup 
- To remove the environment do `conda env remove -n deep`
- To create an enviroment `conda create -n deep python=3.9`
- `conda activate deep`
- setup ipykernel `python -m ipykernel install --user --name deep --display-name "Python (deep)"`
- do `pre-commit install`
- `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://qurepypi:38b4dcfcd1ed104ee76b@pypi.qure.ai/simple/ -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html 
- for developmenet do `pip install nbdev ipykernel ipython ipywidgets pytest setuptools twine jupyter clearml lightning mdutils tensorboard`

## tests

- run tests for model predictions `pytest -m "model_test"`
- run all tests other than model predictions `pytest -m "not model_test"`
- run all tests `pytest`
- Read more about pytest [here](https://realpython.com/pytest-python-testing/)


## Resources 
- https://mmcv.readthedocs.io/en/latest/get_started/installation.html
