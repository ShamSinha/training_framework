# qct_nodule_detection
nodule detection in 3D Lung ct scan. works for other 3D ct scans tasks too.

## Setup 
- To remove the environment do `conda env remove -n qct_deep`
- To create an enviroment `conda create -n qct_deep python=3.9`
- `conda activate qct_deep`
- setup ipykernel `python -m ipykernel install --user --name qct_deep --display-name "Python (qct_deep)"`
- do `pre-commit install`
- `pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://qurepypi:38b4dcfcd1ed104ee76b@pypi.qure.ai/simple/ -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html 
- for developmenet do `pip install nbdev ipykernel ipython ipywidgets pytest setuptools twine jupyter clearml lightning mdutils tensorboard`

## qure custom-dependencies
create a folder called `deps`. clone the following repos and do `pip install -e .` on each 
- qct_utils
- medct


# Updates
- [Generate cache for metrics](https://github.com/qureai/qct_nodule_detection/issues/37) :smiling_imp:
- [Generate lung masks on a folder of sitk data](https://github.com/qureai/qct_data/issues/117#issuecomment-1316397366) :rocket:
- [Visualize the outputs on scan](https://github.com/qureai/qct_nodule_detection/issues/73) :heart_eyes:
- [setup clearml](https://wiki.qure.ai/doc/clearml-experiment-manager-gzty4tLY6F) :rocket:
- [Generate metrics by meta-data initial code](https://github.com/qureai/qct_nodule_detection/issues/131) :rocket:

## tests

- run tests for model predictions `pytest -m "model_test"`
- run all tests other than model predictions `pytest -m "not model_test"`
- run all tests `pytest`
- Read more about pytest [here](https://realpython.com/pytest-python-testing/)


## Resources 
- https://mmcv.readthedocs.io/en/latest/get_started/installation.html