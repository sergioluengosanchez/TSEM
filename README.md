## Introduction
Most real-world problems require dealing with incomplete data. Bayesian networks robustly handle missing values during inference, but learning them from incomplete datasets is not straightforward. Using the structural expectation-maximization algorithm is the most common approach to address this problem, but its main limitation is its highly demanding computational cost. As structural expectation-maximization spends most of its running time performing inference, efficient inference is essential. This can be achieved by bounding the inference complexity of the Bayesian network candidates.  This paper proposes a tractable adaptation of the structural expectation-maximization algorithm that theoretically provides guarantees on its convergence. We perform experiments that support empirically our claims.

## Prerequirements and installing guide

This software has been developed as a Python 2.7.15 package and includes some functionalities in Cython and C++11 (version 5.4.0). Consequently, it is needed a Python environment and internet connectivity to download additional package dependencies. Python software can be downloaded from <https://www.python.org/downloads/>.

We provide the steps for a clean installation in Ubuntu 16.04. This software has not been tried under Windows.

The package also uses the following dependencies. 

|Library     |Version|License|
|------------|-------|-------|
| pandas     |   0.23|  BSD 3|
|  numpy     | 1.14.3|    BSD|
| Cython     | 0.28.2| Apache|
|cloudpickle |  0.5.3|  BSD 3|
|scikit-learn| 0.20.2|  BSD 3|


They can be installed through the following sentence:
sudo pip install "Library"
where Library must be replaced by the library to be installed.

Open the folder where you have saved TSEM project files (e.g., "~/Downloads/TSEM") and compile Cython files running the following commands in the command console:

python2.7 setup_dt.py build_ext --inplace

python2.7 setup_tw.py build_ext --inplace

python2.7 setup_et.py build_ext --inplace

python2.7 setup_cplus.py build_ext --inplace

python2.7 setup_cplus_data.py build_ext --inplace

python2.7 setup_gs.py build_ext --inplace

python2.7 setup_etc.py build_ext --inplace

## Example.py

File "example.py" provides a demo that shows how to use the code to: learn Bayesian networks in the presence of missing values, interpreting the returned models, and performing inference. 
