# BML Project: Practical Bayesian Optimization of Machine Learning Algorithms

This project is an implementation of the following paper : *Practical Bayesian Optimization of Machine Learning Algorithms* by J. Snoek, H. Larochelle & R.P. Adams

## 
We tried to implement Bayesian Optimization with the maximization of the Expected Improvement or the Expected Improvement per Second acquisition functoins. We tested our implementention with 4 different data sets on 2 Machine Learning algorithms: SVM and Ridge Regression. We tried different kernels and compared their performance.


## Running the tests

The project is composed of:
- a python script *functions.py*: it contains the function needed to perform the bayesian optimization for hyperparameters and the classes BayesianOptimizationEI/BayesianOptimizationEIperS.
- a notebook *notebook.ipynb*: it contains all the tests and the figures. We tried with different data sets, we provide the url to download them in the notebook. Be sure to select the data and the algorithm you want to test by running **exclusively** the cell charging the data. Beware, a Run All command will not enable you to test both algorithms with all the data provided.

All files (functions, notebook and datasets) have to be install in the same folder. 
