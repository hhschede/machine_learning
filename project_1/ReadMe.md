Machine Learning: Project 1 

This ReadMe contains descriptions of the different jupyter notebooks used for project 1 of the machine learning course. We give a brief description of the notebook files and explain where they were implemented in the project.

./data folder:
	This folder contains the data used in this code. There is a train and test set, both of which are in CSV files. The training set contains labeled samples while the test set does not.

./implementations folder:
	In this folder you find all the methods used in the code. 
- helpers.py : Contains basic functions.
- implementation.py: Contains the functions used for linear and logistic regression, along with some plotting functions and data augmentation functions.
- pca.py: Contains the PCA method used for feature reduction.
- plot.py : Contains functions to plot the learning curves of the models.
- preprocessing.py: Contains process_data() and transform_data() methods. These methods perform feature and outlier filtering, data standarization, polynomial building, PCA reduction and others.

testing_implemented_functions.ipynb:
	This jupyter notebook regroups linear and logistic regression. In this notebook we demonstrated that our implemented function works and present different functions that we will be used in other jupyter notebooks.

report_figures.ipynb:
	In this file, there is the code used to cross-validate our model and to generate the figures used in the report. The figures are outputed as .png

run.ipynb:
	This jupyter notebook was used to compute the last optimized model and predict the labels on the test set that will be submitted on kaggle.
