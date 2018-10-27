Machine Learning: Project 1 
In this document you may find the description of the different jupyter notebooks used for project 1 of machine learning. We will present the notebook files, state for which part of the project they were used for and finally point from which file originated the figures in the report.
/data folder:
	In this folder you find the data used in this code

/implementations folder:
	In this folder you find all the methods used in the code. 
- helpers.py : Contain basic functions
- implementation.py: Contain the functions used for linear and logistic regression
- pca.py: Contain the PCA method used for feature reduction
- plot.py : Contain function to plot the learning curves of the models
- preprocessing.py: Contain process_data() and transform_data() methods. These methods perform feature filtering, data standarization, polynomial building, PCA reduction and others.

testing_implemented_functions.ipynb:
	This jupyter notebook regroups linear and logistic regression. In this notebook we 	demonstrated that our implemented function works and present that different 	functions that we will be using in other jupyter notebooks.
\cross_validation folder:
	In this folder is the code used to cross validate our model and to generate the 	figures used in the report.
run.ipynb:
	This jupyter notebook was used to compute the last optimized model and predict 	the labels on the test set that will be submitted on kaggle.
