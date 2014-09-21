import numpy as np
import pandas as pd
from sklearn.decomposition.pca import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.mixture import GMM
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

X_test = pd.read_csv('Data/test.csv', header=None).as_matrix()
y = pd.read_csv('Data/trainLabels.csv', header=None)[0].as_matrix()
X = pd.read_csv('Data/train.csv', header=None).as_matrix()