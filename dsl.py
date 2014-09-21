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

pca2 = PCA(n_components=2, whiten=True)
pca2.fit(np.r_[X, X_test])
X_pca = pca2.transform(X)
i0 = np.argwhere(y == 0)[:, 0]
i1 = np.argwhere(y == 1)[:, 0]
X0 = X_pca[i0, :]
X1 = X_pca[i1, :]
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X1[:, 1], 'b*')

pca = PCA(whiten=True)
X_all = pca.fit_transform(np.r_[X, X_test])
print (pca.explained_variance_ratio_)

def kde_plot(x):
        from scipy.stats.kde import gaussian_kde
        kde = gaussian_kde(x)
        positions = np.linspace(x.min(), x.max())
        smoothed = kde(positions)
        plt.plot(positions, smoothed)

def qq_plot(x):
    from scipy.stats import probplot
    probplot(x, dist='norm', plot=plt)
    
kde_plot(X_all[:, 0])
kde_plot(X_all[:, 2])
kde_plot(X_all[:, 30])
kde_plot(X_all[:, 38])