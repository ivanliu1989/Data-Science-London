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

qq_plot(X[:,0])
qq_plot(X[:,2])
qq_plot(X[:,30])
qq_plot(X[:,38])

class PcaGmm(BaseEstimator):
    def __init__(self, X_all,
                 pca_components = 12, gmm_components = 4,
                 covariance_type = "full", min_covar = 0.1,
                 gamma = 0, C = 1.0):
        self.pca_components = pca_components
        self.gmm_components = gmm_components
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.gamma = gamma
        self.C = C
        self.X_all = X_all
        X_all = X_all[:, :pca_components]
        self.gmm = GMM(n_components = gmm_components,
                       covariance_type = covariance_type,
                       min_covar = min_covar)
        self.gmm.fit(X_all)
    def fit(self, X, y):
        X = X[:, :self.pca_components]
        X = self.gmm.predict_proba(X)
        self.svm = SVC(C = self.C, gamma = self.gamma)
        self.svm.fit(X, y)
    def predict(self, X):
        X = X[:, :self.pca_components]
        return self.svm.predict(self.gmm.predict_proba(X))
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    def transform(self, X, y = None):
        X = X[:, :self.pca_components]
        return self.gmm.predict_proba(X)
    def __str__(self):
        return "PCA(%d)-GMM(%d, %s, %f)-SVM(C=%f, gamma=%f)" % (self.pca_components, self.gmm_components,self.covariance_type, self.min_covar,self.C, self.gamma)
                                                                
clf = PcaGmm(X_test, 12, 4, 'full', 0, gamma = .6, C = 0.3)
X_t = clf.transform(pca.transform(X))
X0 = X_t[i0, :]
X1 = X_t[i1, :]
pca2 = PCA(n_components=2)
pca2.fit(np.r_[X0, X1])
X0 = pca2.transform(X0)
X1 = pca2.transform(X1)
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X1[:, 1], 'b*')