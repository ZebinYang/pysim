import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from aspline import ASpline

stats = importr('stats')
fps = importr('fps')
numpy2ri.activate()


class SIM(BaseEstimator, RegressorMixin):

    def __init__(self, mu=0, sigma=1, method="first", spline="sm", reg_lambda=0.5, reg_gamma=0.1, knot_num=10, degree=2, random_state=0):

        self.mu = mu 
        self.sigma = sigma
        self.method = method
        self.spline = spline
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree
        
        self.random_state = random_state

    def first_stein(self, x, y):
        
        n_samples, n_features = x.shape
        s1 = (x - self.mu)/ self.sigma**2
        zbar = np.mean(y.reshape(-1, 1) * s1, axis=0)
        sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda))
        return beta

    def second_stein(self, x, y):
        
        n_samples, n_features = x.shape
        s1 = (x - self.mu)/ self.sigma**2
        sigmat = np.zeros((n_features, n_features))
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]), s1, axes=([0], [0])) / n_samples
        sigmat[np.diag_indices_from(sigmat)] += - np.mean(y) / self.sigma ** 2        

        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda))
        return beta
    
    def estimate_shape_function(self, x, y):

        if self.spline == "bs":
            self.link_fit_ = ASpline(knot_num=self.knot_num, ridge_gamma=self.reg_gamma,
                             xmin=self.xmin_, xmax=self.xmax_, degree=self.degree)
            self.link_fit_.fit(x, y)
        elif self.spline == "sm":
            self.link_fit_ = stats.smooth_spline(x, y)

    def visualize_shape_function(self):

        if self.spline == "bs":
            pred = self.link_fit_.predict(np.linspace(self.xmin_, self.xmax_, 100).reshape([-1, 1]))
        elif self.spline == "sm":
            pred = stats.predict(self.link_fit_, np.linspace(self.xmin_, self.xmax_, 100))[1]
        plt.plot(np.linspace(self.xmin_, self.xmax_, 100), pred)

    def fit(self, x, y):

        n_samples, n_features = x.shape
        np.random.seed(self.random_state)

        # stein's method
        if self.method == "first":
            self.beta_ = self.first_stein(x, y)
        elif self.method == "second":
            self.beta_ = self.second_stein(x, y)

        xb = np.dot(x, self.beta_)
        self.xmin_ = np.min(xb)
        self.xmax_ = np.max(xb)
        self.estimate_shape_function(xb, y)
        return self

    def predict(self, x):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "link_fit_")
        xb = np.dot(x, self.beta_)

        if self.spline == "bs":
            pred = self.link_fit_.predict(xb)
        elif self.spline == "sm":
            pred = stats.predict(self.link_fit_, xb)[1]
        return pred