import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from .aspline import ASpline
from pygam import LinearGAM, s

utils = importr('utils')
utils.install_packages('fps')

fps = importr('fps')
numpy2ri.activate()


class SIM(BaseEstimator, RegressorMixin):

    def __init__(self, mu=0, sigma=1, method="first", spline="bs", reg_lambda=0.5, reg_gamma=0.1, knot_num=10, degree=2, random_state=0):

        self.mu = mu 
        self.sigma = sigma
        self.method = method
        self.spline = spline
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree
        
        self.random_state = random_state

    def first_stein_hard_thresholding(self, x, y):

        s1 = (x - self.mu) / self.sigma ** 2
        zbar = np.mean(y.reshape(-1, 1) * s1, axis=0)
        zbar[np.abs(zbar) < self.reg_lambda * np.max(np.abs(zbar))] = 0
        beta = zbar / np.linalg.norm(zbar)
        return beta

    def first_stein(self, x, y):

        s1 = (x - self.mu) / self.sigma ** 2
        zbar = np.mean(y.reshape(-1, 1) * s1, axis=0)
        sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * np.max(np.abs(zbar))))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda))
        return beta

    def second_stein(self, x, y):
        
        n_samples, n_features = x.shape
        s1 = (x - self.mu) / self.sigma ** 2
        sigmat = np.zeros((n_features, n_features))
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]), s1, axes=([0], [0])) / n_samples
        sigmat[np.diag_indices_from(sigmat)] += - np.mean(y) / self.sigma ** 2        

        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * np.max(np.abs(zbar))))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda))
        return beta
    
    def estimate_shape_function(self, x, y):

        if self.spline == "bs":
            #augmented bspline
            self.link_fit_ = ASpline(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                             xmin=self.xmin_, xmax=self.xmax_, degree=self.degree)
            self.link_fit_.fit(x, y)
        elif self.spline == "mono":
            #p-spline with monotonic constraint
            link_fit1_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_inc')).fit(x, y)
            link_fit2_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_dec')).fit(x, y)
            if np.linalg.norm(link_fit1_.predict(x) - y.ravel()) <= np.linalg.norm(link_fit2_.predict(x) - y.ravel()):
                self.link_fit_ = link_fit1_
            else:
                self.link_fit_ = link_fit2_
            
    def visualize_shape_function(self, return_data=False):

        if self.spline == "bs":
            pred = self.link_fit_.predict(np.linspace(self.xmin_, self.xmax_, 100).reshape([-1, 1]))
        elif self.spline == "mono":
            pred = self.link_fit_.predict(np.linspace(self.xmin_, self.xmax_, 100).reshape([-1, 1]))
        if return_data:
            return np.linspace(self.xmin_, self.xmax_, 100), pred
        else:
            return plt.plot(np.linspace(self.xmin_, self.xmax_, 100), pred)

    def fit(self, x, y):

        np.random.seed(self.random_state)
        if self.method == "first":
            self.beta_ = self.first_stein(x, y)
        elif self.method == "first_thresholding":
            self.beta_ = self.first_stein_hard_thresholding(x, y)
        elif self.method == "second":
            self.beta_ = self.second_stein(x, y)
        
        self.beta_ = - self.beta_ if (self.beta_[np.abs(self.beta_) > 0][0] < 0) else self.beta_
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
        elif self.spline == "mono":
            pred = self.link_fit_.predict(xb)
        return pred