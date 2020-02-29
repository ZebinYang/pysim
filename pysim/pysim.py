import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from .aspline import ASplineClassifier, ASplineRegressor
from pygam import LinearGAM, s

utils = importr('utils')
utils.install_packages('fps')

fps = importr('fps')
numpy2ri.activate()


class SIM(BaseEstimator, RegressorMixin):

    def __init__(self, task="Regression", method="first", spline="augbs", reg_lambda=0.1,
                 reg_gamma=0.1, knot_num=20, degree=2, random_state=0):

        self.task = task
        self.method = method
        self.spline = spline
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree
        
        self.random_state = random_state

    def first_stein_hard_thresholding(self, x, y, sample_weight=None):

        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        zbar[np.abs(zbar) < self.reg_lambda * np.sum(np.abs(zbar))] = 0
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta

    def first_stein(self, x, y, sample_weight=None):

        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)
        u, s, v = np.linalg.svd(sigmat)
        sigmat = np.dot(np.dot(u, np.diag(s)), u.T)
        
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * np.sum(np.abs(zbar))))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * np.sum(np.abs(zbar))))
        return beta

    def second_stein(self, x, y, sample_weight=None):

        n_samples, n_features = x.shape
        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]) * sample_weight.reshape([-1, 1]), s1, axes=([0], [0]))
        sigmat -= np.average(y, axis=0, weights=sample_weight) * self.inv_cov
        u, s, v = np.linalg.svd(sigmat)
        sigmat = np.dot(np.dot(u, np.diag(s)), u.T)

        beta_svd_l1norm = np.sum(np.abs(np.linalg.svd(sigmat)[0][:, 0]))  
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * beta_svd_l1norm))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * np.sum(np.abs(beta_svd_l1norm))))
        return beta
    
    def estimate_shape_function(self, x, y, sample_weight=None):

        if self.spline == "augbs":
            #augmented bspline
            if self.task == "Regression":
                self.link_fit_ = ASplineRegressor(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                                 xmin=self.xmin_, xmax=self.xmax_, degree=self.degree)
            elif self.task == "Classification":
                self.link_fit_ = ASplineClassifier(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                                 xmin=self.xmin_, xmax=self.xmax_, degree=self.degree)
            self.link_fit_.fit(x, y, sample_weight)

        elif self.spline == "ps":
            #p-spline
            self.link_fit_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma)).fit(x, y)

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

        xgrid = np.linspace(self.xmin_, self.xmax_, 100).reshape([-1, 1])
        ygrid = self.link_fit_.predict(xgrid)
        if return_data:
            return xgrid, ygrid
        else:
            return plt.plot(xgrid, ygrid)

    def fit(self, x, y, sample_weight=None):

        np.random.seed(self.random_state)
        n_samples, n_features = x.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples

        if self.method == "first":
            self.beta_ = self.first_stein(x, y, sample_weight)
        elif self.method == "first_thresholding":
            self.beta_ = self.first_stein_hard_thresholding(x, y, sample_weight)
        elif self.method == "second":
            self.beta_ = self.second_stein(x, y, sample_weight)

        if len(self.beta_[np.abs(self.beta_) > 0]) > 0:
            if (self.beta_[np.abs(self.beta_) > 0][0] < 0):
                self.beta_ = - self.beta_
        xb = np.dot(x, self.beta_)
        self.xmin_ = np.min(xb)
        self.xmax_ = np.max(xb)
        self.estimate_shape_function(xb, y, sample_weight)
        return self

    def predict(self, x):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "link_fit_")
        xb = np.dot(x, self.beta_)
        pred = self.link_fit_.predict(xb)
        return pred