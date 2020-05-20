import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

try:
    stats = importr("stats")
except:
    utils = importr('utils')
    utils.install_packages('stats', repos='http://cran.us.r-project.org')
    stats = importr("stats")


class BaseSMSpline(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for Smoothing Spline classification and regression.
     """

    @abstractmethod
    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1):

        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax

    def _estimate_density(self, x):
        
        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def _validate_hyperparameters(self):

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        
        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("method must be an element of [uniform, quantile], got %s." % self.knot_dist)

        if (self.reg_gamma < 0) or (self.reg_gamma > 1):
            raise ValueError("reg_gamma must be >= 0 and <1, got %s." % self.reg_gamma)

        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

    def diff(self, x, order=1):
        
        derivative = np.array(stats.predict(self.sm_, x, deriv=order)[1]).ravel()
        return derivative

    def visualize(self):

        check_is_fitted(self, "sm_")

        fig = plt.figure(figsize=(6, 4))
        inner = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0]) 
        xgrid = np.linspace(self.xmin, self.xmax, 100).reshape([-1, 1])
        ygrid = self.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)
        
        ax1_density = plt.Subplot(fig, inner[1]) 
        xint = ((np.array(self.bins_[1:]) + np.array(self.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)
        plt.show()

    def decision_function(self, x):

        check_is_fitted(self, "sm_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        if "coefficients" in self.sm_.names:
            pred = np.array(stats.predict_glm(sm_, ro.r("data.frame")(x=x))).ravel()
        elif "spar" in self.sm_.names:
            pred = np.array(stats.predict_smooth_spline(self.sm_, x, deriv=0)[1]).ravel()
        return pred


class SMSplineRegressor(BaseSMSpline, RegressorMixin):

    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1):

        super(SMSplineRegressor, self).__init__(knot_num=knot_num,
                                  knot_dist=knot_dist,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred, sample_weight=None):
        return np.average((label - pred) ** 2, axis=0, weights=sample_weight)

    def fit(self, x, y, sample_weight=None):

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self._estimate_density(x)

        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num >= 4:
            self.knot_num = min(unique_num, self.knot_num)
            if self.knot_dist == "uniform":
                self.sm_ = stats.smooth_spline(x, y, nknots=self.knot_num,
                                     spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
            elif self.knot_dist == "quantile":
                knots = np.percentile(x, list(np.linspace(0, 100, self.knot_num + 2, dtype=np.float32))).tolist()
                knots = (knots - self.xmin) / (self.xmax - self.xmin)
                self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                     spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
        else:
            self.sm_ = stats.glm(Formula('y ~ x'), family="gaussian",
                    data=pd.DataFrame({"x":x.ravel(), "y":y.ravel()}),
                    weights=sample_weight)
        return self

    def predict(self, x):

        pred = self.decision_function(x)
        return pred
    

class SMSplineClassifier(BaseSMSpline, ClassifierMixin):

    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1):

        super(SMSplineClassifier, self).__init__(knot_num=knot_num,
                                  knot_dist=knot_dist,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax)
        self.EPS = 10 ** (-8)

    def get_loss(self, label, pred, sample_weight=None):
        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, self.EPS, 1. - self.EPS)
            return - np.average(label * np.log(pred) + (1 - label) * np.log(1 - pred),
                                axis=0, weights=sample_weight)
       
    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()

    def fit(self, x, y, sample_weight=None):

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self._estimate_density(x)
        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num >= 4:
            y = y.copy() * 4 - 2
            self.knot_num = min(unique_num, self.knot_num)
            if self.knot_dist == "uniform":
                self.sm_ = stats.smooth_spline(x, y, nknots=self.knot_num,
                                     spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
            elif self.knot_dist == "quantile":
                knots = np.percentile(x, list(np.linspace(0, 100, self.knot_num + 2, dtype=np.float32))).tolist()
                knots = (knots - self.xmin) / (self.xmax - self.xmin)
                self.sm_ = stats.smooth_spline(x, y, all_knots=ro.FloatVector(knots),
                                     spar=self.reg_gamma, w=sample_weight, tol=1e-6 * (np.max(x) - np.min(x)))
        else:
            self.sm_ = stats.glm(Formula('y ~ x'), family="binomial",
                    data=pd.DataFrame({"x":x.ravel(), "y":y.ravel()}),
                    weights=sample_weight)
        return self
    
    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)