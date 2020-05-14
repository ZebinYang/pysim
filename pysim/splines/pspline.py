import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
stats = importr("stats")

class BasePSpline(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for Smoothing Spline classification and regression.
     """

    @abstractmethod
    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        self.constraint = constraint
        
    def _estimate_density(self, x):
        
        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def _validate_hyperparameters(self):
        
        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

        if self.degree < 0:
            raise ValueError("degree must be >= 0, got" % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        
        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("method must be an element of [uniform, quantile], got %s." % self.knot_dist)

        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if (self.reg_gamma < 0) or (self.reg_gamma > 1):
            raise ValueError("reg_gamma must be >= 0 and <1, got %s." % self.reg_gamma)

        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

        if self.constraint is not None:
            if self.constraint not in ["mono"]:
                raise ValueError("constraint must be None or mono, got %s." % (self.constraint))

    def diff(self, x, order=1):
        
        derivative = np.array(stats.predict(self.sm_, x, deriv=order)[1])
        return derivative

    def visualize(self):

        check_is_fitted(self, "ps_")

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


class PSplineRegressor(BasePSpline, RegressorMixin):

    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        super(PSplineRegressor, self).__init__(knot_num=knot_num,
                                  knot_dist=knot_dist,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax,
                                  degree=degree,
                                  constraint=constraint)

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
            sample_weight = sample_weight * n_samples
           
        if self.constraint is None:
            self.ps_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_inc'))
            self.ps_.fit(x, y, sample_weight)
        elif self.constraint == "mono":
            ps1_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_inc')).fit(x, y)
            ps2_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_dec')).fit(x, y)
            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
            else:
                self.ps_ = ps2_
        return self

    def decision_function(self, x):

        check_is_fitted(self, "ps_")
        pred = self.ps_.predict_mu(x)
        return pred

    def predict(self, x):

        pred = self.decision_function(x)
        return pred
    

class PSplineClassifier(BasePSpline, ClassifierMixin):

    def __init__(self, knot_num=20, knot_dist="uniform", reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        super(PSplineClassifier, self).__init__(knot_num=knot_num,
                                   knot_dist=knot_dist,
                                   reg_gamma=reg_gamma,
                                   xmin=xmin,
                                   xmax=xmax,
                                   degree=degree,
                                   constraint=constraint)
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
            sample_weight = sample_weight * n_samples
            
        if self.constraint is None:
            self.ps_ = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                     lam=self.reg_gamma))
            self.ps_.fit(x, y, sample_weight)
        elif self.constraint == "mono":
            ps1_ = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_inc')).fit(x, y)
            ps2_ = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints='monotonic_dec')).fit(x, y)
            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
            else:
                self.ps_ = ps2_

        return self
    
    def decision_function(self, x):

        check_is_fitted(self, "ps_")
        pred_proba = self.ps_.predict_mu(x)
        pred_proba = np.clip(pred_proba, self.EPS, 1. - self.EPS)
        pred = np.log(pred_proba / (1 - pred_proba))
        return pred

    def predict_proba(self, x):

        pred_proba = self.ps_.predict_mu(x)
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)