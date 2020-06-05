import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from pygam import LinearGAM, LogisticGAM, s

EPSILON = 1e-7

__all__ = ["PSplineRegressor", "PSplineClassifier"]


class BasePSpline(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        self.knot_num = knot_num
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        self.constraint = constraint
        
    def _estimate_density(self, x):
        
        """method to estimate the density of input data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        """

        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def _validate_hyperparameters(self):
                
        """method to validate model parameters
        """

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

        if self.degree < 0:
            raise ValueError("degree must be >= 0, got" % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)

        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if self.reg_gamma < 0:
            raise ValueError("reg_gamma must be >= 0, got %s." % self.reg_gamma)

        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

        if self.constraint is not None:
            if self.constraint not in ["mono"]:
                raise ValueError("constraint must be None or mono, got %s." % (self.constraint))

    def visualize(self):

        """draw the fitted shape function
        """

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

    """PSpline regression.

    Details:
    1. This is an API for the python package `pygam`, and we use the p-spline by treating it as an univariate GAM.
    2. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.
    
    Parameters
    ----------

    knot_num : int, optional. default=20
           the number of knots

    reg_gamma : float, optional. default=0.1
            the roughness penalty strength of the spline algorithm, range from 0 to :math:`+\infty` 
    
    degree : int, optional. default=2
          the order of the spline
    
    xmin : float, optional. default=-1
        the min boundary of the input
    
    xmax : float, optional. default=1
        the max boundary of the input
    
    constraint : None or str, optional. default=None
        constraint=None means no constrant and constraint="mono" for monotonic constraint
    """

    def __init__(self, knot_num=20,  reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        super(PSplineRegressor, self).__init__(knot_num=knot_num,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax,
                                  degree=degree,
                                  constraint=constraint)

    def _validate_input(self, x, y):
        
        """method to validate data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred, sample_weight=None):
        
        """method to calculate the MSE loss
        
        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        float
            the MSE loss
        """
        
        loss = np.average((label - pred) ** 2, axis=0, weights=sample_weight)
        return loss
    
    def fit(self, x, y, sample_weight=None):

        """fit the p-spline

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self._estimate_density(x)

        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)
           
        if self.constraint is None:
            self.ps_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num,
                            spline_order=self.degree, lam=self.reg_gamma))
            self.ps_.fit(x, y, sample_weight)

        elif self.constraint == "mono":
            ps1_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                          lam=self.reg_gamma, constraints='monotonic_inc')).fit(x, y)

            ps2_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                          lam=self.reg_gamma, constraints='monotonic_dec')).fit(x, y)

            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
            else:
                self.ps_ = ps2_
        return self

    def decision_function(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """

        check_is_fitted(self, "ps_")
        
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        pred = self.ps_.predict_mu(x)
        return pred

    def predict(self, x):

        """output f(x) for given samples
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """
        pred = self.decision_function(x)
        return pred
    

class PSplineClassifier(BasePSpline, ClassifierMixin):

    """PSpline classification.

    Details:
    1. This is an API for the python package `pygam`, and we use the p-spline by treating it as an univariate GAM.
    2. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.
    3. reg_gamma will be increased if the current value is too small
    
    Parameters
    ----------
    knot_num : int, optional. default=20
           the number of knots

    reg_gamma : float, optional. default=0.1
            the roughness penalty strength of the spline algorithm, range from 0 to :math:`+\infty` 
    
    degree : int, optional. default=2
          the order of the spline
    
    xmin : float, optional. default=-1
        the min boundary of the input
    
    xmax : float, optional. default=1
        the max boundary of the input
    
    constraint : None or str, optional. default=None
        constraint=None means no constrant and constraint="mono" for monotonic constraint
    """

    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        super(PSplineClassifier, self).__init__(knot_num=knot_num,
                                   reg_gamma=reg_gamma,
                                   xmin=xmin,
                                   xmax=xmax,
                                   degree=degree,
                                   constraint=constraint)
        
    def get_loss(self, label, pred, sample_weight=None):
        
        """method to calculate the cross entropy loss
        
        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        float 
            the cross entropy loss
        """

        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, EPSILON, 1. - EPSILON)
            loss = - np.average(label * np.log(pred) + (1 - label) * np.log(1 - pred),
                                axis=0, weights=sample_weight)
        return loss
       
    def _validate_input(self, x, y):
        
        """method to validate data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()

    def fit(self, x, y, sample_weight=None):

        """fit the p-spline

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self._estimate_density(x)
        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.round(sample_weight / np.sum(sample_weight) * n_samples, 4)
            
        if self.constraint is None:
            i = 0
            exit = True
            while exit:
                try:
                    self.ps_ = LogisticGAM(s(0, basis="ps", n_splines=self.knot_num,
                                    spline_order=self.degree, lam=self.reg_gamma + 10 ** (i - 3)))
                    self.ps_.fit(x, y, sample_weight)
                    exit = False
                except ValueError:
                    i += 1
            self.reg_gamma *= 10 ** i
            
        elif self.constraint == "mono":
            i = 0
            exit = True
            while exit:
                try:
                    ps1_ = LogisticGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 10 ** (i - 3), constraints='monotonic_inc')).fit(x, y)
                    exit = False
                except ValueError:
                    i += 1

            j = 0
            exit = True
            while exit:
                try:
                    ps2_ = LogisticGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 10 ** (j - 3), constraints='monotonic_dec')).fit(x, y)
                    exit = False
                except ValueError:
                    j += 1

            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
                self.reg_gamma += 10 ** (i - 3)
            else:
                self.ps_ = ps2_
                self.reg_gamma += 10 ** (j - 3)

        return self
    
    def decision_function(self, x):

        """output f(x) for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x) 
        """

        check_is_fitted(self, "ps_")
                
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        pred_proba = self.ps_.predict_mu(x)
        pred_proba[np.isnan(pred_proba)] = 0.5
        pred_proba = np.clip(pred_proba, EPSILON, 1. - EPSILON)
        pred = np.log(pred_proba / (1 - pred_proba))
        return pred

    def predict_proba(self, x):

        """output probability prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing probability prediction
        """
        pred_proba = self.ps_.predict_mu(x)
        return pred_proba

    def predict(self, x):
        
        """output binary prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing binary prediction
        """

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)