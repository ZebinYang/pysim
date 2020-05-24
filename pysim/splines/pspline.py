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

class BasePSpline(BaseEstimator, metaclass=ABCMeta):

    """Base class for P-Spline classification and regression.

    Parameters
    ----------

    :type  knot_num: int, optional. default=20
    :param knot_num: The number of knots

    :type  reg_gamma: float, optional. default=0.1
    :param reg_gamma: The roughness penalty strength of the spline algorithm
    
        For spline="smoothing_spline", it ranges from 0 to 1 

        For spline="p_spline","mono_p_spline" or "a_spline", it ranges from 0 to $+\infty$.
    
    :type  degree: int, optional. default=2
    :param degree: The order of the spline
    
    :type  xmin: float, optional. default=-1
    :param xmin: The min boundary of the input
    
    :type  xmax: float, optional. default=1
    :param xmax: The max boundary of the input
    
    :type  constraint: None or str, optional. default=None
    :param constraint: The extra constraint for p-spline, it can be "mono" for monotonic constraint or None for no constraint

    :type  random_state: int, optional. default=0
    :param random_state: The random seed
    """
    
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
        x : array-like of shape (n_samples, n_features),
            containing the input dataset
        Returns
        -------
        None
        """

        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def _validate_hyperparameters(self):
                
        """method to validate model parameters
        Parameters
        ---------
        None
        Returns
        -------
        None
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
        Parameters
        ---------
        None
        Returns
        -------
        None
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
        x : array-like of shape (n_samples, 1),
            containing the input dataset
        y : array-like of shape (n_samples,),
            containing the output dataset
        Returns
        -------
        None
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred, sample_weight=None):
        
        """method to calculate the MSE loss
        Parameters
        ---------
        label : array-like of shape (n_samples,),
            containing the input dataset
        pred : array-like of shape (n_samples,),
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional,
            containing sample weights
        Returns
        -------
        loss : float,
            the MSE value
        """
        
        loss = np.average((label - pred) ** 2, axis=0, weights=sample_weight)
        return loss
    
    def fit(self, x, y, sample_weight=None):

        """fit the p-spline

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features),
            containing the input dataset
        y : array-like of shape (n_samples,),
            containing target values
        sample_weight : array-like of shape (n_samples,), optional,
            containing sample weights
        Returns
        -------
        self : object,
            Returns fitted p-spline object
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
                    self.ps_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num,
                                    spline_order=self.degree, lam=self.reg_gamma + 0.1 * i))
                    self.ps_.fit(x, y, sample_weight)
                    exit = False
                except ValueError:
                    i += 1

        elif self.constraint == "mono":
            i = 0
            exit = True
            while exit:
                try:
                    ps1_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 0.1 * i, constraints='monotonic_inc')).fit(x, y)
                    exit = False
                except ValueError:
                    i += 1

            i = 0
            exit = True
            while exit:
                try:
                    ps2_ = LinearGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 0.1 * i, constraints='monotonic_dec')).fit(x, y)
                    exit = False
                except ValueError:
                    i += 1

            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
            else:
                self.ps_ = ps2_
        return self

    def decision_function(self, x):

        """output f(x) for given samples
        Parameters
        ---------
        x : array-like of shape (n_samples, 1),
            containing the input dataset
        Returns
        -------
        pred : np.array of shape (n_samples,),
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
        x : array-like of shape (n_samples, 1),
            containing the input dataset
        Returns
        -------
        pred : np.array of shape (n_samples,),
            containing f(x) 
        """
        pred = self.decision_function(x)
        return pred
    

class PSplineClassifier(BasePSpline, ClassifierMixin):

    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, constraint=None):

        super(PSplineClassifier, self).__init__(knot_num=knot_num,
                                   reg_gamma=reg_gamma,
                                   xmin=xmin,
                                   xmax=xmax,
                                   degree=degree,
                                   constraint=constraint)
        self.EPS = 10 ** (-8)
        
    def get_loss(self, label, pred, sample_weight=None):
        
        """method to calculate the cross entropy loss
        Parameters
        ---------
        label : array-like of shape (n_samples,),
            containing the input dataset
        pred : array-like of shape (n_samples,),
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional,
            containing sample weights
        Returns
        -------
        loss : float
            the cross entropy value
        """

        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, self.EPS, 1. - self.EPS)
            loss = - np.average(label * np.log(pred) + (1 - label) * np.log(1 - pred),
                                axis=0, weights=sample_weight)
        return loss
       
    def _validate_input(self, x, y):
        
        """method to validate data
        Parameters
        ---------
        x : array-like of shape (n_samples, 1),
            containing the input dataset
        y : array-like of shape (n_samples,),
            containing the output dataset
        Returns
        -------
        None
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
        x : array-like of shape (n_samples, n_features),
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional,
            containing sample weights
        Returns
        -------
        self : object,
            Returns fitted p-spline object
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
                                    spline_order=self.degree, lam=self.reg_gamma + 0.1 * i))
                    self.ps_.fit(x, y, sample_weight)
                    exit = False
                except ValueError:
                    i += 1

        elif self.constraint == "mono":
            i = 0
            exit = True
            while exit:
                try:
                    ps1_ = LogisticGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 0.1 * i, constraints='monotonic_inc')).fit(x, y)
                    exit = False
                except ValueError:
                    i += 1

            i = 0
            exit = True
            while exit:
                try:
                    ps2_ = LogisticGAM(s(0, basis="ps", n_splines=self.knot_num, spline_order=self.degree,
                                  lam=self.reg_gamma + 0.1 * i, constraints='monotonic_dec')).fit(x, y)
                    exit = False
                except ValueError:
                    i += 1

            if ps1_.loglikelihood(x, y) >= ps2_.loglikelihood(x, y):
                self.ps_ = ps1_
            else:
                self.ps_ = ps2_

        return self
    
    def decision_function(self, x):

        """output f(x) for given samples
        Parameters
        ---------
        x : array-like of shape (n_samples, 1),
            containing the input dataset
        Returns
        -------
        pred : np.array of shape (n_samples,),
            containing f(x) 
        """

        check_is_fitted(self, "ps_")
                
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        pred_proba = self.ps_.predict_mu(x)
        pred_proba[np.isnan(pred_proba)] = 0.5
        pred_proba = np.clip(pred_proba, self.EPS, 1. - self.EPS)
        pred = np.log(pred_proba / (1 - pred_proba))
        return pred

    def predict_proba(self, x):

        """output probability prediction for given samples
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features),
            containing the input dataset
        Returns
        -------
        pred : np.array of shape (n_samples,),
            containing probability prediction
        """
        pred_proba = self.ps_.predict_mu(x)
        return pred_proba

    def predict(self, x):
        
        """output binary prediction for given samples
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features),
            containing the input dataset
        Returns
        -------
        pred : np.array of shape (n_samples,),
            containing binary prediction
        """

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)