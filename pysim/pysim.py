import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
from pygam import LinearGAM, LogisticGAM, s

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

from abc import ABCMeta, abstractmethod
from .aspline import ASplineClassifier, ASplineRegressor

from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

utils = importr("utils")
utils.install_packages("fps")

fps = importr("fps")
numpy2ri.activate()


class BaseSim(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for sim classification and regression.
     """

    @abstractmethod
    def __init__(self, method="first_order", spline="a_spline", reg_lambda=0.1,
                 reg_gamma=10, knot_num=20, degree=2, random_state=0):

        self.method = method
        self.spline = spline
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree
        
        self.random_state = random_state

    def _validate_hyperparameters(self):
        
        if self.method not in ["first_order", "second_order", "first_order_thres"]:
            raise ValueError("method must be an element of [first_order, second_order, first_order_thres], got %s." % self.method)
        
        if self.spline not in ["a_spline", "p_sline", "p_spline_mono"]:
            raise ValueError("spline must be an element of [a_spline, p_sline, p_spline_mono], got %s." % self.spline)
        
        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

        if self.degree < 0:
            raise ValueError("degree must be >= 0, got" % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)

        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if self.reg_lambda <= 0:
            raise ValueError("reg_lambda must be > 0, got %s." % self.reg_lambda)

        if self.reg_gamma <= 0:
            raise ValueError("reg_gamma must be > 0, got %s." % self.reg_gamma)

    def _first_order_thres(self, x, y, sample_weight=None, proj_mat=None):

        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        zbar[np.abs(zbar) < self.reg_lambda * np.sum(np.abs(zbar))] = 0
        if proj_mat is not None:
            zbar = np.dot(proj_mat, zbar)
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta

    def _first_order(self, x, y, sample_weight=None, proj_mat=None):

        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)
        if proj_mat is not None:
            sigmat = np.dot(np.dot(proj_mat, sigmat), proj_mat)
        u, s, v = np.linalg.svd(sigmat)
        sigmat = np.dot(np.dot(u, np.diag(s)), u.T)
        
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * np.sum(np.abs(zbar))))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * np.sum(np.abs(zbar))))
        return beta

    def _second_order(self, x, y, sample_weight=None, proj_mat=None):

        n_samples, n_features = x.shape
        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        sigmat = np.tensordot(s1 * y.reshape([-1, 1]) * sample_weight.reshape([-1, 1]), s1, axes=([0], [0]))
        sigmat -= np.average(y, axis=0, weights=sample_weight) * self.inv_cov
        if proj_mat is not None:
            sigmat = np.dot(np.dot(proj_mat, sigmat), proj_mat)
        u, s, v = np.linalg.svd(sigmat)
        sigmat = np.dot(np.dot(u, np.diag(s)), u.T)

        beta_svd_l1norm = np.sum(np.abs(np.linalg.svd(sigmat)[0][:, 0]))  
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * beta_svd_l1norm))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * np.sum(np.abs(beta_svd_l1norm))))
        return beta
    
    def visualize(self):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")

        fig = plt.figure(figsize=(12, 4))
        visu = gridspec.GridSpec(1, 2, wspace=0.15)
        ax1 = plt.Subplot(fig, visu[0]) 
        xgrid = np.linspace(self.shape_fit_.xmin, self.shape_fit_.xmax, 100).reshape([-1, 1])
        ygrid = self.shape_fit_.decision_function(xgrid)
        ax1.plot(xgrid, ygrid)
        ax1.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1)

        ax2 = plt.Subplot(fig, visu[1]) 
        active_beta = []
        active_beta_inx = []
        for idx, beta in enumerate(self.beta_.ravel()):
            if np.abs(beta) > 0:
                active_beta.append(beta)
                active_beta_inx.append(idx)

        rects = ax2.barh(np.arange(len(active_beta)),
                    [self.beta_.ravel()[idx] for _, idx in sorted(zip(np.abs(active_beta), active_beta_inx))])
        ax2.set_yticks(np.arange(len(active_beta)))
        ax2.set_yticklabels(["X" + str(idx + 1) for _, idx in sorted(zip(np.abs(active_beta), active_beta_inx))])
        ax2.set_xlim(np.min(active_beta) - 0.1, np.max(active_beta) + 0.1)
        ax2.set_ylim(-1, len(active_beta_inx))
        ax2.set_title("Projection Indice", fontsize=12)
        fig.add_subplot(ax2)
        plt.show()

    def fit(self, x, y, sample_weight=None, proj_mat=None):

        np.random.seed(self.random_state)
        
        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)
        
        if self.method == "first_order":
            self.beta_ = self._first_order(x, y, sample_weight, proj_mat)
        elif self.method == "first_order_thres":
            self.beta_ = self._first_order_thres(x, y, sample_weight, proj_mat)
        elif self.method == "second_order":
            self.beta_ = self._second_order(x, y, sample_weight, proj_mat)

        if len(self.beta_[np.abs(self.beta_) > 0]) > 0:
            if (self.beta_[np.abs(self.beta_) > 0][0] < 0):
                self.beta_ = - self.beta_
        xb = np.dot(x, self.beta_)
        self._estimate_shape(xb, y, sample_weight, xmin=np.min(xb), xmax=np.max(xb))
        return self

    def decision_function(self, x):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")
        xb = np.dot(x, self.beta_)
        pred = self.shape_fit_.decision_function(xb)
        return pred


class SimRegressor(BaseSim, RegressorMixin):

    def __init__(self, method="first_order", spline="a_spline", reg_lambda=0.1,
                 reg_gamma=10, knot_num=20, degree=2, random_state=0):

        super(SimRegressor, self).__init__(method=method,
                               spline=spline,
                               reg_lambda=reg_lambda,
                               reg_gamma=reg_gamma,
                               knot_num=knot_num,
                               degree=degree,
                               random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)
        return x, y

    def _estimate_shape(self, x, y, sample_weight=None, xmin=-1, xmax=1):

        if self.spline == "a_spline":
            #adaptive spline
            self.shape_fit_ = ASplineRegressor(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                                 xmin=xmin, xmax=xmax, degree=self.degree)
            self.shape_fit_.fit(x, y, sample_weight)

        elif self.spline == "p_spline":
            #penalized spline
            self.shape_fit_ = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma)).fit(x, y)

        elif self.spline == "p_spline_mono":
            #p-spline with monotonic constraint
            shape_fit1 = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints="monotonic_inc")).fit(x, y)
            shape_fit2 = LinearGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints="monotonic_dec")).fit(x, y)
            if mean_squared_error(y.ravel(), shape_fit1.predict(x)) <= mean_squared_error(y.ravel(), shape_fit2.predict(x)):
                self.shape_fit_ = shape_fit1
            else:
                self.shape_fit_ = shape_fit2

    def predict(self, x):

        pred = self.decision_function(x)
        return pred

class SimClassifier(BaseSim, ClassifierMixin):

    def __init__(self, method="first_order", spline="a_spline", reg_lambda=0.1,
                 reg_gamma=10, knot_num=20, degree=2, random_state=0):

        super(SimClassifier, self).__init__(method=method,
                               spline=spline,
                               reg_lambda=reg_lambda,
                               reg_gamma=reg_gamma,
                               knot_num=knot_num,
                               degree=degree,
                               random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y

    def _estimate_shape(self, x, y, sample_weight=None, xmin=-1, xmax=1):

        if self.spline == "a_spline":
            #adaptive spline
            self.shape_fit_ = ASplineClassifier(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                             xmin=xmin, xmax=xmax, degree=self.degree)
            self.shape_fit_.fit(x, y, sample_weight)

        elif self.spline == "p_spline":
            #penalized spline
            self.shape_fit_ = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                                lam=self.reg_gamma)).fit(x, y)

        elif self.spline == "p_spline_mono":
            #p-spline with monotonic constraint
            shape_fit1 = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints="monotonic_inc")).fit(x, y)
            shape_fit2 = LogisticGAM(s(0, n_splines=self.knot_num, spline_order=self.degree,
                             lam=self.reg_gamma, constraints="monotonic_dec")).fit(x, y)
            if roc_auc_score(y.ravel(), shape_fit1.predict_proba(x)) <= roc_auc_score(y.ravel(), shape_fit2.predict_proba(x)):
                self.shape_fit_ = shape_fit1
            else:
                self.shape_fit_ = shape_fit2

    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False).reshape([-1, 1])
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba).reshape([-1, 1])