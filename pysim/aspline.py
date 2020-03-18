import scipy
import numpy as np
import pandas as pd 
from scipy.linalg import cholesky
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from patsy import dmatrix, build_design_matrices


class BaseASpline(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for ASpline classification and regression.
     """

    @abstractmethod
    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, epsilon=0.00001, threshold=0.99, maxiter=10):

        self.knot_num = knot_num
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        self.epsilon = epsilon
        self.threshold = threshold
        self.maxiter = maxiter

    @staticmethod
    def diff_matrix(order, knot_num):
        results = [] # a container to collect the rows
        n_rows = order + 2
        for _ in range(n_rows): 
            row = [1] # a starter 1 in the row
            if results: # then we are in the second row or beyond
                last_row = results[-1] # reference the previous row
                row.extend([sum(pair) for pair in zip(last_row, last_row[1:])])
                row.append(1)
            results.append(row) # add the row to the results.

        diss_operator = [] # a container to collect the rows
        for i, item in enumerate(np.array(row)):
            diss_operator.append(item * (-1) ** i)
        diss_operator.reverse()
        D = np.zeros((knot_num, knot_num + order + 1), dtype=np.float32)
        for i in range(knot_num):
            D[i,i:(i+order+2)] = diss_operator
        return D

    @property
    def variable_density_(self):
        """Return the projection indices.
        Returns
        -------
        projection_indices_ : ndarray of shape (d, n_estimators)
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        return np.array([estimator.beta_.flatten() for estimator in self.estimators_]).T


    def _validate_hyperparameters(self):
        
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

        if self.xmin >= self.xmax:
            raise ValueError("xmin must be < xmax, got %s and %s." % (self.xmin, self.xmax))

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)

        if self.threshold < 0:
            raise ValueError("threshold must be >= 0, got %s." % self.threshold)

        if not isinstance(self.maxiter, int):
            raise ValueError("maxiter must be an integer, got %s." % self.maxiter)

        if self.maxiter <= 0:
            raise ValueError("maxiter must be > 0, got" % self.maxiter)

            
    def visualize(self):

        check_is_fitted(self, "coef_")

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

        check_is_fitted(self, "coef_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        design_matrix = np.asarray(build_design_matrices([self.selected_xphi_.design_info],
                                         {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])
        pred = np.dot(design_matrix, self.coef_).ravel()
        return pred


class ASplineRegressor(BaseASpline, RegressorMixin):

    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, epsilon=0.00001, threshold=0.99, maxiter=10):

        super(ASplineRegressor, self).__init__(knot_num=knot_num,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax,
                                  degree=degree,
                                  epsilon=epsilon,
                                  threshold=threshold,
                                  maxiter=maxiter)

    @staticmethod
    def _get_loss(label, pred):
        return - np.mean((label - pred) ** 2)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.reshape([-1, 1])

    def estimate_density(self, x):
        
        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def fit(self, x, y, sample_weight=None):

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self.estimate_density(x)
        
        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = sample_weight * n_samples
        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1",
                   {"x": [self.xmin, self.xmax], "knots": knots, "degree": self.degree})
        init_basis = np.asarray(build_design_matrices([xphi.design_info],
                   {"x": x, "knots": knots, "degree": self.degree})[0])

        best_loss = np.inf
        D = self.diff_matrix(self.degree, self.knot_num)
        update_w = np.ones([self.knot_num, 1], dtype=np.float32) 
        BWB = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), init_basis, axes=([0], [0]))
        BWY = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), y, axes=([0], [0]))
        for i in range(self.maxiter):
            DwD = np.tensordot(D * update_w.reshape([-1, 1]), D, axes=([0], [0]))
            try:
                U = cholesky(BWB + 10 * self.reg_gamma * DwD)
                M = scipy.linalg.lapack.clapack.dtrtri(U)[0]
                update_a_temp = np.dot(np.dot(M, M.T.conj()), BWY)
            except:
                update_a_temp = np.dot(np.linalg.pinv(BWB + 10 * self.reg_gamma * DwD, rcond=1e-5), BWY)
            new_loss = self._get_loss(y, np.dot(init_basis, update_a_temp))
            if new_loss - best_loss >= 0:
                break
            best_loss = new_loss
            update_a = update_a_temp
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)

        self.selected_knots_ = list(np.array(knots)[np.reshape(update_w * np.dot(D, update_a) ** 2 > self.threshold, [-1])])
        self.selected_xphi_ = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1", 
               {"x": [self.xmin, self.xmax], "knots": self.selected_knots_, "degree": self.degree})
        selected_basis = np.asarray(build_design_matrices([self.selected_xphi_.design_info],
                          {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])
        seBWB = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), selected_basis, axes=([0], [0]))
        seBWY = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), y, axes=([0], [0]))
        self.coef_ = np.dot(np.linalg.pinv(seBWB, rcond=1e-5), seBWY)
        return self

    def predict(self, x):

        pred = self.decision_function(x)
        return pred
    

class ASplineClassifier(BaseASpline, ClassifierMixin):

    def __init__(self, knot_num=20, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, epsilon=0.00001, threshold=0.99,
                 maxiter=10, maxiter_irls=10):

        super(ASplineClassifier, self).__init__(knot_num=knot_num,
                                   reg_gamma=reg_gamma,
                                   xmin=xmin,
                                   xmax=xmax,
                                   degree=degree,
                                   epsilon=epsilon,
                                   threshold=threshold,
                                   maxiter=maxiter)

        self.maxiter_irls = maxiter_irls
        self.EPS = 10**(-8)

    @staticmethod
    def _link(x):
        with np.errstate(divide="ignore", over="ignore"):
            return 1 / (1 + np.exp(-x))

    @staticmethod
    def _inv_link(x):
        with np.errstate(divide="ignore", over="ignore"):
            return np.log(x) - np.log(1 - x)
    
    @staticmethod
    def _get_loss(label, pred):
        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, 10 ** (-8), 1. - 10 ** (-8))
            return - np.mean(label * np.log(pred) + (1 - label) * np.log(1 - pred))
       
    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y

    def estimate_density(self, x):
        
        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def fit(self, x, y, sample_weight=None):

        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        self.estimate_density(x)
        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = sample_weight * n_samples

        self.estimate_density(x)
        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1",
                       {"x": [self.xmin, self.xmax], "knots": knots, "degree": self.degree})
        init_basis = np.asarray(build_design_matrices([xphi.design_info],
                          {"x": x, "knots": knots, "degree": self.degree})[0])

        tempy = y.copy().astype(np.float32)
        tempy[tempy==0] = 0.01
        tempy[tempy==1] = 0.99
        D = self.diff_matrix(self.degree, self.knot_num)
        update_w = np.ones([self.knot_num, 1], dtype=np.float32) 
        DwD = np.tensordot(D * update_w.reshape([-1, 1]), D, axes=([0], [0]))
        BWB = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), init_basis, axes=([0], [0]))
        BWY = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), self._inv_link(tempy), axes=([0], [0]))
        update_a = np.dot(np.linalg.pinv(BWB + self.reg_gamma * DwD, rcond=1e-5), BWY)
        best_loss = self._get_loss(y, self._link(np.dot(init_basis, update_a)))
        for i in range(self.maxiter):
            best_loss_irls = np.inf
            for j in range(self.maxiter_irls):
                lp = np.dot(init_basis, update_a)
                mu = self._link(lp)
                omega = mu * (1 - mu)
                mask = (np.abs(omega) >= self.EPS) * np.isfinite(omega)
                mask = mask.ravel()
                if np.sum(mask) == 0:
                    break

                BW = init_basis[mask] * sample_weight[mask].reshape([-1, 1])
                DwD = np.tensordot(D * update_w.reshape([-1, 1]), D, axes=([0], [0]))
                BWOB = np.tensordot(BW * omega[mask].reshape([-1, 1]), init_basis[mask], axes=([0], [0]))
                update_a_temp = np.dot(np.linalg.pinv(BWOB + self.reg_gamma * DwD, rcond=1e-5),
                                BWOB.dot(update_a) + np.tensordot(BW, y[mask] - mu[mask], axes=([0], [0])))
                new_loss = self._get_loss(y, self._link(np.dot(init_basis, update_a_temp)))
                if new_loss - best_loss_irls >= 0:
                    break
                best_loss_irls = new_loss
                update_a = update_a_temp

            if best_loss_irls - best_loss >= 0:
                break
            best_loss = best_loss_irls
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)

        self.selected_knots_ = list(np.array(knots)[(update_w * np.dot(D, update_a) ** 2 > self.threshold).ravel()])
        self.selected_xphi_ = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1", 
               {"x": [self.xmin, self.xmax], "knots": self.selected_knots_, "degree": self.degree})
        selected_basis = np.asarray(build_design_matrices([self.selected_xphi_.design_info],
                          {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])

        seBWB = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), selected_basis, axes=([0], [0]))
        seBWY = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), self._inv_link(tempy), axes=([0], [0]))
        self.coef_ = np.dot(np.linalg.pinv(seBWB, rcond=1e-5), seBWY)
        for j in range(self.maxiter_irls):
            lp = np.dot(selected_basis, self.coef_)
            mu = self._link(lp)
            omega = mu * (1 - mu)
            mask = (np.abs(omega) >= self.EPS) * np.isfinite(omega)
            mask = mask.ravel()
            if np.sum(mask) == 0:
                break
            seBW = selected_basis[mask] * sample_weight[mask].reshape([-1, 1])
            seBWOB = np.tensordot(seBW * omega[mask].reshape([-1, 1]), selected_basis[mask], axes=([0], [0]))
            self.coef_ = np.dot(np.linalg.pinv(seBWOB, rcond=1e-5),
                          seBWOB.dot(self.coef_) + np.tensordot(seBW, y[mask] - mu[mask], axes=([0], [0])))
        return self
    
    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)