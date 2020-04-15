import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

import scipy
from scipy.linalg import cholesky


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

    def _diff_matrix(self, order, knot_num):
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
            D[i,i:(i + order + 2)] = diss_operator
        return D

    def _estimate_density(self, x):
        
        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

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

        if self.xmin > self.xmax:
            raise ValueError("xmin must be <= xmax, got %s and %s." % (self.xmin, self.xmax))

        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0, got %s." % self.epsilon)

        if self.threshold < 0:
            raise ValueError("threshold must be >= 0, got %s." % self.threshold)

        if not isinstance(self.maxiter, int):
            raise ValueError("maxiter must be an integer, got %s." % self.maxiter)

        if self.maxiter <= 0:
            raise ValueError("maxiter must be > 0, got" % self.maxiter)

    def _create_basis(self, inputs, p, knot_vector):

        if p == 0:
            basis = np.where(np.all([knot_vector[:-1] <= inputs,
                                   inputs < knot_vector[1:]], axis=0), 1.0, 0.0)
            basis[np.where(inputs == knot_vector[-1])[0], -1] = 1.0
            return basis
        else:
            basis_p_minus_1 = self._create_basis(inputs, p - 1, knot_vector)

        first_term_numerator = inputs - knot_vector[:-p]
        first_term_denominator = knot_vector[p:] - knot_vector[:-p]

        second_term_numerator = knot_vector[(p + 1):] - inputs
        second_term_denominator = (knot_vector[(p + 1):] - knot_vector[1:-p])

        with np.errstate(divide="ignore", invalid="ignore"):
            first_term = np.where(first_term_denominator != 0.0,
                                  (first_term_numerator /
                                   first_term_denominator), 0.0)
            second_term = np.where(second_term_denominator != 0.0,
                                   (second_term_numerator /
                                    second_term_denominator), 0.0)

        basis = (first_term[:, :-1] * basis_p_minus_1[:, :-1] +
                 second_term * basis_p_minus_1[:, 1:])
        basis[np.where(inputs == knot_vector[-1])[0], -1] = 1.0
        return basis

    def diff(self, x, order=1):
        
        # This function evaluates the derivative of the fitted ASpline w.r.t. the inputs, 
        # which is adopted from https://github.com/johntfoster/bspline/blob/master/bspline/bspline.py.

        def diff_inner(inputs, t, p):

            numer1 = +p
            numer2 = -p
            denom1 = t[p:-1] - t[:-(p+1)]
            denom2 = t[(p+1):] - t[1:-p]

            with np.errstate(divide="ignore", invalid="ignore"):
                ci1 = np.where(denom1 != 0., (numer1 / denom1), 0.)
                ci2 = np.where(denom2 != 0., (numer2 / denom2), 0.)

            Bi1 = self._create_basis(inputs, p - 1, t[:-1]) 
            Bi2 = self._create_basis(inputs, p - 1, t[1:])
            return ((ci1, Bi1, t[:-1], p - 1), (ci2, Bi2, t[1:], p - 1))

        x = x.copy()
        x = check_array(x, accept_sparse=["csr", "csc", "coo"])
        if order > self.degree: 
            return np.zeros(x.shape[0], dtype=np.float32)

        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        knot_vector = np.array([self.xmin] * (self.degree + 1) + self.selected_knots_ + [self.xmax] * (self.degree + 1))
        terms = [ (1., self._create_basis(x, self.degree, knot_vector), knot_vector, self.degree) ]
        for k in range(order):
            tmp = []
            for Ci, Bi, t, p in terms:
                tmp.extend((Ci * cn, Bn, tn, pn) for cn, Bn, tn, pn in diff_inner(x, t, p))
            terms = tmp
        basis_derivatives = np.sum([ci * Bi for ci, Bi, _, _ in terms], 0)
        return np.dot(basis_derivatives, self.coef_)

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
        selected_basis = self._create_basis(x, self.degree, self.selected_knot_vector_)
        pred = np.dot(selected_basis, self.coef_).ravel()
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
        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        knot_vector = np.array([self.xmin] * (self.degree + 1) + knots + [self.xmax] * (self.degree + 1))
        init_basis = self._create_basis(x, self.degree, knot_vector)

        D = self._diff_matrix(self.degree, self.knot_num)
        update_w = np.ones([self.knot_num, 1], dtype=np.float32) 
        BWB = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), init_basis, axes=([0], [0]))
        BWY = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), y, axes=([0], [0]))
        
        if self.reg_gamma == 0:
            try:
                U = cholesky(BWB)
                M = scipy.linalg.lapack.clapack.dtrtri(U)[0]
                self.coef_ = np.dot(np.dot(M, M.T.conj()), BWY)
            except:
                self.coef_ = np.dot(np.linalg.pinv(BWB, rcond=1e-5), BWY)
            self.selected_knots_ = knots
            self.selected_knot_vector_ = knot_vector
            return self
        
        for i in range(self.maxiter):
            DwD = np.tensordot(D * update_w.reshape([-1, 1]), D, axes=([0], [0]))
            try:
                U = cholesky(BWB + self.reg_gamma * DwD)
                M = scipy.linalg.lapack.clapack.dtrtri(U)[0]
                update_a = np.dot(np.dot(M, M.T.conj()), BWY)
            except:
                update_a = np.dot(np.linalg.pinv(BWB + self.reg_gamma * DwD, rcond=1e-5), BWY)
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)

        self.selected_knots_ = list(np.array(knots)[np.reshape(update_w * np.dot(D, update_a) ** 2 > self.threshold, [-1])])
        self.selected_knot_vector_ = np.array([self.xmin] * (self.degree + 1) + self.selected_knots_ + [self.xmax] * (self.degree + 1))
        selected_basis = self._create_basis(x, self.degree, self.selected_knot_vector_)
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

        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        knot_vector = np.array([self.xmin] * (self.degree + 1) + knots + [self.xmax] * (self.degree + 1))
        init_basis = self._create_basis(x, self.degree, knot_vector)

        tempy = y.copy().astype(np.float32)
        tempy[tempy==0] = 0.01
        tempy[tempy==1] = 0.99

        if self.reg_gamma == 0:
            best_loss_irls = np.inf
            BWB = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), selected_basis, axes=([0], [0]))
            BWY = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), self._inv_link(tempy), axes=([0], [0]))
            self.coef_ = np.dot(np.linalg.pinv(seBWB, rcond=1e-3), seBWY)
            for j in range(self.maxiter_irls):
                lp = np.dot(init_basis, self.coef_)
                mu = self._link(lp)
                omega = mu * (1 - mu)
                mask = (np.abs(omega) >= self.EPS) * np.isfinite(omega)
                mask = mask.ravel()
                if np.sum(mask) == 0:
                    break
                BW = selected_basis[mask] * sample_weight[mask].reshape([-1, 1])
                BWOB = np.tensordot(BW * omega[mask].reshape([-1, 1]), init_basis[mask], axes=([0], [0]))
                coef_temp = np.dot(np.linalg.pinv(BWOB, rcond=1e-3), BWOB.dot(self.coef_) \
                        + np.tensordot(BW, y[mask] - mu[mask], axes=([0], [0])))
                new_loss = self.get_loss(y, self._link(np.dot(init_basis, coef_temp)), sample_weight)
                if new_loss - best_loss_irls >= 0:
                    break
                self.coef_ = coef_temp
                best_loss_irls = new_loss
            self.selected_knots_ = knots
            self.selected_knot_vector_ = knot_vector
            return self
        
        D = self._diff_matrix(self.degree, self.knot_num)
        update_w = np.ones([self.knot_num, 1], dtype=np.float32) 
        DwD = np.tensordot(D * update_w.reshape([-1, 1]), D, axes=([0], [0]))
        BWB = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), init_basis, axes=([0], [0]))
        BWY = np.tensordot(init_basis * sample_weight.reshape([-1, 1]), self._inv_link(tempy), axes=([0], [0]))
        # The value of rcond for classifier is set to a relatively large value (1e-3) as compared to that of regressor (1e-5).
        update_a = np.dot(np.linalg.pinv(BWB + self.reg_gamma * DwD, rcond=1e-3), BWY)
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
                new_loss = self.get_loss(y, self._link(np.dot(init_basis, update_a_temp)), sample_weight)
                if new_loss - best_loss_irls >= 0:
                    break
                update_a = update_a_temp
                best_loss_irls = new_loss
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)

        self.selected_knots_ = list(np.array(knots)[(update_w * np.dot(D, update_a) ** 2 > self.threshold).ravel()])
        self.selected_knot_vector_ = np.array([self.xmin] * (self.degree + 1) + self.selected_knots_ + [self.xmax] * (self.degree + 1))
        selected_basis = self._create_basis(x, self.degree, self.selected_knot_vector_)

        best_loss_irls = np.inf
        seBWB = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), selected_basis, axes=([0], [0]))
        seBWY = np.tensordot(selected_basis * sample_weight.reshape([-1, 1]), self._inv_link(tempy), axes=([0], [0]))
        self.coef_ = np.dot(np.linalg.pinv(seBWB, rcond=1e-3), seBWY)
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
            coef_temp = np.dot(np.linalg.pinv(seBWOB, rcond=1e-3), seBWOB.dot(self.coef_) \
                    + np.tensordot(seBW, y[mask] - mu[mask], axes=([0], [0])))
            new_loss = self.get_loss(y, self._link(np.dot(selected_basis, coef_temp)), sample_weight)
            if new_loss - best_loss_irls >= 0:
                break
            self.coef_ = coef_temp
            best_loss_irls = new_loss
        return self
    
    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)