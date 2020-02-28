import scipy
import numpy as np
import pandas as pd 
from scipy.linalg import cholesky
from matplotlib import pyplot as plt

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from patsy import dmatrix, build_design_matrices

def diff_matrix(order, knot_num):
    results = [] # a container to collect the rows
    n_rows = order + 2
    for _ in range(n_rows): 
        row = [1] # a starter 1 in the row
        if results: # then we're in the second row or beyond
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

    
class ASplineRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, knot_num=100, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, epsilon=0.00001, threshold=0.99, maxiter=10):

        self.knot_num = knot_num
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        self.epsilon = epsilon
        self.threshold = threshold
        self.maxiter = maxiter

    def fit(self, x, y, sample_weight=None):

        n_samples = x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight *= n_samples
        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1",
                   {"x": [self.xmin, self.xmax], "knots": knots, "degree": self.degree})
        basis = np.asarray(build_design_matrices([xphi.design_info],
                   {"x": x, "knots": knots, "degree": self.degree})[0])
        D = diff_matrix(self.degree, self.knot_num)
        w = np.ones([self.knot_num], dtype=np.float32) 
        W = np.diag(w)

        BB = basis.T.dot(np.diag(sample_weight)).dot(basis)
        BY = basis.T.dot(np.diag(sample_weight)).dot(y)
        for i in range(self.maxiter):
            U = cholesky(BB + self.reg_gamma * np.dot(np.dot(D.T, W), D))
            M = scipy.linalg.lapack.clapack.dtrtri(U)[0]
            update_a = np.dot(np.dot(M, M.T.conj()), BY)
            # The original implementation of matrix inversion is very slow and so it is commented. 
            # update_a = np.dot(np.linalg.inv(np.dot(basis.T, basis) + self.reg_gamma * np.dot(np.dot(D.T, W), D)), np.dot(basis.T, y))
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)
            W = np.diag(update_w.reshape([-1]))

        self.selected_knots_ = list(np.array(knots)[np.reshape(update_w * np.dot(D, update_a) ** 2 > self.threshold, [-1])])
        self.selected_xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1", 
               {"x": [self.xmin, self.xmax], "knots": self.selected_knots_, "degree": self.degree})
        selected_basis = np.asarray(build_design_matrices([self.selected_xphi.design_info],
                          {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])
        self.coef_ = np.dot(np.linalg.pinv(selected_basis.T.dot(np.diag(sample_weight)).dot(selected_basis),
                      selected_basis.T.dot(np.diag(sample_weight)).dot(y))
        return self

    def predict(self, x):

        check_is_fitted(self, "coef_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        design_matrix = np.asarray(build_design_matrices([self.selected_xphi.design_info],
                                  {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])
        pred = np.dot(design_matrix, self.coef_)
        return pred
    

class ASplineClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, knot_num=100, reg_gamma=0.1, xmin=-1, xmax=1, degree=2, epsilon=0.00001, threshold=0.99,
                 maxiter=10, maxiter_irls=20):

        self.knot_num = knot_num
        self.reg_gamma = reg_gamma
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        self.epsilon = epsilon
        self.threshold = threshold
        self.maxiter = maxiter
        self.maxiter_irls = maxiter_irls
        self.EPS = 10**(-5)

    def link(self, x):
        return 1 / (1 + np.exp(-x))

    def inv_link(self, x):
        return np.log(x) - np.log(1 - x)
    
    def fit(self, x, y):

        knots = list(np.linspace(self.xmin, self.xmax, self.knot_num + 2, dtype=np.float32)[1:-1])
        xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1",
                       {"x": [self.xmin, self.xmax], "knots": knots, "degree": self.degree})
        init_basis = np.asarray(build_design_matrices([xphi.design_info],
                          {"x": x, "knots": knots, "degree": self.degree})[0])
        D = diff_matrix(self.degree, self.knot_num)
        w = np.ones([self.knot_num], dtype=np.float32) 
        W = np.diag(w)

        tempy = y.copy()
        tempy[tempy==0] = 0.01
        tempy[tempy==1] = 0.99
        update_a = np.dot(np.linalg.pinv(np.dot(init_basis.T, init_basis)), np.dot(init_basis.T, self.inv_link(tempy)))

        for i in range(self.maxiter):
            tempy = y.copy()
            basis = init_basis.copy()
            # The original implementation of matrix inversion is very slow and so it is commented. 
            for j in range(self.maxiter_irls):
                lp = np.dot(basis, update_a)
                mu = self.link(lp)
                omega = mu * (1 - mu)
                mask = (np.abs(omega) >= self.EPS) * np.isfinite(omega)
                mask = mask.ravel()
                if np.sum(mask) == 0:
                    break

                tempy = tempy[mask] # update
                lp = lp[mask] # update
                mu = mu[mask] # update
                omega = np.diag(omega[mask]) # update
                basis = basis[mask,:]
                left_ = np.linalg.pinv(basis.T.dot(omega).dot(basis) + self.reg_gamma * D.T.dot(W).dot(D))
                right = basis.T.dot(omega.dot(basis).dot(update_a) + tempy - mu)
                update_a = left_.dot(right)
            update_w = 1 / (np.dot(D, update_a) ** 2 + self.epsilon ** 2)
            W = np.diag(update_w.reshape([-1]))

        self.selected_knots_ = list(np.array(knots)[np.reshape(update_w * np.dot(D, update_a) ** 2 > self.threshold, [-1])])
        self.selected_xphi = dmatrix("bs(x, knots = knots, degree=degree, include_intercept=True) - 1", 
               {"x": [self.xmin, self.xmax], "knots": self.selected_knots_, "degree": self.degree})
        selected_basis = np.asarray(build_design_matrices([self.selected_xphi.design_info],
                          {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])

        tempy = y.copy()
        basis = selected_basis.copy()
        tempy[tempy==0] = 0.01
        tempy[tempy==1] = 0.99
        self.coef_ = np.dot(np.linalg.pinv(np.dot(basis.T, basis)), np.dot(basis.T, self.inv_link(tempy)))
        for j in range(self.maxiter_irls):
            lp = np.dot(basis, self.coef_)
            mu = self.link(lp)
            omega = mu * (1 - mu)
            mask = (np.abs(omega) >= self.EPS) * np.isfinite(omega)
            mask = mask.ravel()
            if np.sum(mask) == 0:
                break
            tempy = tempy[mask] # update
            lp = lp[mask] # update
            mu = mu[mask] # update
            omega = np.diag(omega[mask]) # update
            basis = basis[mask,:]
            left_ = np.linalg.pinv(basis.T.dot(omega).dot(basis))
            right = basis.T.dot(omega.dot(basis).dot(self.coef_) + tempy - mu)
            self.coef_ = left_.dot(right)
        return self
    
    def decision_function(self, x):

        check_is_fitted(self, "coef_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        design_matrix = np.asarray(build_design_matrices([self.selected_xphi.design_info],
                                  {"x": x, "knots": self.selected_knots_, "degree": self.degree})[0])
        pred = np.dot(design_matrix, self.coef_)
        return pred

    def predict(self, x):

        check_is_fitted(self, "coef_")
        return self.decision_function(x) > 0