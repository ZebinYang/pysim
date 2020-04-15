import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

from abc import ABCMeta, abstractmethod
from .aspline import ASplineClassifier, ASplineRegressor

from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

utils = importr("devtools")
utils.install_github("https://github.com/vqv/fps")

fps = importr("fps")
numpy2ri.activate()


class BaseSim(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for sim classification and regression.
     """

    @abstractmethod
    def __init__(self, method="first_order", reg_lambda=0.1, reg_gamma=10, 
                 knot_num=20, degree=2, random_state=0):

        self.method = method
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree
        
        self.random_state = random_state

    def _validate_hyperparameters(self):
        
        if self.method not in ["first_order", "second_order", "first_order_thres"]:
            raise ValueError("method must be an element of [first_order, second_order, first_order_thres], got %s." % self.method)
                
        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

        if self.degree < 0:
            raise ValueError("degree must be >= 0, got" % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)

        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if (self.reg_lambda < 0) or (self.reg_lambda > 1):
            raise ValueError("reg_lambda must be >= 0 and <=1, got %s." % self.reg_lambda)

        if self.reg_gamma < 0:
            raise ValueError("reg_gamma must be >= 0, got %s." % self.reg_gamma)

    def _validate_sample_weight(self, n_samples, sample_weight):
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight.ravel() / np.sum(sample_weight)
        return sample_weight

    def _first_order_thres(self, x, y, sample_weight=None, proj_mat=None):

        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        if proj_mat is not None:
            zbar = np.dot(proj_mat, zbar)
        zbar[np.abs(zbar) < self.reg_lambda * np.max(np.abs(zbar))] = 0
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta.reshape([-1, 1])

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
        
        reg_lambda_max = np.max(np.abs(sigmat) - np.abs(sigmat) * np.eye(sigmat.shape[0]), axis=0).max()
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * reg_lambda_max))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * reg_lambda_max))
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

        reg_lambda_max = np.max(np.abs(sigmat) - np.abs(sigmat) * np.eye(sigmat.shape[0]), axis=0).max()
        spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(self.reg_lambda * reg_lambda_max))
        beta = np.array(fps.coef_fps(spca_solver, self.reg_lambda * reg_lambda_max))
        return beta

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
    
    def fit_inner_update(self, x, y, sample_weight=None, proj_mat=None, val_ratio=0.2, max_inner_iter=10,
                  n_inner_iter_no_change=5, max_epoches=100, n_epoch_no_change=5, batch_size=100,
                  learning_rate=1e-3, beta_1=0.9, beta_2=0.999, tol=0.0001, verbose=False):
        
        x, y = self._validate_input(x, y)
        n_samples = x.shape[0]
        batch_size = min(batch_size, n_samples)
        sample_weight = self._validate_sample_weight(n_samples, sample_weight)

        if is_regressor(self):
            idx1, idx2 = train_test_split(np.arange(n_samples),test_size=val_ratio, random_state=self.random_state)
            tr_x, tr_y, val_x, val_y = x[idx1], y[idx1], x[idx2], y[idx2]
        elif is_classifier(self):
            idx1, idx2 = train_test_split(np.arange(n_samples),test_size=val_ratio, stratify=y, random_state=self.random_state)
            tr_x, tr_y, val_x, val_y = x[idx1], y[idx1], x[idx2], y[idx2]

        val_xb = np.dot(val_x, self.beta_)
        if is_regressor(self):
            val_pred = self.shape_fit_.predict(val_xb)
            val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
        elif is_classifier(self):
            val_pred = self.shape_fit_.predict_proba(val_xb)
            val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])

        no_inner_iter_change = 0
        val_loss_inner_iter_best = val_loss
        for inner_iter in range(max_inner_iter):

            m_t = 0 # moving average of the gradient
            v_t = 0 # moving average of the gradient square
            num_updates = 0
            no_epoch_change = 0
            theta_0 = self.beta_ 
            train_size = tr_x.shape[0]
            val_loss_epoch_best = np.inf
            for epoch in range(max_epoches):

                shuffle_index = np.arange(tr_x.shape[0])
                np.random.shuffle(shuffle_index)
                tr_x = tr_x[shuffle_index]
                tr_y = tr_y[shuffle_index]

                for iterations in range(train_size // batch_size):

                    num_updates += 1
                    offset = (iterations * batch_size) % train_size
                    batch_xx = tr_x[offset:(offset + batch_size), :]
                    batch_yy = tr_y[offset:(offset + batch_size)]
                    batch_sample_weight = sample_weight[offset:(offset + batch_size)]

                    xb = np.dot(batch_xx, theta_0)
                    if is_regressor(self):
                        r = batch_yy - self.shape_fit_.predict(xb)
                    elif is_classifier(self):
                        r = batch_yy - self.shape_fit_.predict_proba(xb)
                    
                    # gradient
                    dfxb = self.shape_fit_.diff(xb, order=1)
                    g_t = np.average((- dfxb * r).reshape(-1, 1) * batch_xx, axis=0,
                                weights=batch_sample_weight).reshape(-1, 1)

                    # update the moving average 
                    m_t = beta_1 * m_t + (1 - beta_1) * g_t
                    v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
                    # calculates the bias-corrected estimates
                    m_cap = m_t / (1 - (beta_1 ** (num_updates)))  
                    v_cap = v_t / (1 - (beta_2 ** (num_updates)))
                    # updates the parameters
                    theta_0 = theta_0 - (learning_rate * m_cap) / (np.sqrt(v_cap) + 1e-8)

                # validation loss
                val_xb = np.dot(val_x, theta_0)
                if is_regressor(self):
                    val_pred = self.shape_fit_.predict(val_xb)
                    val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
                elif is_classifier(self):
                    val_pred = self.shape_fit_.predict_proba(val_xb)
                    val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
                # stop criterion
                if val_loss > val_loss_epoch_best - tol:
                    no_epoch_change += 1
                else:
                    no_epoch_change = 0
                if val_loss < val_loss_epoch_best:
                    val_loss_epoch_best = val_loss
                
                if no_epoch_change >= n_epoch_no_change:
                    break
                if verbose:
                    print("Inner iter:", inner_iter + 1, "epoch:", epoch + 1, "with validation loss:", np.round(val_loss, 5))
  
            ## thresholding and normalization
            if proj_mat is not None:
                theta_0 = np.dot(proj_mat, theta_0)

            theta_0[np.abs(theta_0) < self.reg_lambda * np.max(np.abs(theta_0))] = 0
            if np.linalg.norm(theta_0) > 0:
                theta_0 = theta_0 / np.linalg.norm(theta_0)
                if (theta_0[np.abs(theta_0) > 0][0] < 0):
                    theta_0 = - theta_0

            # ridge update
            self.beta_ = theta_0
            tr_xb = np.dot(tr_x, self.beta_)
            self._estimate_shape(tr_xb, tr_y, sample_weight[idx1], xmin=np.min(tr_xb), xmax=np.max(tr_xb))
            
            val_xb = np.dot(val_x, self.beta_)
            if is_regressor(self):
                val_pred = self.shape_fit_.predict(val_xb)
                val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
            elif is_classifier(self):
                val_pred = self.shape_fit_.predict_proba(val_xb)
                val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])

            if val_loss > val_loss_inner_iter_best - tol:
                no_inner_iter_change += 1
            else:
                no_inner_iter_change = 0
            if val_loss < val_loss_inner_iter_best:
                val_loss_inner_iter_best = val_loss
            if no_inner_iter_change >= n_inner_iter_no_change:
                break
            
    def decision_function(self, x):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")
        xb = np.dot(x, self.beta_)
        pred = self.shape_fit_.decision_function(xb)
        return pred

    def visualize(self):

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")

        xlim_min = - max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))
        xlim_max = max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))

        fig = plt.figure(figsize=(12, 4))
        outer = gridspec.GridSpec(1, 2, wspace=0.15)      
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0]) 
        xgrid = np.linspace(self.shape_fit_.xmin, self.shape_fit_.xmax, 100).reshape([-1, 1])
        ygrid = self.shape_fit_.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)
        
        ax1_density = plt.Subplot(fig, inner[1]) 
        xint = ((np.array(self.shape_fit_.bins_[1:]) + np.array(self.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.shape_fit_.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)

        ax2 = plt.Subplot(fig, outer[1]) 
        if len(self.beta_) <= 10:
            rects = ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
            ax2.set_yticks(np.arange(len(self.beta_)))
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(self.beta_.ravel()))][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(self.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        else:
            active_beta = []
            active_beta_idx = []
            for idx, beta in enumerate(self.beta_.ravel()):
                if np.abs(beta) > 0:
                    active_beta.append(beta)
                    active_beta_idx.append(idx)
           
            rects = ax2.barh(np.arange(len(active_beta)), [beta for beta in active_beta][::-1])
            if len (active_beta) > 10:
                input_ticks = np.linspace(0.1 * len(active_beta), len(active_beta) * 0.9, 4).astype(int)
                input_labels = ["X" + str(active_beta_idx[idx] + 1) for idx in input_ticks][::-1] 
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(input_labels)
            else:
                ax2.set_yticks(np.arange(len(active_beta)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in active_beta_idx][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(active_beta_idx))
            ax2.axvline(0, linestyle="dotted", color="black")
        ax2.set_title("Projection Indice", fontsize=12)
        fig.add_subplot(ax2)
        plt.show()


class SimRegressor(BaseSim, RegressorMixin):

    def __init__(self, method="first_order", reg_lambda=0.1, reg_gamma=10, knot_num=20, degree=2, random_state=0):

        super(SimRegressor, self).__init__(method=method,
                                reg_lambda=reg_lambda,
                                reg_gamma=reg_gamma,
                                knot_num=knot_num,
                                degree=degree,
                                random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def _estimate_shape(self, x, y, sample_weight=None, xmin=-1, xmax=1):

        self.shape_fit_ = ASplineRegressor(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                             xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y, sample_weight)

    def predict(self, x):

        pred = self.decision_function(x)
        return pred


class SimClassifier(BaseSim, ClassifierMixin):

    def __init__(self, method="first_order", reg_lambda=0.1, reg_gamma=10, knot_num=20, degree=2, random_state=0):

        super(SimClassifier, self).__init__(method=method,
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
        return x, y.ravel()

    def _estimate_shape(self, x, y, sample_weight=None, xmin=-1, xmax=1):

        #adaptive spline
        self.shape_fit_ = ASplineClassifier(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                         xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y, sample_weight)

    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)
