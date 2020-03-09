import time 
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from abc import ABCMeta, abstractmethod
from sklearn.utils.extmath import softmax
from sklearn.utils.extmath import stable_cumsum
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score

from .pysim import SimRegressor, SimClassifier


class BaseSimBooster(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for sim classification and regression.
     """

    @abstractmethod
    def __init__(self, n_estimators, val_ratio=0.2, degree=2, knot_num=20, ortho_shrink=1, loss_threshold=0.01, 
                 reg_lambda=0.1, reg_gamma=10, random_state=0):

        self.n_estimators = n_estimators
        self.val_ratio = val_ratio
        self.ortho_shrink = ortho_shrink
        self.loss_threshold = loss_threshold
    
        self.degree = degree
        self.knot_num = knot_num
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma

        self.random_state = random_state

    def _validate_hyperparameters(self):

        if not isinstance(self.n_estimators, int):
            raise ValueError("n_estimators must be an integer, got %s." % self.n_estimators)

        if self.n_estimators < 0:
            raise ValueError("n_estimators must be >= 0, got" % self.n_estimators)

        if self.val_ratio <= 0:
            raise ValueError("val_ratio must be > 0, got" % self.val_ratio)

        if self.val_ratio >= 1:
            raise ValueError("val_ratio must be < 1, got %s." % self.val_ratio)

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

        if self.degree < 0:
            raise ValueError("degree must be >= 0, got" % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)

        if self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if isinstance(self.reg_lambda, list):
            for val in self.reg_lambda:
                if val < 0:
                    raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda_list = self.reg_lambda  
        elif isinstance(self.reg_lambda, float):
            if self.reg_lambda < 0:
                raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda_list = [self.reg_lambda]

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma  
        elif isinstance(self.reg_gamma, float):
            if self.reg_gamma < 0:
                raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]

    @property
    def importance_ratios_(self):
        """Return the estimator importance ratios (the higher, the more important the feature).
        Returns
        -------
        importance_ratios_ : ndarray of shape (n_estimators,)
            The estimator importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")
        estimator_importance = []
        for indice, estimator in enumerate(self.best_estimators_):

            xgrid = np.linspace(estimator.shape_fit_.xmin, estimator.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = estimator.shape_fit_.decision_function(xgrid)
            estimator_importance.append(np.std(ygrid))
        importance_ratio = estimator_importance / np.sum(estimator_importance)
        return importance_ratio

    @property
    def orthogonality_measure_(self):
        """Return the orthogonality measure (the lower, the better).
        Returns
        -------
        orthogonality_measure_ : float scalar
        """
        if self.best_estimators_ is None or len(self.best_estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")
            
        ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                  self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
        if self.projection_indices_.shape[1] > 1:
            ortho_measure /= ((self.projection_indices_.shape[1] ** 2 - self.projection_indices_.shape[1]))
        else:
            ortho_measure = np.nan
        return ortho_measure

    @property
    def projection_indices_(self):
        """Return the projection indices.
        Returns
        -------
        projection_indices_ : ndarray of shape (d, n_estimators)
        """
        if self.best_estimators_ is None or len(self.best_estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        return np.array([estimator.beta_.flatten() for estimator in self.best_estimators_]).T

    def visualize(self, cols_per_row=3):

        check_is_fitted(self, "best_estimators_")

        max_ids = len(self.best_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)

        xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        for indice, estimator in enumerate(self.best_estimators_):

            inner = outer[indice].subgridspec(2, 2, wspace=0.15, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = np.linspace(estimator.shape_fit_.xmin, estimator.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = estimator.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid)
            ax1_main.set_xticklabels([])
            ax1_main.set_title("          Component " + str(indice + 1) +
                               " (IR: " + str(np.round(100 * self.importance_ratios_[indice], 2)) + "%)", fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(estimator.shape_fit_.bins_[1:]) + np.array(estimator.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, estimator.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)

            ax2 = fig.add_subplot(inner[:, 1])
            if len(estimator.beta_) <= 10:
                rects = ax2.barh(np.arange(len(estimator.beta_)), [beta for beta in estimator.beta_.ravel()][::-1])
                ax2.set_yticks(np.arange(len(estimator.beta_)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(estimator.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(estimator.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                active_beta = []
                active_beta_idx = []
                for idx, beta in enumerate(estimator.beta_.ravel()):
                    if np.abs(beta) > 0:
                        active_beta.append(beta)
                        active_beta_idx.append(idx)
                rects = ax2.barh(np.arange(len(active_beta)), [beta for beta in active_beta][::-1])
                ax2.set_yticks(np.arange(len(active_beta)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in active_beta_idx][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(active_beta_idx))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
        plt.show()
    
    def decision_function(self, x):

        check_is_fitted(self, "best_estimators_")

        pred = 0
        for estimator in self.best_estimators_:
            pred += estimator.predict(x)
        return pred


class SimBoostRegressor(BaseSimBooster, RegressorMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, loss_threshold=0.01, random_state=0):

        super(SimBoostRegressor, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=ortho_shrink,
                                      loss_threshold=loss_threshold,
                                      random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.reshape([-1, 1])

    def visualize_val_perf(self):

        check_is_fitted(self, "best_estimators_")

        fig = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(self.estimator_val_mse) + 1, 1), self.estimator_val_mse)
        plt.axvline(np.argmin(self.estimator_val_mse) + 1, linestyle="dotted", color="red")
        plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
        plt.plot(np.argmin(self.estimator_val_mse) + 1, np.min(self.estimator_val_mse), "*", markersize=12, color="red")
        plt.plot(len(self.best_estimators_), self.estimator_val_mse[len(self.best_estimators_) - 1], "o", markersize=8, color="red")
        plt.xlabel("Number of Estimators", fontsize=12)
        plt.ylabel("Validation MSE", fontsize=12)
        plt.xlim(0.5, len(self.estimator_val_mse) + 0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def fit(self, x, y, sample_weight=None):

        start = time.time()
        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)

        indices = np.arange(n_samples)
        idx1, idx2 = train_test_split(indices, test_size=self.val_ratio, random_state=self.random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        pred_val = 0
        z = y.copy().ravel()
        self.estimators_ = []
        self.estimator_val_mse = []
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                projection_indices_ = np.array([estimator.beta_.flatten() for estimator in self.estimators_]).T
                u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"], 
                      "reg_lambda": self.reg_lambda_list,
                      "reg_gamma": self.reg_lambda_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                         scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                         cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x, z, sample_weight=sample_weight, proj_mat=proj_mat)
            estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            estimator.fit(x[idx1], z[idx1], sample_weight=sample_weight[idx1], proj_mat=proj_mat)

            # update    
            z = z - estimator.predict(x)
            pred_val += estimator.predict(x[idx2])
            val_loss = mean_squared_error(y[idx2], pred_val)
            self.estimators_.append(estimator)
            self.estimator_val_mse.append(val_loss)
       
        self.tr_idx = idx1
        self.val_idx = idx2
        best_loss = np.min(self.estimator_val_mse)
        if np.sum((self.estimator_val_mse / best_loss - 1) < self.loss_threshold) > 0:
            best_idx = np.where((self.estimator_val_mse / best_loss - 1) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmin(self.estimator_val_mse)
        self.best_estimators_ = self.estimators_[:(best_idx + 1)]
        self.time_cost_ = time.time() - start
        return self

    def predict(self, x):

        check_is_fitted(self, "best_estimators_")
        pred = self.decision_function(x)
        return pred


class SimLogitBoostClassifier(BaseSimBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, loss_threshold=0.01, random_state=0):

        super(SimLogitBoostClassifier, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=ortho_shrink,
                                      loss_threshold=loss_threshold,
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
   
    def visualize_val_perf(self):

        check_is_fitted(self, "best_estimators_")

        fig = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(self.estimator_val_auc) + 1, 1), self.estimator_val_auc)
        plt.axvline(np.argmax(self.estimator_val_auc) + 1, linestyle="dotted", color="red")
        plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
        plt.plot(np.argmax(self.estimator_val_auc) + 1, np.max(self.estimator_val_auc), "*", markersize=12, color="red")
        plt.plot(len(self.best_estimators_), self.estimator_val_auc[len(self.best_estimators_) - 1], "o", markersize=8, color="red")
        plt.xlabel("Number of Estimators", fontsize=12)
        plt.ylabel("Validation AUC", fontsize=12)
        plt.xlim(0.5, len(self.estimator_val_auc) + 0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def fit(self, x, y, sample_weight=None):

        start = time.time()
        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)

        indices = np.arange(n_samples)
        idx1, idx2 = train_test_split(indices, test_size=self.val_ratio, random_state=self.random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        pred_val = 0
        pred_train = 0 
        roc_auc_opt = -np.inf
        self.estimators_ = []
        self.estimator_val_auc = []
        proba_train = 0.5 * np.ones(len(idx1))
        proba_val = 0.5 * np.ones(len(idx2))
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                projection_indices_ = np.array([estimator.beta_.flatten() for estimator in self.estimators_]).T
                u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            sample_weight[idx1] = proba_train * (1 - proba_train)
            sample_weight[idx1] /= np.sum(sample_weight[idx1])
            sample_weight[idx1] = np.maximum(sample_weight[idx1], 2 * np.finfo(np.float64).eps)

            with np.errstate(divide="ignore", over="ignore"):
                z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                                -1. / (1. - np.hstack([proba_train, proba_val]))) 
                z = np.clip(z, a_min=-8, a_max=8)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"], 
                      "reg_lambda": self.reg_lambda_list,
                      "reg_gamma": self.reg_lambda_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                          scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)

            grid.fit(x, z, sample_weight=sample_weight, proj_mat=proj_mat)
            estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            estimator.fit(x[idx1], z[idx1], sample_weight=sample_weight[idx1], proj_mat=proj_mat)

            # update
            pred_train += estimator.predict(x[idx1])
            pred_val += estimator.predict(x[idx2])
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

            val_auc = roc_auc_score(y[idx2], proba_val)
            self.estimators_.append(estimator)
            self.estimator_val_auc.append(val_auc)
       
        self.tr_idx = idx1
        self.val_idx = idx2
        best_auc = np.max(self.estimator_val_auc)
        if np.sum((1 - self.estimator_val_auc / best_auc) < self.loss_threshold) > 0:
            best_idx = np.where((1 - self.estimator_val_auc / best_auc) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmax(self.estimator_val_auc)
        self.best_estimators_ = self.estimators_[:(best_idx + 1)]
        self.time_cost_ = time.time() - start
        return self
    
    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba).reshape([-1, 1])


class SimAdaBoostClassifier(BaseSimBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, loss_threshold=0.01, random_state=0):

        super(SimAdaBoostClassifier, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=ortho_shrink,
                                      loss_threshold=loss_threshold,
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

    def visualize_val_perf(self):

        check_is_fitted(self, "best_estimators_")

        fig = plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(self.estimator_val_auc) + 1, 1), self.estimator_val_auc)
        plt.axvline(np.argmax(self.estimator_val_auc) + 1, linestyle="dotted", color="red")
        plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
        plt.plot(np.argmax(self.estimator_val_auc) + 1, np.max(self.estimator_val_auc), "*", markersize=12, color="red")
        plt.plot(len(self.best_estimators_), self.estimator_val_auc[len(self.best_estimators_) - 1], "o", markersize=8, color="red")
        plt.xlabel("Number of Estimators", fontsize=12)
        plt.ylabel("Validation AUC", fontsize=12)
        plt.xlim(0.5, len(self.estimator_val_auc) + 0.5)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

    def fit(self, x, y, sample_weight=None):

        start = time.time()
        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)

        indices = np.arange(n_samples)
        idx1, idx2 = train_test_split(indices, test_size=self.val_ratio, random_state=self.random_state)
        val_fold = np.ones((len(indices)))
        val_fold[idx1] = -1

        roc_auc_opt = -np.inf
        self.estimators_ = []
        self.estimator_val_auc = []
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                projection_indices_ = np.array([estimator.beta_.flatten() for estimator in self.estimators_]).T
                u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"], 
                      "reg_lambda": self.reg_lambda_list,
                      "reg_gamma": self.reg_lambda_list}
            grid = GridSearchCV(SimClassifier(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                          scoring={"auc": make_scorer(roc_auc_score)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x, y, sample_weight=sample_weight, proj_mat=proj_mat)
            estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_auc"] == 1))[0][0]])
            estimator.fit(x[idx1], y[idx1], sample_weight=sample_weight[idx1], proj_mat=proj_mat)

            # Instances incorrectly classified
            y_predict = estimator.predict(x[idx1])
            estimator_error = np.mean(np.average(y_predict != y[idx1], weights=sample_weight[idx1], axis=0))
            if estimator_error <= 0:
                break

            y_codes = np.array([-1., 1.])
            y_coding = y_codes.take(np.array([0, 1]) == y[idx1].reshape([-1, 1]))
            with np.errstate(divide="ignore", over="ignore"):
                pred_proba_tr = estimator.predict_proba(x[idx1])
                pred_proba_tr = np.clip(pred_proba_tr, np.finfo(pred_proba_tr.dtype).eps, 1 - np.finfo(pred_proba_tr.dtype).eps)
                estimator_weight = -0.5 * np.sum(y_coding * np.log(np.vstack([1 - pred_proba_tr, pred_proba_tr])).T, axis=1)
                sample_weight[idx1] *= np.exp(estimator_weight * ((sample_weight[idx1] > 0) | (estimator_weight < 0)))
                sample_weight[idx1] /= sample_weight[idx1].sum()

            pred_val = 0
            for est in self.estimators_ + [estimator]:
                pred_proba_val = est.predict_proba(x[idx2])
                pred_proba_val = np.clip(pred_proba_val, np.finfo(pred_proba_val.dtype).eps, 1 - np.finfo(pred_proba_val.dtype).eps)
                log_pred_proba_val = np.log(np.vstack([1 - pred_proba_val, pred_proba_val])).T
                pred_val += (log_pred_proba_val[:, 1] - (1. / 2) * log_pred_proba_val.sum(axis=1))
            proba_val = 1 / (1 + np.exp(- pred_val))
            val_auc = roc_auc_score(y[idx2], proba_val)
            self.estimators_.append(estimator)
            self.estimator_val_auc.append(val_auc)
       
        self.tr_idx = idx1
        self.val_idx = idx2
        best_auc = np.max(self.estimator_val_auc)
        if np.sum((1 - self.estimator_val_auc / best_auc) < self.loss_threshold) > 0:
            best_idx = np.where((1 - self.estimator_val_auc / best_auc) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmax(self.estimator_val_auc)
        self.best_estimators_ = self.estimators_[:(best_idx + 1)]
        self.time_cost_ = time.time() - start
        return self
    
    def decision_function(self, x):

        pred = 0
        for estimator in self.best_estimators_:
            pred_proba = estimator.predict_proba(x)
            pred_proba = np.clip(pred_proba, np.finfo(pred_proba.dtype).eps, 1 - np.finfo(pred_proba.dtype).eps)
            log_proba = np.log(np.vstack([1 - pred_proba, pred_proba])).T
            pred += (log_proba[:, 1] - (1. / 2) * log_proba.sum(axis=1))
        return pred

    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)