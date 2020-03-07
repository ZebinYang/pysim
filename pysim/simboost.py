import time 
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

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
    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, random_state=0):

        self.n_estimators = n_estimators
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres
        self.degree = degree
        self.knot_num = knot_num
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.ortho_shrink = ortho_shrink
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

        if not isinstance(self.early_stop_thres, int):
            raise ValueError("early_stop_thres must be an integer, got %s." % self.early_stop_thres)
            
        if self.early_stop_thres <= 0:
            raise ValueError("early_stop_thres must be > 0, got %s." % self.early_stop_thres)

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
        for indice, estimator in enumerate(self.estimators_):

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
        if self.estimators_ is None or len(self.estimators_) == 0:
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
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        return np.array([estimator.beta_.flatten() for estimator in self.estimators_]).T

    def visualize(self):

        check_is_fitted(self, "estimators_")

        max_ids = len(self.estimators_)
        fig = plt.figure(figsize=(12, 4.2 * max_ids))
        outer = gridspec.GridSpec(max_ids, 2, hspace=0.2)
        
        xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        for indice, estimator in enumerate(self.estimators_):

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[indice * 2], wspace=0.15, height_ratios=[6, 1])
            ax1_main = plt.Subplot(fig, inner[0]) 
            xgrid = np.linspace(estimator.shape_fit_.xmin, estimator.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = estimator.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid)
            if indice == 0:
                ax1_main.set_title("Shape Function", fontsize=12)
            ax1_main.text(0.25, 0.9, "IR: " + str(np.round(100 * self.importance_ratios_[indice], 2)) + "%",
                  fontsize=24, horizontalalignment="center", verticalalignment="center", transform=ax1_main.transAxes)
            fig.add_subplot(ax1_main)

            ax1_density = plt.Subplot(fig, inner[1]) 
            xint = ((np.array(estimator.shape_fit_.bins_[1:]) + np.array(estimator.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, estimator.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)

            ax2 = plt.Subplot(fig, outer[indice * 2 + 1]) 
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
            if indice == 0:
                ax2.set_title("Projection Indice", fontsize=12)
            fig.add_subplot(ax2)
        plt.show()

    def decision_function(self, x):

        check_is_fitted(self, "estimators_")

        pred = 0
        for estimator in self.estimators_:
            pred += estimator.predict(x)
        return pred


class SimBoostRegressor(BaseSimBooster, RegressorMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, random_state=0):

        super(SimBoostRegressor, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=ortho_shrink,
                                      random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.reshape([-1, 1])

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

        mse_opt = np.inf
        self.estimators_ = []
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                u, _, _ = np.linalg.svd(self.projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"]}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num,
                                  reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma, random_state=self.random_state), 
                         scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                         cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x, z, sample_weight=sample_weight, proj_mat=proj_mat)
            estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            estimator.fit(x[idx1], z[idx1], sample_weight=sample_weight[idx1], proj_mat=proj_mat)

            # early stop
            pred_val_temp = pred_val + estimator.predict(x[idx2]).reshape([-1, 1])
            mse_new = mean_squared_error(y[idx2], pred_val_temp)
            if mse_opt > mse_new:           
                mse_opt = mse_new
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop_thres:
                break

            # update    
            z = z - estimator.predict(x)
            pred_val += estimator.predict(x[idx2]).reshape([-1, 1])
            self.estimators_.append(estimator)
        
        self.tr_idx = idx1
        self.val_idx = idx2
        self.time_cost_ = time.time() - start
        return self

    def predict(self, x):

        check_is_fitted(self, "estimators_")
        pred = self.decision_function(x)
        return pred


class SimLogitBoostClassifier(BaseSimBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1,
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, random_state=0):

        super(SimLogitBoostClassifier, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=ortho_shrink,
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
        proba_train = 0.5 * np.ones(len(idx1))
        proba_val = 0.5 * np.ones(len(idx2))
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                u, _, _ = np.linalg.svd(self.projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            sample_weight[idx1] = proba_train * (1 - proba_train)
            sample_weight[idx1] /= np.sum(sample_weight[idx1])
            sample_weight[idx1] = np.maximum(sample_weight[idx1], 2 * np.finfo(np.float64).eps)

            with np.errstate(divide="ignore", over="ignore"):
                z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                                -1. / (1. - np.hstack([proba_train, proba_val]))) 
                z = np.clip(z, a_min=-8, a_max=8)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"]}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num,
                                  reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma, random_state=self.random_state), 
                          scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x, z, sample_weight=sample_weight, proj_mat=proj_mat)
            estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            estimator.fit(x[idx1], z[idx1], sample_weight=sample_weight[idx1], proj_mat=proj_mat)

            # stop criterion
            pred_val_temp = pred_val + estimator.predict(x[idx2])
            roc_auc_new = roc_auc_score(y[idx2], 1 / (1 + np.exp(-pred_val_temp)))
            if roc_auc_opt < roc_auc_new:           
                roc_auc_opt = roc_auc_new
                early_stop_count = 0
            else:
                early_stop_count +=1

            if early_stop_count >= self.early_stop_thres:
                break

            # update
            pred_train += estimator.predict(x[idx1])
            pred_val += estimator.predict(x[idx2])
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
            self.estimators_.append(estimator)

        self.tr_idx = idx1
        self.val_idx = idx2
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
                 degree=2, knot_num=20, reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, random_state=0):

        super(SimAdaBoostClassifier, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
                                      early_stop_thres=early_stop_thres,
                                      degree=degree,
                                      knot_num=knot_num,
                                      reg_lambda=reg_lambda,
                                      reg_gamma=reg_gamma,
                                      ortho_shrink=1,
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
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= n_features) or (self.ortho_shrink == 0):
                proj_mat = np.eye(n_features)
            else:
                u, _, _ = np.linalg.svd(self.projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"]}
            grid = GridSearchCV(SimClassifier(degree=self.degree, knot_num=self.knot_num, 
                                   reg_lambda=self.reg_lambda, reg_gamma=self.reg_gamma,
                                   random_state=self.random_state), 
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
            y_coding = y_codes.take([0, 1] == y[idx1])
            with np.errstate(divide="ignore", over="ignore"):
                pred_proba = estimator.predict_proba(x[idx1])
                pred_proba = np.clip(pred_proba, np.finfo(pred_proba.dtype).eps, 1 - np.finfo(pred_proba.dtype).eps)
                estimator_weight = -0.5 * np.sum(y_coding * np.log(np.vstack([1 - pred_proba, pred_proba])).T, axis=1)
                sample_weight[idx1] *= np.exp(estimator_weight * ((sample_weight[idx1] > 0) | (estimator_weight < 0)))

            sample_weight[idx1] /= sample_weight[idx1].sum()
            log_pred_proba_val = np.log(pred_proba[idx2])
            pred_val = self.decision_function(x[idx2]) + (log_pred_proba_val - (1. / 2) * log_pred_proba_val.sum(axis=1)[:, np.newaxis])
            pred_val_proba = 1 / (1 + np.exp(- pred_val))
            roc_auc_new = roc_auc_score(y[idx2], pred_val_proba)
            # stop criterion
            if roc_auc_opt < roc_auc_new:           
                roc_auc_opt = roc_auc_new
                early_stop_count = 0
            else:
                early_stop_count +=1

            if early_stop_count >= self.early_stop_thres:
                break
            
            # update
            self.estimators_.append(estimator)
        
        self.tr_idx = idx1
        self.val_idx = idx2
        self.time_cost_ = time.time() - start
        return self
    
    def decision_function(self, x):

        pred = 0
        for estimator in self.estimators_:
            pred_proba = estimator.predict_proba(x)
            pred_proba = np.clip(pred_proba, np.finfo(pred_proba.dtype).eps, None)
            log_proba = np.log(pred_proba)
            pred += (log_proba - (1. / 2) * log_proba.sum(axis=1)[:, np.newaxis])
        return pred

    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)