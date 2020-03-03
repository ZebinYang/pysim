import time 
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score

from .pysim import SIMRegressor, SIMClassifier


class BaseSIMBooster(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for sim classification and regression.
     """

    @abstractmethod
    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1, random_state=0):

        self.val_ratio = val_ratio
        self.n_estimators = n_estimators
        self.early_stop_thres = early_stop_thres
        
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

    def visualize(self):

        check_is_fitted(self, "sim_estimators_")

        idx = 0
        max_ids = len(self.sim_estimators_)
        fig = plt.figure(figsize=(12, 4.2 * max_ids))
        outer = gridspec.GridSpec(max_ids, 1, hspace=0.2)
        for indice, model in enumerate(self.sim_estimators_):

            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[indice], wspace=0.15)
            ax1 = plt.Subplot(fig, inner[0]) 
            xgrid = np.linspace(model.shape_fit_.xmin, model.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = model.shape_fit_.predict(xgrid)
            ax1.plot(xgrid, ygrid)
            if indice == 0:
                ax1.set_title("Shape Function", fontsize=12)
            ax1.text(0.25, 0.9, 'IR: ' + str(np.round(100 * self.importance_ratio_[indice], 2)) + "%",
                  fontsize=24, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1]) 
            active_beta = []
            active_beta_inx = []
            for idx, beta in enumerate(model.beta_.ravel()):
                if np.abs(beta) > 0:
                    active_beta.append(beta)
                    active_beta_inx.append(idx)

            rects = ax2.barh(np.arange(len(active_beta)),
                        [model.beta_.ravel()[idx] for _, idx in sorted(zip(np.abs(active_beta), active_beta_inx))])
            ax2.set_yticks(np.arange(len(active_beta)))
            ax2.set_yticklabels(["X" + str(idx + 1) for _, idx in sorted(zip(np.abs(active_beta), active_beta_inx))])
            ax2.set_xlim(np.min(active_beta) - 0.1, np.max(active_beta) + 0.1)
            ax2.set_ylim(-1, len(active_beta_inx))
            if indice == 0:
                ax2.set_title("Projection Indice", fontsize=12)
            fig.add_subplot(ax2)
        plt.show()

    def _predict(self, x):

        check_is_fitted(self, "sim_estimators_")
        
        pred = 0
        for sim_clf in self.sim_estimators_:
            pred += sim_clf.predict(x)
        return pred


class SIMBoostRegressor(BaseSIMBooster, RegressorMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1, random_state=0):

        super(SIMBoostRegressor, self).__init__(n_estimators=n_estimators,
                                   val_ratio=val_ratio,
                                   early_stop_thres=early_stop_thres,
                                   random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)
        return x, y

    def fit(self, x, y, sample_weight=None):

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

        pred_train = 0 
        pred_val = 0
        z = y.ravel().copy()

        mse_opt = np.inf
        self.time_cost_ = 0
        self.sim_estimators_ = []
        self.sim_importance_ = []
        for i in range(self.n_estimators):

            # fit SIM model
            param_grid = {"method": ["second_order", 'first_order']}
            grid = GridSearchCV(SIMRegressor(degree=2, knot_num=20, spline="a_spline", reg_lambda=0.1, reg_gamma=10,
                                  random_state=self.random_state), 
                         scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                         cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            # time
            start = time.time()
            grid.fit(x, z, sample_weight=sample_weight)
            model = grid.estimator.set_params(**grid.cv_results_['params'][np.where((grid.cv_results_['rank_test_mse'] == 1))[0][0]])
            model.fit(x[idx1, :], z[idx1], sample_weight=sample_weight[idx1])
            self.time_cost_ += time.time() - start

            # early stop
            pred_val_temp = pred_val + model.predict(x[idx2, :]).reshape([-1, 1])
            mse_new = mean_squared_error(y[idx2], pred_val_temp)
            if mse_opt > mse_new:           
                mse_opt = mse_new
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop_thres:
                break

            # update    
            pred_train += model.predict(x[idx1, :]).reshape([-1, 1])
            pred_val += model.predict(x[idx2, :]).reshape([-1, 1])
            z = z - model.predict(x)
            
            self.sim_estimators_.append(model)
            xgrid = np.linspace(model.shape_fit_.xmin, model.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = model.shape_fit_.predict(xgrid)
            self.sim_importance_.append(np.std(ygrid))
        
        self.tr_idx = idx1
        self.val_idx = idx2
        self.importance_ratio_ = self.sim_importance_ / np.sum(self.sim_importance_)
        self.betas_ = np.array([model.beta_.flatten() for model in self.sim_estimators_])
        self.ortho_measure_ = np.linalg.norm(np.dot(self.betas_, self.betas_.T) - np.eye(self.betas_.shape[0]))

        return self

    def predict(self, x):

        check_is_fitted(self, "sim_estimators_")
        pred = self._predict(x)
        return pred


class SIMLogitBoostClassifier(BaseSIMBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1, random_state=0):

        super(SIMLogitBoostClassifier, self).__init__(n_estimators=n_estimators,
                                       val_ratio=val_ratio,
                                       early_stop_thres=early_stop_thres,
                                       random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y

    def fit(self, x, y, sample_weight=None):

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

        pred_train = 0 
        pred_val = 0
        probs = 0.5 * np.ones(n_samples)
        sample_weight = 1 / n_samples * np.ones(n_samples)

        roc_auc_opt = -np.inf
        self.time_cost_ = 0
        self.sim_estimators_ = []
        self.sim_importance_ = []
        for i in range(self.n_estimators):

            sample_weight = probs * (1 - probs)
            sample_weight /= np.sum(sample_weight)
            sample_weight = np.maximum(sample_weight, 2 * np.finfo(np.float64).eps)

            with np.errstate(divide='ignore', over='ignore'):
                z = np.where(y.ravel(), 1. / probs, -1. / (1. - probs)) 
                z = np.clip(z, a_min=-8, a_max=8)

            # fit SIM model
            param_grid = {"method": ["second_order", 'first_order']}
            grid = GridSearchCV(SIMRegressor(degree=2, knot_num=20, spline="a_spline", reg_lambda=0.1, reg_gamma=10,
                                  random_state=self.random_state), 
                          scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            # time
            start = time.time()
            grid.fit(x, z, sample_weight=sample_weight)
            model = grid.estimator.set_params(**grid.cv_results_['params'][np.where((grid.cv_results_['rank_test_mse'] == 1))[0][0]])
            model.fit(x[idx1, :], z[idx1], sample_weight=sample_weight[idx1])
            self.time_cost_ += time.time() - start

            # stop criterion
            pred_val_temp = pred_val + 0.5 * model.predict(x[idx2, :])
            roc_auc_new = roc_auc_score(y[idx2], 1 / (1 + np.exp(-2 * pred_val_temp)))
            if roc_auc_opt < roc_auc_new:           
                roc_auc_opt = roc_auc_new
                early_stop_count = 0
            else:
                early_stop_count +=1

            if early_stop_count >= self.early_stop_thres:
                break

            # update
            pred_train += 0.5 * model.predict(x[idx1, :])
            pred_val += 0.5 * model.predict(x[idx2, :])
            probs = 1 / (1 + np.exp(-2 * np.hstack([pred_train, pred_val])))
            self.sim_estimators_.append(model)
            xgrid = np.linspace(model.shape_fit_.xmin, model.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = model.shape_fit_.predict(xgrid)
            self.sim_importance_.append(np.std(ygrid))
        
        self.tr_idx = idx1
        self.val_idx = idx2
        self.importance_ratio_ = self.sim_importance_ / np.sum(self.sim_importance_)
        self.betas_ = np.array([model.beta_.flatten() for model in self.sim_estimators_])
        self.ortho_measure_ = np.linalg.norm(np.dot(self.betas_, self.betas_.T) - np.eye(self.betas_.shape[0]))

        return self
    
    def predict_proba(self, x):

        pred_proba_inv = 0.5 * self._predict(x)
        pred_prob = 1 / (1 + np.exp(-2 * pred_proba_inv))
        return pred_prob

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)
    

class SIMAdaBoostClassifier(BaseSIMBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, early_stop_thres=1, random_state=0):

        super(SIMAdaBoostClassifier, self).__init__(n_estimators=n_estimators,
                                       val_ratio=val_ratio,
                                       early_stop_thres=early_stop_thres,
                                       random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y

    def fit(self, x, y, sample_weight=None):

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

        pred_train = 0 
        pred_val = 0
        probs = 0.5 * np.ones(n_samples)
        sample_weight = 1 / n_samples * np.ones(n_samples)

        roc_auc_opt = -np.inf
        self.time_cost_ = 0
        self.sim_estimators_ = []
        self.sim_importance_ = []
        for i in range(self.n_estimators):

            sample_weight = probs * (1 - probs)
            sample_weight /= np.sum(sample_weight)
            sample_weight = np.maximum(sample_weight, 2 * np.finfo(np.float64).eps)

            # fit SIM model
            param_grid = {"method": ["second_order", 'first_order']}
            grid = GridSearchCV(SIMClassifier(degree=2, knot_num=20, spline="a_spline", reg_lambda=0.1, reg_gamma=10,
                                   random_state=self.random_state), 
                          scoring={"auc": make_scorer(roc_auc_score)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            # time
            start = time.time()
            grid.fit(x, y, sample_weight=sample_weight)
            model = grid.estimator.set_params(**grid.cv_results_['params'][np.where((grid.cv_results_['rank_test_mse'] == 1))[0][0]])
            model.fit(x[idx1, :], y[idx1], sample_weight=sample_weight[idx1])
            self.time_cost_ += time.time() - start

            estimator_error = np.average((model.predict(x[idx1, :]) != y[idx1]), axis=0, weights=sample_weight)
            if estimator_error <= 0:
                break
            
            estimator_weight = np.log((1 − estimator_error) / estimator_error)
            pred_val_temp = pred_val + model.predict(x[idx2, :])
            roc_auc_new = roc_auc_score(y[idx2], 1 / (1 + np.exp(-2 * pred_val_temp)))

            sample_weight = sample_weight * np.exp(estimator_weight * (pred_train != y[idx1]))
            # stop criterion
            if roc_auc_opt < roc_auc_new:           
                roc_auc_opt = roc_auc_new
                early_stop_count = 0
            else:
                early_stop_count +=1

            if early_stop_count >= self.early_stop_thres:
                break

            # update
            self.sim_estimators_.append(model)
            xgrid = np.linspace(model.shape_fit_.xmin, model.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = model.shape_fit_.predict(xgrid)
            self.sim_importance_.append(np.std(ygrid))
        
        self.tr_idx = idx1
        self.val_idx = idx2
        self.importance_ratio_ = self.sim_importance_ / np.sum(self.sim_importance_)
        self.betas_ = np.array([model.beta_.flatten() for model in self.sim_estimators_])
        self.ortho_measure_ = np.linalg.norm(np.dot(self.betas_, self.betas_.T) - np.eye(self.betas_.shape[0]))

        return self
    
    def predict_proba(self, x):

        pred_proba_inv = 0.5 * self._predict(x)
        pred_prob = 1 / (1 + np.exp(-2 * pred_proba_inv))
        return pred_prob

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)