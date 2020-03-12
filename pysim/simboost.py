import time 
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from abc import ABCMeta, abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax, stable_cumsum
from sklearn.utils import check_array, check_X_y, column_or_1d
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer, FunctionTransformer, OneHotEncoder
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

from pysim import SimRegressor, SimClassifier


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
                             "call `fit` before `importance_ratios_`.")
        estimator_importance = []
        for indice, pipe in enumerate(self.best_estimators_):
            estimator = pipe["sim"]
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
                             "call `fit` before `orthogonality_measure_`.")
            
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
                             "call `fit` before `projection_indices_`.")

        return np.array([pipe["sim_estimator"].beta_.flatten() for pipe in self.best_estimators_]).T
    
    def _validate_sample_weight(self, n_samples, sample_weight):
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)
        return sample_weight
    
    def _preprocess_meta_info(self, n_features, meta_info):
        
        if meta_info is None:
            meta_info = {}
            for i in range(n_features):
                meta_info.update({"X" + str(i + 1):{"type":"continuous"}})
            meta_info.update({"Y":{"type":"target"}})
        
        if not isinstance(meta_info, dict):
            raise ValueError("meta_info must be None or a dict, got" % meta_info)
        else:
            self.cvalues_ = {}  
            self.cfeature_num_ = 0
            self.nfeature_num_ = 0
            self.cfeature_list_ = []
            self.nfeature_list_ = []
            self.cfeature_index_list_ = []
            self.nfeature_index_list_ = []
            for idx, (feature_name, feature_info) in enumerate(meta_info.items()):
                if feature_info["type"] == "target":
                    continue
                if feature_info["type"] == "categorical":
                    self.cfeature_num_ += 1
                    self.cfeature_list_.append(feature_name)
                    self.cfeature_index_list_.append(idx)
                    self.cvalues_.update({feature_name:meta_info[feature_name]["values"]})
                else:
                    self.nfeature_num_ +=1
                    self.nfeature_list_.append(feature_name)
                    self.nfeature_index_list_.append(idx)
            if n_features != (self.cfeature_num_ + self.nfeature_num_):
                raise ValueError("meta_info and n_features mismatch!")

    def validation_performance(self):

        check_is_fitted(self, "best_estimators_")

        if is_regressor(self):
            fig = plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, len(self.val_mse_) + 1, 1), self.val_mse_)
            plt.axvline(np.argmin(self.val_mse_) + 1, linestyle="dotted", color="red")
            plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
            plt.plot(np.argmin(self.val_mse_) + 1, np.min(self.val_mse_), "*", markersize=12, color="red")
            plt.plot(len(self.best_estimators_), self.val_mse_[len(self.best_estimators_) - 1], "o", markersize=8, color="red")
            plt.xlabel("Number of Estimators", fontsize=12)
            plt.ylabel("Validation MSE", fontsize=12)
            plt.xlim(0.5, len(self.val_mse_) + 0.5)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.show()
        if is_classifier(self):
            fig = plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, len(self.val_auc_) + 1, 1), self.val_auc_)
            plt.axvline(np.argmax(self.val_auc_) + 1, linestyle="dotted", color="red")
            plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
            plt.plot(np.argmax(self.val_auc_) + 1, np.max(self.val_auc_), "*", markersize=12, color="red")
            plt.plot(len(self.best_estimators_), self.val_auc_[len(self.best_estimators_) - 1], "o", markersize=8, color="red")
            plt.xlabel("Number of Estimators", fontsize=12)
            plt.ylabel("Validation AUC", fontsize=12)
            plt.xlim(0.5, len(self.val_auc_) + 0.5)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.show()
        
    def visualize(self, cols_per_row=3):

        check_is_fitted(self, "best_estimators_")

        max_ids = len(self.best_estimators_) + len(self.cestimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)

        xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        for indice, pipe in enumerate(self.best_estimators_):

            estimator = pipe["sim_estimator"]
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
        
        for indice, estimator in enumerate(self.cestimators_):
                
            feature_name = self.cfeature_list_[indice]                
            beta = np.hstack([estimator[2].intercept_, estimator[2].coef_])
            inner = outer[len(self.best_estimators_) + indice].subgridspec(2, 1, wspace=0.15, height_ratios=[6, 1])
            ax1 = plt.Subplot(fig, inner[0])
            ax1.bar(np.arange(len(beta)), beta)
            ax1.set_xticklabels([])
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[1])
            cvalues = self.cdensity_[feature_name]["density"]["values"]
            cscores = self.cdensity_[feature_name]["density"]["scores"]
            ax2.bar(np.arange(len(cvalues)), cscores)
            ax1.get_shared_x_axes().join(ax1, ax2)

            input_ticks = (np.arange(len(cvalues)) if len(cvalues) <= 6 else 
                              np.linspace(0.1 * len(beta), len(beta) * 0.9, 4).astype(int))
            input_labels = [cvalues[i] for i in input_ticks]
            if len("".join(list(map(str, input_labels)))) > 30:
                input_labels = [str(cvalues[i])[:4] for i in input_ticks]

            ax2.set_xticks(input_ticks)
            ax2.set_xticklabels(input_labels)
            ax2.set_yticklabels([])
            fig.add_subplot(ax2)
            ax1.set_title(feature_name)
        plt.show()
    
    def _fit_dummy(self, x, y, sample_weight, feature_indice, feature_name):
        
        unique, counts = np.unique(x[:, feature_indice], return_counts=True)
        density = np.zeros((len(self.cvalues_[feature_name])))
        density[unique.astype(int)] = counts / x.shape[0]
        self.cdensity_.update({feature_name:{"density":{"values":self.cvalues_[feature_name],
                                        "scores":density}}})

        cestimator = Pipeline(steps = [('select', FunctionTransformer(lambda data, indice: data[:, [feature_indice]],
                                                                   kw_args={"indice": feature_indice},
                                                                   validate=False)),
                         ('ohe', OneHotEncoder(sparse=False, drop="first",
                                                 categories=[np.arange(len(self.cvalues_[feature_name]), dtype=np.float)])),
                         ('lr', LinearRegression())])        
        cestimator.fit(x, y, lr__sample_weight=sample_weight)
        return cestimator
    
    def fit(self, x, y, sample_weight=None, meta_info=None):

        start = time.time()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        self._validate_hyperparameters()
        self._preprocess_meta_info(n_features, meta_info)
        sample_weight = self._validate_sample_weight(n_samples, sample_weight)

        self.tr_idx, self.val_idx = train_test_split(np.arange(n_samples), test_size=self.val_ratio, random_state=self.random_state)
        self._fit(x, y, sample_weight)
        self.time_cost_ = time.time() - start
        return self

    def decision_function(self, x):

        check_is_fitted(self, "best_estimators_")

        pred_dummy = np.sum([est.predict(x) for est in self.cestimators_], axis=0)
        pred_simboost = np.sum([est.predict(x) for est in self.best_estimators_], axis=0)
        pred = pred_dummy + pred_simboost
        return pred


class SimBoostRegressor(BaseSimBooster, RegressorMixin):

    def __init__(self, n_estimators, val_ratio=0.2, degree=2, knot_num=20,
                 reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, loss_threshold=0.01, random_state=0):

        super(SimBoostRegressor, self).__init__(n_estimators=n_estimators,
                                   val_ratio=val_ratio,
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

    def _fit(self, x, y, sample_weight=None):
   
        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        pred_val = 0
        z = y.ravel()
        self.cdensity_ = {}
        self.cval_mse_ = []
        self.cestimators_ = []
        
        self.val_mse_ = []
        self.estimators_ = []
        
        for i in range(self.n_estimators):

            if i < len(self.cfeature_list_):
                feature_name = self.cfeature_list_[i]
                feature_indice = self.cfeature_index_list_[i]
                cestimator = self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx], feature_indice, feature_name)
                z = z - cestimator.predict(x)
                pred_val += cestimator.predict(x[self.val_idx])
                val_loss = mean_squared_error(y[self.val_idx], pred_val)
                self.cval_mse_.append(val_loss)
                self.cestimators_.append(cestimator)
            else:
                # projection matrix
                if (i == len(self.cfeature_list_)) or (i >= (len(self.cfeature_list_) + self.nfeature_num_)) or (self.ortho_shrink == 0):
                    proj_mat = np.eye(self.nfeature_num_)
                else:
                    projection_indices_ = np.array([estimator["sim_estimator"].beta_.flatten() for estimator in self.estimators_]).T
                    u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                    proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

                # fit Sim estimator
                param_grid = {"method": ["second_order", "first_order"], 
                              "reg_lambda": self.reg_lambda_list,
                              "reg_gamma": self.reg_lambda_list}
                grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                             scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                             cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
                grid.fit(x[:, self.nfeature_index_list_], y, sample_weight=sample_weight, proj_mat=proj_mat)
                estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
                sim_estimator = Pipeline(steps=[('select', FunctionTransformer(lambda data: data[:, self.nfeature_index_list_], validate=False)),
                                      ('sim_estimator', estimator)])
                sim_estimator.fit(x[self.tr_idx], z[self.tr_idx],
                            sim_estimator__sample_weight=sample_weight[self.tr_idx], sim_estimator__proj_mat=proj_mat)

                # update    
                z = z - estimator.predict(x[:, self.nfeature_index_list_])
                pred_val += estimator.predict(x[self.val_idx][:, self.nfeature_index_list_])
                val_loss = mean_squared_error(y[self.val_idx], pred_val)
                self.val_mse_.append(val_loss)
                self.estimators_.append(sim_estimator)

        best_loss = np.min(self.val_mse_)
        if np.sum((self.val_mse_ / best_loss - 1) < self.loss_threshold) > 0:
            best_idx = np.where((self.val_mse_ / best_loss - 1) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmin(self.val_mse_)
        self.best_estimators_ = self.estimators_[:(best_idx + 1)]

    def predict(self, x):

        check_is_fitted(self, "best_estimators_")
        pred = self.decision_function(x)
        return pred
    
    
class SimBoostClassifier(BaseSimBooster, ClassifierMixin):

    def __init__(self, n_estimators, val_ratio=0.2, degree=2, knot_num=20,
                 reg_lambda=0.1, reg_gamma=10, ortho_shrink=1, loss_threshold=0.01, random_state=0):

        super(SimBoostClassifier, self).__init__(n_estimators=n_estimators,
                                      val_ratio=val_ratio,
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

    def _fit(self, x, y, sample_weight=None):

        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        pred_train = 0
        pred_val = 0
        z = y.ravel()
        self.cdensity_ = {}
        self.cval_auc_ = []
        self.cestimators_ = []
        
        roc_auc_opt = -np.inf
        self.val_auc_ = []
        self.estimators_ = []
        proba_train = 0.5 * np.ones(len(self.tr_idx))
        proba_val = 0.5 * np.ones(len(self.val_idx))
        for i in range(self.n_estimators):

            sample_weight[self.tr_idx] = proba_train * (1 - proba_train)
            sample_weight[self.tr_idx] /= np.sum(sample_weight[self.tr_idx])
            sample_weight[self.tr_idx] = np.maximum(sample_weight[self.tr_idx], 2 * np.finfo(np.float64).eps)

            with np.errstate(divide="ignore", over="ignore"):
                z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                                -1. / (1. - np.hstack([proba_train, proba_val]))) 
                z = np.clip(z, a_min=-8, a_max=8)

            if i < len(self.cfeature_list_):
                feature_name = self.cfeature_list_[i]
                feature_indice = self.cfeature_index_list_[i]
                cestimator = self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx], feature_indice, feature_name)
                pred_train += cestimator.predict(x[self.tr_idx])
                pred_val += cestimator.predict(x[self.val_idx])
                proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
                proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
                
                val_auc = roc_auc_score(y[self.val_idx], proba_val)
                self.cval_auc_.append(val_auc)
                self.cestimators_.append(cestimator)
            else:
                # projection matrix
                if (i == len(self.cfeature_list_)) or (i >= (len(self.cfeature_list_) + self.nfeature_num_)) or (self.ortho_shrink == 0):
                    proj_mat = np.eye(self.nfeature_num_)
                else:
                    projection_indices_ = np.array([estimator["sim_estimator"].beta_.flatten() for estimator in self.estimators_]).T
                    u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                    proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

                # fit Sim estimator
                param_grid = {"method": ["second_order", "first_order"], 
                              "reg_lambda": self.reg_lambda_list,
                              "reg_gamma": self.reg_lambda_list}
                grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                              scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                              cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)

                grid.fit(x[:, self.nfeature_index_list_], z, sample_weight=sample_weight, proj_mat=proj_mat)
                estimator = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
                sim_estimator = Pipeline(steps = [('select', FunctionTransformer(lambda data: data[:, self.nfeature_index_list_],
                                              validate=False)),
                                       ('sim_estimator', estimator)])
                sim_estimator.fit(x[self.tr_idx], z[self.tr_idx],
                            sim_estimator__sample_weight=sample_weight[self.tr_idx], sim_estimator__proj_mat=proj_mat)
                # update
                pred_train += sim_estimator.predict(x[self.tr_idx])
                pred_val += sim_estimator.predict(x[self.val_idx])
                proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
                proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

                val_auc = roc_auc_score(y[self.val_idx], proba_val)
                self.val_auc_.append(val_auc)
                self.estimators_.append(sim_estimator)

        best_auc = np.max(self.val_auc_)
        if np.sum((1 - self.val_auc_ / best_auc) < self.loss_threshold) > 0:
            best_idx = np.where((1 - self.val_auc_ / best_auc) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmax(self.val_auc_)
        self.best_estimators_ = self.estimators_[:(best_idx + 1)]
    
    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba).reshape([-1, 1])