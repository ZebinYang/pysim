import time 
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from abc import ABCMeta, abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import softmax
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression, RidgeCV
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
        elif (isinstance(self.reg_lambda, float)) or (isinstance(self.reg_lambda, int)):
            if (self.reg_lambda < 0) or (self.reg_lambda > 1):
                raise ValueError("reg_lambda must be >= 0 and <=1, got %s." % self.reg_lambda)
            self.reg_lambda_list = [self.reg_lambda]

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma  
        elif (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if self.reg_gamma < 0:
                raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]

    @property
    def importance_ratios_(self):
        """Return the estimator importance ratios (the higher, the more important the feature).
        Returns
        -------
        importance_ratios_ : ndarray of shape (n_estimators,)
            The estimator importances.
        """
        importance_ratios_ = {}
        if (self.component_importance_ is not None) and (len(self.component_importance_) > 0):
            total_importance = np.sum([item["importance"] for key, item in self.component_importance_.items()])
            importance_ratios_ = {key: {"type": item["type"],
                               "indice": item["indice"],
                               "ir": item["importance"] / total_importance} for key, item in self.component_importance_.items()}
        return importance_ratios_


    @property
    def projection_indices_(self):
        """Return the projection indices.
        Returns
        -------
        projection_indices_ : ndarray of shape (d, n_estimators)
        """
        projection_indices = np.array([])
        if self.nfeature_num_ > 0:
            if (self.best_estimators_ is not None) and (len(self.best_estimators_) > 0):
                projection_indices = np.array([est["sim"].beta_.flatten() 
                                    for est in self.best_estimators_ if "sim" in est.named_steps.keys()]).T
        return projection_indices
        
    @property
    def orthogonality_measure_(self):
        """Return the orthogonality measure (the lower, the better).
        Returns
        -------
        orthogonality_measure_ : float scalar
        """
        ortho_measure = np.nan
        if self.nfeature_num_ > 0:
            if (self.best_estimators_ is not None) and (len(self.best_estimators_) > 0):
                ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                      self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
                if len(self.best_estimators_) > 1:
                    ortho_measure /= ((self.projection_indices_.shape[1] ** 2 - self.projection_indices_.shape[1]))
        return ortho_measure

    def _validate_sample_weight(self, n_samples, sample_weight):
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight.ravel() / np.sum(sample_weight)
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
            self.dummy_values_ = {}  
            self.cfeature_num_ = 0
            self.nfeature_num_ = 0
            self.feature_list_ = []
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
                    self.dummy_values_.update({feature_name:meta_info[feature_name]["values"]})
                else:
                    self.nfeature_num_ +=1
                    self.nfeature_list_.append(feature_name)
                    self.nfeature_index_list_.append(idx)
                self.feature_list_.append(feature_name)
            if n_features != (self.cfeature_num_ + self.nfeature_num_):
                raise ValueError("meta_info and n_features mismatch!")

    def validation_performance(self):

        check_is_fitted(self, "best_estimators_")

        if is_regressor(self):
            fig = plt.figure(figsize=(6, 4))
            plt.plot(np.arange(0, len(self.val_mse_), 1), self.val_mse_)
            plt.axvline(np.argmin(self.val_mse_), linestyle="dotted", color="red")
            plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
            plt.plot(np.argmin(self.val_mse_), np.min(self.val_mse_), "*", markersize=12, color="red")
            plt.plot(len(self.best_estimators_), self.val_mse_[len(self.best_estimators_)], "o", markersize=8, color="red")
            plt.xlabel("Number of Estimators", fontsize=12)
            plt.ylabel("Validation MSE", fontsize=12)
            plt.xlim(-0.5, len(self.val_mse_) - 0.5)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.show()
        if is_classifier(self):
            fig = plt.figure(figsize=(6, 4))
            plt.plot(np.arange(0, len(self.val_auc_), 1), self.val_auc_)
            plt.axvline(np.argmax(self.val_auc_), linestyle="dotted", color="red")
            plt.axvline(len(self.best_estimators_), linestyle="dotted", color="red")
            plt.plot(np.argmax(self.val_auc_), np.max(self.val_auc_), "*", markersize=12, color="red")
            plt.plot(len(self.best_estimators_), self.val_auc_[len(self.best_estimators_)], "o", markersize=8, color="red")
            plt.xlabel("Number of Estimators", fontsize=12)
            plt.ylabel("Validation AUC", fontsize=12)
            plt.xlim(-0.5, len(self.val_auc_) - 0.5)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.show()
        
    def visualize(self, cols_per_row=3, folder="./results/", name="demo", save_png=False, save_eps=False):

        check_is_fitted(self, "best_estimators_")

        subfig_idx = 0
        max_ids = len(self.best_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        for indice, est in enumerate(self.best_estimators_):
            
            estimator_key = list(self.importance_ratios_)[indice]
            if "sim" in est.named_steps.keys():
                sim = est["sim"]
                inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.15, height_ratios=[6, 1], width_ratios=[3, 1])
                ax1_main = fig.add_subplot(inner[0, 0])
                xgrid = np.linspace(sim.shape_fit_.xmin, sim.shape_fit_.xmax, 100).reshape([-1, 1])
                ygrid = sim.shape_fit_.decision_function(xgrid)
                ax1_main.plot(xgrid, ygrid, color="red")
                ax1_main.set_xticklabels([])
                ax1_main.set_title("SIM " + str(self.importance_ratios_[estimator_key]["indice"] + 1) +
                             " (IR: " + str(np.round(100 * self.importance_ratios_[estimator_key]["ir"], 2)) + "%)",
                             fontsize=16)
                fig.add_subplot(ax1_main)

                ax1_density = fig.add_subplot(inner[1, 0])  
                xint = ((np.array(sim.shape_fit_.bins_[1:]) + np.array(sim.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
                ax1_density.bar(xint, sim.shape_fit_.density_, width=xint[1] - xint[0])
                ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
                ax1_density.set_yticklabels([])
                fig.add_subplot(ax1_density)

                ax2 = fig.add_subplot(inner[:, 1])
                if len(sim.beta_) <= 10:
                    rects = ax2.barh(np.arange(len(sim.beta_)), [beta for beta in sim.beta_.ravel()][::-1])
                    ax2.set_yticks(np.arange(len(sim.beta_)))
                    ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(sim.beta_.ravel()))][::-1])
                    ax2.set_xlim(xlim_min, xlim_max)
                    ax2.set_ylim(-1, len(sim.beta_))
                    ax2.axvline(0, linestyle="dotted", color="black")
                else:
                    active_beta = []
                    active_beta_idx = []
                    for idx, beta in enumerate(sim.beta_.ravel()):
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
                fig.add_subplot(ax2)
                subfig_idx += 1
                
        for indice, est in enumerate(self.best_estimators_):
            
            if "dummy_lr" in est.named_steps.keys():

                feature_name = list(est.named_steps.keys())[0]
                dummy_values = self.dummy_density_[feature_name]["density"]["values"]
                dummy_scores = self.dummy_density_[feature_name]["density"]["scores"]
                dummy_coef = est["dummy_lr"].coef_

                inner = outer[subfig_idx].subgridspec(1, 2, wspace=0.0, width_ratios=[8, 1])
                ax1_density = fig.add_subplot(inner[0, 0])
                ax1_density.bar(np.arange(len(dummy_values)), dummy_scores)
                ax1_density.set_ylim(0, dummy_scores.max() * 1.2)
                
                input_ticks = (np.arange(len(dummy_values)) if len(dummy_values) <= 6 else 
                                  np.linspace(0.1 * len(dummy_coef), len(dummy_coef) * 0.9, 4).astype(int))
                input_labels = [dummy_values[i] for i in input_ticks]
                if len("".join(list(map(str, input_labels)))) > 30:
                    input_labels = [str(dummy_values[i])[:4] for i in input_ticks]

                ax1_main = ax1_density.twinx()
                ax1_main.set_xticks(input_ticks)
                ax1_main.set_xticklabels(input_labels)

                ax1_main.plot(np.arange(len(dummy_values)), dummy_coef, color="red")
                ax1_main.set_title(feature_name +
                             " (IR: " + str(np.round(100 * self.importance_ratios_[feature_name]["ir"], 2)) + "%)", fontsize=16)
                subfig_idx += 1
        plt.show()
        if max_ids > 0:
            if save_png:
                f.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
            if save_eps:
                f.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)

    def _fit_dummy(self, x, y, sample_weight):

        transformer_list = []
        for indice in range(self.cfeature_num_):
            
            feature_name = self.cfeature_list_[indice]
            feature_indice = self.cfeature_index_list_[indice]
            
            unique, counts = np.unique(x[:, feature_indice], return_counts=True)
            density = np.zeros((len(self.dummy_values_[feature_name])))
            density[unique.astype(int)] = counts / x.shape[0]
            self.dummy_density_.update({feature_name:{"density":{"values":self.dummy_values_[feature_name],
                                                                 "scores":density}}})

            transformer_list.append((feature_name,
                             OneHotEncoder(sparse=False, drop="first",
                             categories=[np.arange(len(self.dummy_values_[feature_name]), dtype=np.float)]), [feature_indice]))
        dummy_estimator_all = Pipeline(steps = [("ohe", ColumnTransformer(transformer_list)), ("lr", RidgeCV())])        
        dummy_estimator_all.fit(x, y, lr__sample_weight=sample_weight)

        idx = 0
        for indice in range(self.cfeature_num_):
            
            feature_name = self.cfeature_list_[indice]
            feature_indice = self.cfeature_index_list_[indice]
            dummy_num = self.dummy_values_[feature_name]
            dummy_coef = np.hstack([0.0, dummy_estimator_all["lr"].coef_[idx:(idx + len(dummy_num) - 1)].ravel()])

            dummy_estimator = Pipeline(steps=[(feature_name, FunctionTransformer(lambda data, idx: data[:, [idx]],
                                                validate=False, kw_args={"idx": feature_indice})),
                            ("ohe", OneHotEncoder(sparse=False,
                                          categories=[np.arange(len(self.dummy_values_[feature_name]), dtype=np.float)])),
                            ("dummy_lr", LinearRegression())])
            dummy_estimator.fit(x, y)
            dummy_estimator["dummy_lr"].intercept_ = 0
            dummy_estimator["dummy_lr"].coef_ = dummy_coef
            self.dummy_estimators_.append(dummy_estimator)
            idx += len(dummy_num) - 1

        self.intercept_ += dummy_estimator_all["lr"].intercept_

    def fit(self, x, y, sample_weight=None, meta_info=None):

        start = time.time()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        self._validate_hyperparameters()
        self._preprocess_meta_info(n_features, meta_info)
        sample_weight = self._validate_sample_weight(n_samples, sample_weight)

        self.intercept_ = 0
        self.sim_estimators_ = []
        self.dummy_estimators_ = []
        self.dummy_density_ = {}
        
        self.tr_idx, self.val_idx = train_test_split(np.arange(n_samples), test_size=self.val_ratio,
                                      random_state=self.random_state)
        self._fit(x, y, sample_weight)
        self._pruning(x, y)
        self.time_cost_ = time.time() - start
        return self

    def decision_function(self, x):

        check_is_fitted(self, "best_estimators_")
        pred = np.sum([est.predict(x) for est in self.best_estimators_], axis=0) + self.intercept_ * np.ones(x.shape[0])
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
        return x, y.ravel()

    def _fit(self, x, y, sample_weight=None):
   
        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        z = y.copy()
        # Fit categorical variables
        if self.cfeature_num_ > 0:
            self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx])
            z = z - np.sum([est.predict(x) for est in self.dummy_estimators_], axis=0) - self.intercept_
        else:
            self.intercept_ = np.mean(y)
            z = z - self.intercept_

        # Fit Sim Boosting for numerical variables
        if self.nfeature_num_ == 0:
            return 
        
        for i in range(self.n_estimators):

            # projection matrix
            if (i == 0) or (i >= self.nfeature_num_) or (self.ortho_shrink == 0):
                proj_mat = np.eye(self.nfeature_num_)
            else:
                projection_indices_ = np.array([est["sim"].beta_.flatten() for est in self.sim_estimators_]).T
                u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"], 
                          "reg_lambda": self.reg_lambda_list,
                          "reg_gamma": self.reg_gamma_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                         scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                         cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x[:, self.nfeature_index_list_], y, sample_weight=sample_weight, proj_mat=proj_mat)
            sim = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            sim_estimator = Pipeline(steps=[("select", FunctionTransformer(lambda data: data[:, self.nfeature_index_list_], validate=False)),
                                  ("sim", sim)])
            sim_estimator.fit(x[self.tr_idx], z[self.tr_idx],
                       sim__sample_weight=sample_weight[self.tr_idx], sim__proj_mat=proj_mat)
            # sim_estimator["sim"].fit_inner_update(x[:, self.nfeature_index_list_], z, sample_weight=sample_weight, proj_mat=proj_mat)
            # update    
            z = z - sim_estimator.predict(x)
            self.sim_estimators_.append(sim_estimator)

    def _pruning(self, x, y):
                
        component_importance = {}
        for indice, est in enumerate(self.sim_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                                     "importance": np.std(est.predict(x[self.tr_idx, :]))}})

        for indice, est in enumerate(self.dummy_estimators_):
            feature_name = list(est.named_steps.keys())[0]
            component_importance.update({feature_name: {"type": "dummy_lr", "indice": indice,
                                             "importance": np.std(est.predict(x[self.tr_idx, :]))}})
        
        self.estimators_ = []
        pred_val = self.intercept_ * np.ones(len(self.val_idx))
        self.val_mse_ = [mean_squared_error(y[self.val_idx], pred_val)]
        for key, item in sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1]:

            if item["type"] == "sim":
                est = self.sim_estimators_[item["indice"]]
            elif item["type"] == "dummy_lr":
                est = self.dummy_estimators_[item["indice"]]

            self.estimators_.append(est)
            pred_val += est.predict(x[self.val_idx])
            self.val_mse_.append(mean_squared_error(y[self.val_idx], pred_val))

        best_loss = np.min(self.val_mse_)
        if np.sum((self.val_mse_ / best_loss - 1) < self.loss_threshold) > 0:
            best_idx = np.where((self.val_mse_ / best_loss - 1) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmin(self.val_mse_)
        self.best_estimators_ = self.estimators_[:best_idx]
        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:best_idx])
    
    def predict(self, x):

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
        return x, y.ravel()

    def _fit(self, x, y, sample_weight=None):

        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        z = y.copy()
        if self.cfeature_num_ > 0:
            self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx])
            pred_train = np.sum([est.predict(x[self.tr_idx]) for est in self.dummy_estimators_], axis=0) + self.intercept_
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            pred_val = np.sum([est.predict(x[self.val_idx]) for est in self.dummy_estimators_], axis=0) + self.intercept_
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
        else:
            self.intercept_ = np.mean(y)
            pred_train = self.intercept_ * np.ones(len(self.tr_idx))
            pred_val = self.intercept_ * np.ones(len(self.val_idx))
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

        # Fit Sim Boosting for numerical variables
        if self.nfeature_num_ == 0:
            return 

        for i in range(self.n_estimators):
            sample_weight[self.tr_idx] = proba_train * (1 - proba_train)
            sample_weight[self.tr_idx] /= np.sum(sample_weight[self.tr_idx])
            sample_weight[self.tr_idx] = np.maximum(sample_weight[self.tr_idx], 2 * np.finfo(np.float64).eps)

            with np.errstate(divide="ignore", over="ignore"):
                z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                                -1. / (1. - np.hstack([proba_train, proba_val]))) 
                z = np.clip(z, a_min=-8, a_max=8)

            # projection matrix
            if (i == 0) or (i >= self.nfeature_num_) or (self.ortho_shrink == 0):
                proj_mat = np.eye(self.nfeature_num_)
            else:
                projection_indices_ = np.array([est["sim"].beta_.flatten() for est in self.sim_estimators_]).T
                u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)

            # fit Sim estimator
            param_grid = {"method": ["second_order", "first_order"], 
                          "reg_lambda": self.reg_lambda_list,
                          "reg_gamma": self.reg_gamma_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state), 
                          scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                          cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)

            grid.fit(x[:, self.nfeature_index_list_], z, sample_weight=sample_weight, proj_mat=proj_mat)
            sim = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            sim_estimator = Pipeline(steps = [("select", FunctionTransformer(lambda data: data[:, self.nfeature_index_list_], 
                                                        validate=False)),
                                   ("sim", sim)])
            sim_estimator.fit(x[self.tr_idx], z[self.tr_idx],
                        sim__sample_weight=sample_weight[self.tr_idx], sim__proj_mat=proj_mat)

            # update
            # sim_estimator["sim"].fit_inner_update(x[:, self.nfeature_index_list_], z, sample_weight=sample_weight, proj_mat=proj_mat)
            pred_train += sim_estimator.predict(x[self.tr_idx])
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            pred_val += sim_estimator.predict(x[self.val_idx])
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
            self.sim_estimators_.append(sim_estimator)
     
    def _pruning(self, x, y):
                
        component_importance = {}
        for indice, est in enumerate(self.sim_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                                  "importance": np.std(est.predict(x[self.tr_idx, :]))}})

        for indice, est in enumerate(self.dummy_estimators_):
            feature_name = list(est.named_steps.keys())[0]
            component_importance.update({feature_name: {"type": "dummy_lr", "indice": indice,
                                              "importance": np.std(est.predict(x[self.tr_idx, :]))}})
    
        self.estimators_ = []
        pred_val = self.intercept_ + np.zeros(len(self.val_idx))
        proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
        self.val_auc_ = [roc_auc_score(y[self.val_idx], pred_val)]
        for key, item in sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1]:

            if item["type"] == "sim":
                est = self.sim_estimators_[item["indice"]]
            elif item["type"] == "dummy":
                est = self.dummy_estimators_[item["indice"]]

            self.estimators_.append(est)
            pred_val += est.predict(x[self.val_idx])
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
            self.val_auc_.append(roc_auc_score(y[self.val_idx], proba_val))

        best_auc = np.max(self.val_auc_)
        if np.sum((1 - self.val_auc_ / best_auc) < self.loss_threshold) > 0:
            best_idx = np.where((1 - self.val_auc_ / best_auc) < self.loss_threshold)[0][0]
        else:
            best_idx = np.argmax(self.val_auc_)
        self.best_estimators_ = self.estimators_[:best_idx]
        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:best_idx])

    def predict_proba(self, x):

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)[:, 1]
        return pred_proba

    def predict(self, x):

        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba).reshape([-1, 1])