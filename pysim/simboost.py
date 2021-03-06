import os 
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

from .sim import SimRegressor, SimClassifier

__all__ = ["SimBoostRegressor", "SimBoostClassifier"]


class BaseSimBooster(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, n_estimators, meta_info=None, prjection_method="marginal_regression", spline="smoothing_spline_mgcv", knot_dist="quantile",
                 reg_lambda=0.1, reg_gamma="GCV", degree=3, knot_num=10, middle_update=None,
                 val_ratio=0.2, learning_rate=1.0, ortho_shrink=1,
                 early_stop_thres=np.inf, pruning=False, loss_threshold=0.01, elimination_threshold=0.05, random_state=0):

        self.n_estimators = n_estimators
        self.meta_info = meta_info

        self.spline = spline
        self.prjection_method = prjection_method
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_dist = knot_dist
        self.degree = degree
        self.knot_num = knot_num
        self.middle_update = middle_update

        self.val_ratio = val_ratio
        self.ortho_shrink = ortho_shrink
        self.learning_rate = learning_rate
        
        self.pruning = pruning
        self.loss_threshold = loss_threshold
        self.early_stop_thres = early_stop_thres
        self.elimination_threshold = elimination_threshold

        self.random_state = random_state        

    def _validate_hyperparameters(self):

        """method to validate model parameters
        """

        if not isinstance(self.n_estimators, int):
            raise ValueError("n_estimators must be an integer, got %s." % self.n_estimators)
        elif self.n_estimators < 0:
            raise ValueError("n_estimators must be >= 0, got %s." % self.n_estimators)

        if self.meta_info is not None:
            if not isinstance(self.meta_info, dict):
                raise ValueError("meta_info must be None or a dict, got %s." % self.meta_info)

        if self.spline not in ["a_spline", "smoothing_spline_mgcv", "smoothing_spline_bigsplines", "smoothing_spline_csaps", "p_spline", "mono_p_spline"]:
            raise ValueError("spline must be an element of [a_spline, smoothing_spline_mgcv, smoothing_spline_bigsplines, smoothing_spline_csaps, p_spline, mono_p_spline], got %s." % self.spline)
        
        if isinstance(self.prjection_method, list):
            for val in self.prjection_method:
                if val not in ["random", "first_order", "second_order", "first_order_thres", "marginal_regression", "marginal_regression", "ols"]:
                    raise ValueError("method must be an element of [random, first_order, second_order,\
                                first_order_thres, marginal_regression, ols], got %s." % 
                                 self.prjection_method)
            self.prjection_method_list = self.prjection_method  
        elif isinstance(self.prjection_method, str):
            if self.prjection_method not in ["random", "first_order", "second_order", "first_order_thres", "marginal_regression", "ols"]:
                raise ValueError("method must be an element of [random, first_order, second_order, first_order_thres,\
                                 marginal_regression, ols], got %s." % 
                                 self.prjection_method)
            self.prjection_method_list = [self.prjection_method]

        if isinstance(self.reg_lambda, list):
            for val in self.reg_lambda:
                if val < 0:
                    raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda_list = self.reg_lambda  
        elif (isinstance(self.reg_lambda, float)) or (isinstance(self.reg_lambda, int)):
            if (self.reg_lambda < 0) or (self.reg_lambda > 1):
                raise ValueError("reg_lambda must be >= 0 and <=1, got %s." % self.reg_lambda)
            self.reg_lambda_list = [self.reg_lambda]

        if isinstance(self.reg_gamma, str):
            self.reg_gamma_list = [self.reg_gamma]
        elif isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma  
        elif (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if self.reg_gamma < 0:
                raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)
        elif self.degree < 0:
            raise ValueError("degree must be >= 0, got %s." % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        elif self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got %s." % self.knot_num)

        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("knot_dist must be an element of [uniform, quantile], got %s." % self.knot_dist)

        if self.middle_update is None:
            self.middle_update = {"method":"bfgs",
                           "max_middle_iter":0,
                           "max_inner_iter":0}
        else:
            if not isinstance(self.middle_update, dict):
                raise ValueError("middle_update must be None or a dict containing middle_update %s." % self.middle_update)

        if self.val_ratio <= 0:
            raise ValueError("val_ratio must be > 0, got %s." % self.val_ratio)
        elif self.val_ratio >= 1:
            raise ValueError("val_ratio must be < 1, got %s." % self.val_ratio)

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0, got %s." % self.learning_rate)
        elif self.learning_rate > 1:
            raise ValueError("learning_rate must be <= 1, got %s." % self.learning_rate)
  
        if self.ortho_shrink <= 0:
            raise ValueError("ortho_shrink must be > 0, got %s." % self.ortho_shrink)
        elif self.ortho_shrink > 1:
            raise ValueError("ortho_shrink must be <= 1, got %s." % self.ortho_shrink)

        if not isinstance(self.pruning, bool):
            raise ValueError("pruning must be a bool, got %s." % self.pruning)

        if not isinstance(self.loss_threshold, float):
            raise ValueError("loss_threshold must be a float, got %s." % self.loss_threshold)

        if not isinstance(self.elimination_threshold, float):
                raise ValueError("elimination_threshold must be a float, got %s." % self.elimination_threshold)
        if self.elimination_threshold < 0:
            raise ValueError("elimination_threshold must be >= 0, got %s." % self.elimination_threshold)
        elif self.elimination_threshold > 1:
            raise ValueError("elimination_threshold must be <= 1, got %s." % self.elimination_threshold)
        
        if self.early_stop_thres < 1:
            raise ValueError("early_stop_thres must be greater than 1, got %s." % self.early_stop_thres)
            
    @property
    def importance_ratios_(self):
        """return the estimator importance ratios (the higher, the more important the feature)

        Returns
        -------
        dict of selected estimators
            the importance ratio of each fitted base learner.
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
        """return the projection indices

        Returns
        -------
        ndarray of shape (n_features, n_estimators)
            the projection indices
        """
        projection_indices = np.array([])
        if self.nfeature_num_ > 0:
            if (self.best_estimators_ is not None) and (len(self.best_estimators_) > 0):
                projection_indices = np.array([est["sim"].beta_.flatten() 
                                    for est in self.best_estimators_ if "sim" in est.named_steps.keys()]).T
        return projection_indices
        
    @property
    def orthogonality_measure_(self):
        """return the orthogonality measure (the lower, the better)
        
        Returns
        -------
        float
            the orthogonality measure
        """
        ortho_measure = np.nan
        if self.nfeature_num_ > 0:
            if (self.best_estimators_ is not None) and (len(self.best_estimators_) > 0):
                ortho_measure = np.linalg.norm(np.dot(self.projection_indices_.T,
                                      self.projection_indices_) - np.eye(self.projection_indices_.shape[1]))
                if len(self.best_estimators_) > 1:
                    ortho_measure /= self.projection_indices_.shape[1]
        return ortho_measure

    def _validate_sample_weight(self, n_samples, sample_weight):
                
        """method to validate sample weight
        
        Parameters
        ---------
        n_samples : int
            the number of samples
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        """

        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight.ravel() / np.sum(sample_weight)
        return sample_weight
    
    def _preprocess_meta_info(self, n_features):
        
        """preprocess the meta info of the dataset
        
        Parameters
        ---------
        n_features : int
            the number of features
        """
        
        if self.meta_info is None:
            self.meta_info = {}
            for i in range(n_features):
                self.meta_info.update({"X" + str(i + 1):{"type":"continuous"}})
            self.meta_info.update({"Y":{"type":"target"}})
        
        if not isinstance(self.meta_info, dict):
            raise ValueError("meta_info must be None or a dict, got" % self.meta_info)
        else:
            self.dummy_values_ = {}  
            self.cfeature_num_ = 0
            self.nfeature_num_ = 0
            self.feature_list_ = []
            self.cfeature_list_ = []
            self.nfeature_list_ = []
            self.cfeature_index_list_ = []
            self.nfeature_index_list_ = []
            for idx, (feature_name, feature_info) in enumerate(self.meta_info.items()):
                if feature_info["type"] == "target":
                    continue
                if feature_info["type"] == "categorical":
                    self.cfeature_num_ += 1
                    self.cfeature_list_.append(feature_name)
                    self.cfeature_index_list_.append(idx)
                    self.dummy_values_.update({feature_name:self.meta_info[feature_name]["values"]})
                else:
                    self.nfeature_num_ +=1
                    self.nfeature_list_.append(feature_name)
                    self.nfeature_index_list_.append(idx)
                self.feature_list_.append(feature_name)
            if n_features != (self.cfeature_num_ + self.nfeature_num_):
                raise ValueError("meta_info and n_features mismatch!")

    def validation_performance(self):

        """draw the validation accuracy (regression and AUC for binary classification) against the number of base learners
        """
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
        
    def visualize(self, cols_per_row=3, folder="./results/", name="global_plot", save_png=False, save_eps=False):

        """draw the global interpretation of the fitted model
        
        Parameters
        ---------
        cols_per_row : int, optional, default=3,
            the number of sim models visualized on each row
        folder : str, optional, defalut="./results/"
            the folder of the file to be saved
        name : str, optional, default="global_plot"
            the name of the file to be saved
        save_png : bool, optional, default=False
            whether to save the figure in png form
        save_eps : bool, optional, default=False
            whether to save the figure in eps form
        """

        check_is_fitted(self, "best_estimators_")

        subfig_idx = 0
        max_ids = len(self.best_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        
        if self.projection_indices_.shape[1] > 0:
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
                if len(sim.beta_) <= 20:
                    rects = ax2.barh(np.arange(len(sim.beta_)), [beta for beta in sim.beta_.ravel()][::-1])
                    ax2.set_yticks(np.arange(len(sim.beta_)))
                    ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(sim.beta_.ravel()))][::-1])
                    ax2.set_xlim(xlim_min, xlim_max)
                    ax2.set_ylim(-1, len(sim.beta_))
                    ax2.axvline(0, linestyle="dotted", color="black")
                else:
                    right = np.round(np.linspace(0, np.round(len(sim.beta_) * 0.45).astype(int), 5))
                    left = len(sim.beta_) - 1 - right
                    input_ticks = np.unique(np.hstack([left, right])).astype(int)

                    rects = ax2.barh(np.arange(len(sim.beta_)), [beta for beta in sim.beta_.ravel()][::-1])
                    ax2.set_yticks(input_ticks)
                    ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                    ax2.set_xlim(xlim_min, xlim_max)
                    ax2.set_ylim(-1, len(sim.beta_))
                    ax2.axvline(0, linestyle="dotted", color="black")
                fig.add_subplot(ax2)
                subfig_idx += 1
                
        for indice, est in enumerate(self.best_estimators_):

            if "dummy_lr" in est.named_steps.keys():

                feature_name = list(est.named_steps.keys())[0]
                dummy_values = self.dummy_density_[feature_name]["density"]["values"]
                dummy_scores = self.dummy_density_[feature_name]["density"]["scores"]
                dummy_coef = est["dummy_lr"].coef_

                ax_main = fig.add_subplot(outer[subfig_idx])
                ax_density = ax_main.twinx()
                ax_density.bar(np.arange(len(dummy_values)), dummy_scores, width=0.6)
                ax_density.set_ylim(0, dummy_scores.max() * 1.2)
                ax_density.set_yticklabels([])

                input_ticks = (np.arange(len(dummy_values)) if len(dummy_values) <= 6 else 
                                  np.linspace(0.1 * len(dummy_values), len(dummy_values) * 0.9, 4).astype(int))
                input_labels = [dummy_values[i] for i in input_ticks]
                if len("".join(list(map(str, input_labels)))) > 30:
                    input_labels = [str(dummy_values[i])[:4] for i in input_ticks]

                ax_main.set_xticks(input_ticks)
                ax_main.set_xticklabels(input_labels)
                ax_main.set_ylim(- np.abs(dummy_coef).max() * 1.2, np.abs(dummy_coef).max() * 1.2)
                ax_main.plot(np.arange(len(dummy_values)), dummy_coef, color="red", marker="o")
                ax_main.axhline(0, linestyle="dotted", color="black")
                ax_main.set_title(feature_name +
                                 " (IR: " + str(np.round(100 * self.importance_ratios_[feature_name]["ir"], 2)) + "%)", fontsize=16)
                ax_main.set_zorder(ax_density.get_zorder() + 1)
                ax_main.patch.set_visible(False)

                subfig_idx += 1
        plt.show()
        if max_ids > 0:
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)


    def local_visualize(self, x, y=None, folder="./results/", name="local_plot", save_png=False, save_eps=False):

        """draw the local interpretation of the fitted model
        
        Parameters
        ---------
        cols_per_row : int, optional, default=3,
            the number of sim models visualized on each row
        folder : str, optional, defalut="./results/"
            the folder of the file to be saved
        name : str, optional, default="global_plot"
            the name of the file to be saved
        save_png : bool, optional, default=False
            whether to save the figure in png form
        save_eps : bool, optional, default=False
            whether to save the figure in eps form
        """

        ytick_label = ["Intercept"]
        max_ids = len(self.best_estimators_) + 1
        for indice, est in enumerate(self.best_estimators_):
            if "sim" in est.named_steps.keys():
                estimator_key = list(self.importance_ratios_)[indice]
                ytick_label.append("SIM " + str(self.importance_ratios_[estimator_key]["indice"] + 1))
            if "dummy_lr" in est.named_steps.keys():
                feature_name = list(est.named_steps.keys())[0]
                ytick_label.append(feature_name)

        if is_regressor(self):
            predicted = self.predict(x)
        elif is_classifier(self):
            predicted = self.predict_proba(x)
        
        fig = plt.figure(figsize=(6, round((max_ids + 1) * 0.45)))
        plt.barh(np.arange(max_ids), np.hstack([self.intercept_] + [est.predict(x) for est in self.best_estimators_])[::-1])
        plt.yticks(np.arange(max_ids), ytick_label[::-1])

        if y is not None:
            title = "Predicted: %0.4f | Actual: %0.4f" % (predicted, y)  
        else:
            title = "Predicted: %0.4f"% (predicted)
        plt.title(title, fontsize=12)

        save_path = folder + name
        if (max_ids > 0) & save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if (max_ids > 0) & save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


    def feature_gradient(self, x):
            
        """calculate gradients of the fitted model to the input data
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        array-like of shape (n_samples, n_features)
            containing the gradient of the input dataset
        """

        n_samples = x.shape[0]
        gradient = np.zeros((n_samples, self.nfeature_num_ + self.cfeature_num_))
        for est in self.best_estimators_:
            if "sim" in est.named_steps.keys():
                sim = est["sim"]
                beta = sim.beta_
                shape_gradident = sim.shape_fit_.diff(np.dot(x[:, self.nfeature_index_list_], sim.beta_), 1)
                gradient[:, self.nfeature_index_list_] += (beta * shape_gradident).T
            elif "dummy_lr" in est.named_steps.keys():
                # the gradient for categorical features does not exist,
                # we instead calculate the average difference between the current category and all the other categories.
                gradient[:, est[0].kw_args["idx"]] += (np.sum(est["dummy_lr"].coef_) - est.predict(x)) \
                                            / (len(est["dummy_lr"].coef_) - 1) - est.predict(x)
        return gradient

    def ice_visualize(self, x, cols_per_row=3, folder="./results/", name="ice_visualize", save_png=False, save_eps=False):

        """draw the Individual Conditional Expectation (ICE) plot for the fitted model
        """

        max_ids = self.nfeature_num_ + self.cfeature_num_
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        for feature_indice in range(self.cfeature_num_ + self.nfeature_num_):

            feature_name = self.feature_list_[feature_indice]
            ax = fig.add_subplot(outer[feature_indice])
            ax.scatter(x[:, feature_indice], self.predict(x), color="red", s=50, zorder=10)
            if feature_indice in self.nfeature_index_list_:
                
                xx = np.tile(x, (100, 1))
                xgrid = np.linspace(0, 1, 100)
                xx[:, [feature_indice]] = xgrid.reshape(-1, 1)
                ygrid = self.predict(xx)
                ax.plot(xgrid, ygrid)
                
            elif feature_indice in self.cfeature_index_list_:

                dummy_values = self.dummy_values_[feature_name]
                xgrid = np.arange(len(dummy_values))
                xx = np.tile(x, (len(dummy_values), 1))
                xx[:, [feature_indice]] = xgrid.reshape(-1, 1)
                ygrid = self.predict(xx)
                ax.bar(xgrid, ygrid)
                input_ticks = (np.arange(len(dummy_values)) if len(dummy_values) <= 6 else 
                                  np.linspace(0.1 * len(dummy_values), len(dummy_values) * 0.9, 4).astype(int))
                input_labels = [dummy_values[i] for i in input_ticks]
                if len("".join(list(map(str, input_labels)))) > 30:
                    input_labels = [str(dummy_values[i])[:4] for i in input_ticks]

                ax.set_xticks(input_ticks)
                ax.set_xticklabels(input_labels)
                ax.set_ylim(0, np.abs(ygrid).max() * 1.2)
            ax.set_title(feature_name, fontsize=16)
            fig.add_subplot(ax)
        plt.show()
        save_path = folder + name
        if save_eps:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
        if save_png:
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


    def _fit_dummy(self, x, y, sample_weight):
        
        """fit the categorical variables by one-hot encoding and linear models

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        """
        
        transformer_list = []
        for indice in range(self.cfeature_num_):
            
            feature_name = self.cfeature_list_[indice]
            feature_indice = self.cfeature_index_list_[indice]
            
            unique, counts = np.unique(x[:, feature_indice], return_counts=True)
            density = np.zeros((len(self.dummy_values_[feature_name])))
            density[unique.round().astype(int)] = counts / x.shape[0]
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
            dummy_coef = np.hstack([0.0, dummy_estimator_all["lr"].coef_.ravel()[idx:(idx + len(dummy_num) - 1)]])

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

    def fit(self, x, y, sample_weight=None):

        """fit the SimBoost model

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """
        
        start = time.time()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        self._validate_hyperparameters()
        self._preprocess_meta_info(n_features)
        sample_weight = self._validate_sample_weight(n_samples, sample_weight)

        self.intercept_ = 0
        self.sim_estimators_ = []
        self.dummy_estimators_ = []
        self.dummy_density_ = {}
        self.learning_rates = [1] + [self.learning_rate] * (self.n_estimators - 1)
        
        if is_regressor(self):
            self.tr_idx, self.val_idx = train_test_split(np.arange(n_samples), test_size=self.val_ratio,
                                          random_state=self.random_state)
        elif is_classifier(self):
            self.tr_idx, self.val_idx = train_test_split(np.arange(n_samples), test_size=self.val_ratio,
                                          stratify=y, random_state=self.random_state)

        self._fit(x, y, sample_weight)
        self._pruning(x, y)
        self.time_cost_ = time.time() - start
        return self

    def decision_function(self, x):

        """output f1(beta1^T x) + f2(beta2^T x) + ... for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,),
            containing f1(beta1^T x) + f2(beta2^T x) + ...
        """

        check_is_fitted(self, "best_estimators_")
        pred = self.intercept_ * np.ones(x.shape[0])
        for indice, est in enumerate(self.best_estimators_):
            if "sim" in est.named_steps.keys():
                pred += self.best_weights_[indice] * est.predict(x)
            elif "dummy_lr" in est.named_steps.keys():
                pred += self.best_weights_[indice] * est.predict(x)
        return pred


class SimBoostRegressor(BaseSimBooster, RegressorMixin):

    """
    Base class for sim boost regression (residual boosting).

    Training Steps:
    
    1. Preprocess all categorical features with one-hot encodeing, and then build a linear model between all the dummy variables and the response. (_fit_dummy) 
    
    2. Calculate the pseudo residual using logit boost. (_fit)
    
    3. Calculate the orthogonal enhancement for the next SIM Regressor (ortho_shrink, only used when learning_rate=1.0). (_fit)
    
    4. Fit a SIM Regressor using all numerical features and the pseudo residual. (_fit)
    
    5. Recalculate the pseudo residual subject to learning_rate. (_fit)
    
    6. Repeat steps 2 - 5 until n_estimators is reached. (_fit)
    
    7. Rank the fitted SIM Regressors according to variation they explained. (_pruning)
    
    8. Sequentially add the ranked SIM Regressors (starting from top ranked) and evaluate the validation performance. (_pruning)
    
    9. Select the best number of SIM Regressors according to the validation performance. (_pruning)
    
    10. Interpretation: the pruning procedure (validation_performance), global model (visualize), local interpretation (local_visualize and ice_visualize).

    Parameters
    ----------
    n_estimators : int
        The maximum number of estimators for boosing

    meta_info : None or a dict with features' information. default=None
        Features are classified as:

        continuous:
            Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
            `Wrapper`, a callable function for wrapping the values
        categorical:
            Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories)

        If None, then all the features will be treated as continuous
        
    spline : str, optional. default="smoothing_spline_mgcv"
        The type of spline for fitting the curve
      
        "smoothing_spline_bigsplines": Smoothing spline based on bigsplines package in R

        "smoothing_spline_mgcv": Smoothing spline based on mgcv package in R

        "p_spline": P-spline

        "mono_p_spline": P-spline with monotonic constraint

        "a_spline": Adaptive B-spline

    prjection_method : str, optional. default="marginal_regression"
        The base method for estimating the projection coefficients in sparse SIM
        
        "random": Randomized initialization from the unit sphere

        "first_order": First-order Stein's Identity via sparse PCA solver

        "second_order": Second-order Stein's Identity via sparse PCA solver

        "first_order_thres": First-order Stein's Identity via hard thresholding (A simplified verison)     

        "marginal_regression": Marginal regression subject to hard thresholding
        
        "ols": Least squares estimation subject to hard thresholding.

    knot_dist : str, optional. default="quantile"
        Distribution of knots
      
        "uniform": uniformly over the domain

        "quantile": uniform quantiles of the given input data (not available when spline="p_spline" or "mono_p_spline")

    reg_lambda : float, optional. default=0.1
        The sparsity strength of projection inidce, ranges from 0 to 1 

    reg_gamma : float, optional. default=0.1
        Roughness penalty strength of the spline algorithm
    
        For spline="smoothing_spline_bigsplines", it ranges from 0 to 1, and the suggested tuning grid is 1e-9 to 1e-1; and it can be set to "GCV".

        For spline="smoothing_spline_mgcv", it ranges from 0 to :math:`+\infty`, and it can be set to "GCV".

        For spline="p_spline","mono_p_spline" or "a_spline", it ranges from 0 to :math:`+\infty`
    
    degree : int, optional. default=3
        The order of the spline.
        
        For spline="smoothing_spline_bigsplines", possible values include 1 and 3.
    
        For spline="smoothing_spline_mgcv", possible values include 3, 4, ....
    
    knot_num : int, optional. default=10
        Number of knots
    
    middle_update : None or str, optional. default=None
        The inner update method for each base learner, can be None, "adam" or "bfgs"
   
    val_ratio : float, optional. default=0.2
        The split ratio of validation set, which is used for post-hoc pruning

    ortho_shrink : float, optional. default=1
        Shrinkage strength for orthogonal enhancement, ranges from 0 to 1, valid when learning_rage=1.0
    
    learning_rate : float, optional. default=1.0
        The learning rate controling the shrinkage when performing boosting, ranges from 0 to 1

    early_stop_thres : float. default=np.inf
        The boosting algorithm will be stopped if the validation performance does not get improved for early_stop_thres estimators.
        
    pruning : bool. default=False
        Whether to perform pruning for the base sim estimators
    
    loss_threshold : float, optional. default=0.01
        This parameter is used for post-hoc pruning, ranges from 0 to 1, only used when pruning=True
        To reduce model complexity, we prefer to use fewer base learners, which is as accurate as (1 - loss_threshold) of the best performance)

    random_state : int, optional. default=0
        Random seed
    """


    def __init__(self, n_estimators, meta_info=None, prjection_method="marginal_regression", spline="smoothing_spline_mgcv", knot_dist="quantile",
                 reg_lambda=0.1, reg_gamma="GCV", degree=3, knot_num=10, middle_update=None,
                 val_ratio=0.2, learning_rate=1.0, ortho_shrink=1,
                 early_stop_thres=np.inf, pruning=False, loss_threshold=0.01, elimination_threshold=0.05,
                 random_state=0):

        super(SimBoostRegressor, self).__init__(n_estimators=n_estimators,
                                   meta_info=meta_info,
                                   spline=spline,
                                   prjection_method=prjection_method,
                                   reg_lambda=reg_lambda,
                                   reg_gamma=reg_gamma,
                                   knot_dist=knot_dist,
                                   degree=degree,
                                   knot_num=knot_num,
                                   middle_update=middle_update,
                                   val_ratio=val_ratio,
                                   learning_rate=learning_rate,
                                   ortho_shrink=ortho_shrink,
                                   early_stop_thres=early_stop_thres,
                                   pruning=pruning,
                                   loss_threshold=loss_threshold,
                                   elimination_threshold=elimination_threshold,
                                   random_state=random_state)

    def _validate_input(self, x, y):
        
        """method to validate data
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def _fit(self, x, y, sample_weight=None):
   
        """fit the SimBoost model

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        """

        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        # Initialize the intercept
        z = y.copy()
        self.intercept_ = np.mean(z)
        z = z - self.intercept_

        # Fit categorical variables
        if self.cfeature_num_ > 0:
            self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx])
            z = y - np.sum([est.predict(x) for est in self.dummy_estimators_], axis=0) - self.intercept_
        
        # Fit Sim Boosting for numerical variables
        if self.nfeature_num_ == 0:
            return 
        
        mse_opt = np.inf
        early_stop_count = 0
        for indice in range(self.n_estimators):

            # projection matrix
            if self.learning_rate == 1:
                if (indice == 0) or (indice >= self.nfeature_num_) or (self.ortho_shrink == 0):
                    proj_mat = np.eye(self.nfeature_num_)
                else:
                    projection_indices_ = np.array([est["sim"].beta_.flatten() for est in self.sim_estimators_]).T
                    u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                    proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)
            else:
                proj_mat = None

            # fit Sim estimator
            param_grid = {"method": self.prjection_method_list, 
                      "reg_lambda": self.reg_lambda_list,
                      "reg_gamma": self.reg_gamma_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, spline=self.spline,
                                  knot_dist=self.knot_dist, random_state=self.random_state), 
                         scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit=False,
                         cv=PredefinedSplit(val_fold), param_grid=param_grid, verbose=0, error_score=np.nan)
            grid.fit(x[:, self.nfeature_index_list_], z, sample_weight=sample_weight, proj_mat=proj_mat)
            sim = grid.estimator.set_params(**grid.cv_results_["params"][np.where((grid.cv_results_["rank_test_mse"] == 1))[0][0]])
            sim_estimator = Pipeline(steps=[("select", FunctionTransformer(lambda data: data[:, self.nfeature_index_list_], validate=False)),
                                  ("sim", sim)])
            
            sim_estimator.fit(x[self.tr_idx], z[self.tr_idx],
                       sim__sample_weight=sample_weight[self.tr_idx], sim__proj_mat=proj_mat)
            sim_estimator["sim"].fit_middle_update(x[:, self.nfeature_index_list_], z, 
                    sample_weight=sample_weight, proj_mat=proj_mat, val_ratio=self.val_ratio, **self.middle_update)
            # update
            z = z - self.learning_rates[indice] * sim_estimator.predict(x)
            mse_new = np.mean(z[self.val_idx] ** 2)
            if mse_opt > mse_new:           
                mse_opt = mse_new
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop_thres:
                break

            self.sim_estimators_.append(sim_estimator)

    def _pruning(self, x, y):
          
        """prune the base learners that are not importnat
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples, 1)
            containing the output dataset
        """

        component_importance = {}
        for indice, est in enumerate(self.sim_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                 "importance": np.std(self.learning_rates[indice] * est.predict(x[self.tr_idx, :]))}})

        for indice, est in enumerate(self.dummy_estimators_):
            feature_name = list(est.named_steps.keys())[0]
            component_importance.update({feature_name: {"type": "dummy_lr", "indice": indice,
                                 "importance": np.std(est.predict(x[self.tr_idx, :]))}})
        
        self.weights_ = []
        self.estimators_ = []
        pred_val = self.intercept_ * np.ones(len(self.val_idx))
        self.val_mse_ = [mean_squared_error(y[self.val_idx], pred_val)]
        for key, item in sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1]:

            if item["type"] == "sim":
                est = self.sim_estimators_[item["indice"]]
                pred_val += self.learning_rates[item["indice"]] * est.predict(x[self.val_idx])
                self.weights_.append(self.learning_rates[item["indice"]])
            elif item["type"] == "dummy_lr":
                est = self.dummy_estimators_[item["indice"]]
                pred_val += est.predict(x[self.val_idx])
                self.weights_.append(1)
            self.estimators_.append(est)
            self.val_mse_.append(mean_squared_error(y[self.val_idx], pred_val))

        if not self.pruning:
            best_idx = len(self.val_mse_) - 1
        else:
            best_loss = np.min(self.val_mse_)
            if np.sum((self.val_mse_ / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.val_mse_ / best_loss - 1) < self.loss_threshold)[0][0]
            else:
                best_idx = np.argmin(self.val_mse_)
            
        self.best_weights_ = self.weights_[:best_idx]
        self.best_estimators_ = self.estimators_[:best_idx]
        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:best_idx])
        self.activate_cfeature_index_ = [est[0].kw_args["idx"] for est in self.best_estimators_ if "dummy_lr" in est.named_steps.keys()]

        if self.elimination_threshold:
            eli_idx = np.sum([True if item['ir']>self.elimination_threshold else False for key,item in self.importance_ratios_.items()])
            self.best_weights_ = self.weights_[:eli_idx]
            self.best_estimators_ = self.estimators_[:eli_idx]
            self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:eli_idx])
            self.activate_cfeature_index_ = [est[0].kw_args["idx"] for est in self.best_estimators_ if "dummy_lr" in est.named_steps.keys()]

    def predict(self, x):

        """output prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing prediction
        """  

        pred = self.decision_function(x)
        return pred
    
    
class SimBoostClassifier(BaseSimBooster, ClassifierMixin):

    """
    Base class for sim boost classification (logit boost).

    Training Steps:
    1. Preprocess all categorical features with one-hot encodeing, and then build a linear model between all the dummy variables and the response. (_fit_dummy)
    
    2. Calculate the pseudo residual using logit boost. (_fit)
    
    3. Calculate the orthogonal enhancement for the next SIM Regressor (ortho_shrink, only used when learning_rate=1.0). (_fit)
    
    4. Fit a SIM Regressor using all numerical features and the pseudo residual. (_fit)
    
    5. Recalculate the pseudo residual subject to learning_rate. (_fit)
    
    6. Repeat steps 2 - 5 until n_estimators is reached. (_fit)
    
    7. Rank the fitted SIM Regressors according to variation they explained. (_pruning)
    
    8. Sequentially add the ranked SIM Regressors (starting from top ranked) and evaluate the validation performance. (_pruning)
    
    9. Select the best number of SIM Regressors according to the validation performance. (_pruning)
    
    10. Interpretation: the pruning procedure (validation_performance), global model (visualize), local interpretation (local_visualize and ice_visualize).
    
    Parameters
    ----------
    n_estimators : int
        The maximum number of estimators for boosing

    meta_info : None or a dict with features' information. default=None
        Features are classified as:

        continuous:
            Specify `Type` as `continuous`, and include the keys of `Range` (a list with lower-upper elements pair) and
            `Wrapper`, a callable function for wrapping the values
        categorical:
            Specify `Type` as `categorical`, and include the keys of `Mapping` (a list with all the possible categories)

        If None, then all the features will be treated as continuous
        
    spline : str, optional. default="smoothing_spline_mgcv"
        The type of spline for fitting the curve
      
        "smoothing_spline_bigsplines": Smoothing spline based on bigsplines package in R

        "smoothing_spline_mgcv": Smoothing spline based on mgcv package in R

        "p_spline": P-spline

        "mono_p_spline": P-spline with monotonic constraint

        "a_spline": Adaptive B-spline

    prjection_method : str, optional. default="marginal_regression"
        The base method for estimating the projection coefficients in sparse SIM
        
        "random": Randomized initialization from the unit sphere

        "first_order": First-order Stein's Identity via sparse PCA solver

        "second_order": Second-order Stein's Identity via sparse PCA solver

        "first_order_thres": First-order Stein's Identity via hard thresholding (A simplified verison)     

        "marginal_regression": Marginal regression subject to hard thresholding
        
        "ols": Least squares estimation subject to hard thresholding.

    knot_dist : str, optional. default="quantile"
        Distribution of knots
      
        "uniform": uniformly over the domain

        "quantile": uniform quantiles of the given input data (not available when spline="p_spline" or "mono_p_spline")

    reg_lambda : float, optional. default=0.1
        The sparsity strength of projection inidce, ranges from 0 to 1 

    reg_gamma : float, optional. default=0.1
        Roughness penalty strength of the spline algorithm
    
        For spline="smoothing_spline_bigsplines", it ranges from 0 to 1, and the suggested tuning grid is 1e-9 to 1e-1; and it can be set to "GCV".

        For spline="smoothing_spline_mgcv", it ranges from 0 to :math:`+\infty`, and it can be set to "GCV".

        For spline="p_spline","mono_p_spline" or "a_spline", it ranges from 0 to :math:`+\infty`
    
    degree : int, optional. default=3
        The order of the spline.
        
        For spline="smoothing_spline_bigsplines", possible values include 1 and 3.
    
        For spline="smoothing_spline_mgcv", possible values include 3, 4, ....
    
    knot_num : int, optional. default=10
        Number of knots
    
    middle_update : None or str, optional. default=None
        The inner update method for each base learner, can be None, "adam" or "bfgs"
   
    val_ratio : float, optional. default=0.2
        The split ratio of validation set, which is used for post-hoc pruning

    ortho_shrink : float, optional. default=1
        Shrinkage strength for orthogonal enhancement, ranges from 0 to 1, valid when learning_rage=1.0
    
    learning_rate : float, optional. default=1.0
        The learning rate controling the shrinkage when performing boosting, ranges from 0 to 1

    early_stop_thres : float. default=np.inf
        The boosting algorithm will be stopped if the validation performance does not get improved for early_stop_thres estimators.
        
    pruning : bool. default=False
        Whether to perform pruning for the base sim estimators
    
    loss_threshold : float, optional. default=0.01
        This parameter is used for post-hoc pruning, ranges from 0 to 1, only used when pruning=True
        To reduce model complexity, we prefer to use fewer base learners, which is as accurate as (1 - loss_threshold) of the best performance)

    random_state : int, optional. default=0
        Random seed
    """

    def __init__(self, n_estimators, meta_info=None, prjection_method="marginal_regression", spline="smoothing_spline_mgcv", knot_dist="quantile",
                 reg_lambda=0.1, reg_gamma="GCV", degree=3, knot_num=10, middle_update=None,
                 val_ratio=0.2, learning_rate=1.0, ortho_shrink=1,
                 early_stop_thres=np.inf, pruning=False, loss_threshold=0.01, 
                 elimination_threshold=0.05,
                 random_state=0):

        super(SimBoostClassifier, self).__init__(n_estimators=n_estimators,
                                    meta_info=meta_info,
                                    spline=spline,
                                    prjection_method=prjection_method,
                                    reg_lambda=reg_lambda,
                                    reg_gamma=reg_gamma,
                                    knot_dist=knot_dist,
                                    degree=degree,
                                    knot_num=knot_num,
                                    middle_update=middle_update,
                                    val_ratio=val_ratio,
                                    learning_rate=learning_rate,
                                    ortho_shrink=ortho_shrink,
                                    early_stop_thres=early_stop_thres,
                                    pruning=pruning,
                                    loss_threshold=loss_threshold,
                                    elimination_threshold=elimination_threshold,
                                    random_state=random_state)

    def _validate_input(self, x, y):
        
        """method to validate data
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        """
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
        
        """fit the SimBoost model

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        """

        n_samples = x.shape[0]
        val_fold = np.ones((n_samples))
        val_fold[self.tr_idx] = -1
        
        # Initialize the intercept
        z = y.copy() * 4 - 2
        self.intercept_ = np.mean(z)
        pred_train = self.intercept_ * np.ones(len(self.tr_idx))
        pred_val = self.intercept_ * np.ones(len(self.val_idx))
        proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
        proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

        sample_weight[self.tr_idx] = proba_train * (1 - proba_train)
        sample_weight[self.tr_idx] /= np.sum(sample_weight[self.tr_idx])
        sample_weight[self.tr_idx] = np.maximum(sample_weight[self.tr_idx], 2 * np.finfo(np.float64).eps)

        with np.errstate(divide="ignore", over="ignore"):
            z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                            -1. / (1. - np.hstack([proba_train, proba_val]))) 
            z = np.clip(z, a_min=-8, a_max=8)

        # Fit categorical variables
        if self.cfeature_num_ > 0:
            self._fit_dummy(x[self.tr_idx], z[self.tr_idx], sample_weight[self.tr_idx])
            pred_train = np.sum([est.predict(x[self.tr_idx]) for est in self.dummy_estimators_], axis=0) + self.intercept_
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            pred_val = np.sum([est.predict(x[self.val_idx]) for est in self.dummy_estimators_], axis=0) + self.intercept_
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

        # Fit Sim Boosting for numerical variables
        if self.nfeature_num_ == 0:
            return 

        auc_opt = 0
        early_stop_count = 0
        for indice in range(self.n_estimators):
            sample_weight[self.tr_idx] = proba_train * (1 - proba_train)
            sample_weight[self.tr_idx] /= np.sum(sample_weight[self.tr_idx])
            sample_weight[self.tr_idx] = np.maximum(sample_weight[self.tr_idx], 2 * np.finfo(np.float64).eps)

            with np.errstate(divide="ignore", over="ignore"):
                z = np.where(y.ravel(), 1. / np.hstack([proba_train, proba_val]),
                                -1. / (1. - np.hstack([proba_train, proba_val]))) 
                z = np.clip(z, a_min=-8, a_max=8)

            # projection matrix
            if self.learning_rate == 1:
                if (indice == 0) or (indice >= self.nfeature_num_) or (self.ortho_shrink == 0):
                    proj_mat = np.eye(self.nfeature_num_)
                else:
                    projection_indices_ = np.array([est["sim"].beta_.flatten() for est in self.sim_estimators_]).T
                    u, _, _ = np.linalg.svd(projection_indices_, full_matrices=False)
                    proj_mat = np.eye(u.shape[0]) - self.ortho_shrink * np.dot(u, u.T)
            else:
                proj_mat = None

            # fit Sim estimator
            param_grid = {"method": self.prjection_method_list, 
                      "reg_lambda": self.reg_lambda_list,
                      "reg_gamma": self.reg_gamma_list}
            grid = GridSearchCV(SimRegressor(degree=self.degree, knot_num=self.knot_num, spline=self.spline,
                                  knot_dist=self.knot_dist, random_state=self.random_state), 
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
            sim_estimator["sim"].fit_middle_update(x[:, self.nfeature_index_list_], z, 
                           sample_weight=sample_weight, proj_mat=proj_mat, val_ratio=self.val_ratio, **self.middle_update)
                        
            pred_train += self.learning_rates[indice] * sim_estimator.predict(x[self.tr_idx])
            proba_train = 1 / (1 + np.exp(-pred_train.ravel()))
            pred_val += self.learning_rates[indice] * sim_estimator.predict(x[self.val_idx])
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))

            auc_new = roc_auc_score(y[self.val_idx], proba_val)
            if auc_opt < auc_new:           
                auc_opt = auc_new
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.early_stop_thres:
                break

            self.sim_estimators_.append(sim_estimator)
    
    def _pruning(self, x, y):
          
        """prune the base learners that are not importnat
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples, 1)
            containing the output dataset
        """
        
        component_importance = {}
        for indice, est in enumerate(self.sim_estimators_):
            component_importance.update({"sim " + str(indice + 1): {"type": "sim", "indice": indice,
                                "importance": np.std(self.learning_rate * est.predict(x[self.tr_idx, :]))}})

        for indice, est in enumerate(self.dummy_estimators_):
            feature_name = list(est.named_steps.keys())[0]
            component_importance.update({feature_name: {"type": "dummy_lr", "indice": indice,
                                "importance": np.std(self.learning_rate * est.predict(x[self.tr_idx, :]))}})
    
        self.weights_ = []
        self.estimators_ = []
        pred_val = self.intercept_ + np.zeros(len(self.val_idx))
        proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
        self.val_auc_ = [roc_auc_score(y[self.val_idx], pred_val)]
        for key, item in sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1]:

            if item["type"] == "sim":
                est = self.sim_estimators_[item["indice"]]
                pred_val += self.learning_rates[item["indice"]] * est.predict(x[self.val_idx])
                self.weights_.append(self.learning_rates[item["indice"]])
            elif item["type"] == "dummy_lr":
                est = self.dummy_estimators_[item["indice"]]
                pred_val += est.predict(x[self.val_idx])
                self.weights_.append(1)
                
            self.estimators_.append(est)
            pred_val += est.predict(x[self.val_idx])
            proba_val = 1 / (1 + np.exp(-pred_val.ravel()))
            self.val_auc_.append(roc_auc_score(y[self.val_idx], proba_val))

        if not self.pruning:
            best_idx = len(self.val_auc_) - 1
        else:
            best_auc = np.max(self.val_auc_)
            if np.sum((1 - self.val_auc_ / best_auc) < self.loss_threshold) > 0:
                best_idx = np.where((1 - self.val_auc_ / best_auc) < self.loss_threshold)[0][0]
            else:
                best_idx = np.argmax(self.val_auc_)

        self.best_weights_ = self.weights_[:best_idx]
        self.best_estimators_ = self.estimators_[:best_idx]
        self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:best_idx])
        self.activate_cfeature_index_ = [est[0].kw_args["idx"] for est in self.best_estimators_ if "dummy_lr" in est.named_steps.keys()]

        if self.elimination_threshold:
            eli_idx = np.sum([True if item['ir']>self.elimination_threshold else False for key,item in self.importance_ratios_.items()])
            self.best_weights_ = self.weights_[:eli_idx]
            self.best_estimators_ = self.estimators_[:eli_idx]
            self.component_importance_ = dict(sorted(component_importance.items(), key=lambda item: item[1]["importance"])[::-1][:eli_idx])
            self.activate_cfeature_index_ = [est[0].kw_args["idx"] for est in self.best_estimators_ if "dummy_lr" in est.named_steps.keys()]


    def predict_proba(self, x):

        """output probability prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples, 2)
            containing probability prediction
        """

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)
        return pred_proba

    def predict(self, x):

        """output binary prediction for given samples
        
        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing binary prediction
        """  

        pred_proba = self.predict_proba(x)[:, 1]
        return self._label_binarizer.inverse_transform(pred_proba).reshape([-1, 1])