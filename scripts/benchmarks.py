import time
import numpy as np

from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages('simest')
simest = importr('simest')
numpy2ri.activate()

from pysim import SIM


def lasso(train_x, train_y, test_x, val_ratio=0.2, rand_seed=0):
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))
    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    val_fold = np.ones((len(indices)))
    val_fold[idx1] = -1

    grid = GridSearchCV(Lasso(random_state=rand_seed), param_grid={'alpha': np.logspace(-2, 2, 5)},
                        scoring={'mse': make_scorer(mean_squared_error, greater_is_better=False)},
                        cv=PredefinedSplit(val_fold), refit='mse', n_jobs=10, error_score=-np.inf)
    grid.fit(train_x, train_y.ravel())
    model = grid.best_estimator_
    start = time.time()
    model.fit(train_x[idx1, :], train_y[idx1].ravel())
    time_cost = time.time() - start
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost


def rf(train_x, train_y, test_x, val_ratio=0.2, rand_seed=0):

    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    val_fold = np.ones((len(indices)))
    val_fold[idx1] = -1

    base = RandomForestRegressor(n_estimators=100, random_state=rand_seed)
    grid = GridSearchCV(base, param_grid={'max_depth': (3, 4, 5, 6, 7, 8)},
                        scoring={'mse': make_scorer(mean_squared_error, greater_is_better=False)},
                        cv=PredefinedSplit(val_fold), refit='mse', error_score=-np.inf)
    grid.fit(train_x, train_y.ravel())
    model = grid.best_estimator_
    start = time.time()
    model.fit(train_x[idx1, :], train_y[idx1].ravel())
    time_cost = time.time() - start
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost


def mlp(train_x, train_y, test_x, val_ratio=0.2, epoches=10000, early_stop=500, rand_seed=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))
    
    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    model = MLPRegressor(hidden_layer_sizes=[100, 60], max_iter=epoches, batch_size=min(500, int(np.floor(datanum * 0.20))), \
                  activation='tanh', early_stopping=True,
                  random_state=rand_seed, validation_fraction=val_ratio, n_iter_no_change=early_stop)
    start = time.time()
    model.fit(train_x, train_y.ravel())
    time_cost = time.time() - start
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost


def svm(train_x, train_y, test_x, val_ratio=0.2, rand_seed=0):
 
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))
    
    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    val_fold = np.ones((len(indices)))
    val_fold[idx1] = -1

    grid = GridSearchCV(SVR(kernel='rbf'), param_grid={"C": np.logspace(-2, 2, 5),
                                      "gamma": np.logspace(-2, 2, 5)},
                        scoring={'mse': make_scorer(mean_squared_error, greater_is_better=False)},
                        cv=PredefinedSplit(val_fold), refit='mse', n_jobs=10, error_score=-np.inf)
    grid.fit(train_x, train_y.ravel())
    model = grid.best_estimator_
    start = time.time()
    model.fit(train_x[idx1, :], train_y[idx1].ravel())
    time_cost = time.time() - start
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost


def sim_est(train_x, train_y, test_x, val_ratio=0.2, rand_seed=0):

    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    val_fold = np.ones((len(indices)))
    val_fold[idx1] = -1

    scores = []
    clf_list = []
    time_cost_list = []
    lambda_list = np.logspace(-2, 2, 5)
    for lamda in lambda_list:
        start = time.time()
        clf = simest.sim_est(train_x[idx1, :], train_y[idx1], ro.NULL, ro.NULL, 1, ro.NULL,
                      lamda, method="smooth.pen", progress=False)
        pred = simest.predict_sim_est(clf, train_x[idx2, :])
        clf_list.append(clf)
        time_cost_list.append(time.time() - start)
        scores.append(mean_squared_error(pred, train_y[idx2]))
    best_indice = np.where(scores==np.min(scores))[0][0]
    model = clf_list[best_indice]
    best_lambda = lambda_list[best_indice]
    time_cost = time_cost_list[best_indice]
    pred_train = np.array(simest.predict_sim_est(model, train_x[idx1, :]))
    pred_val = np.array(simest.predict_sim_est(model, train_x[idx2, :]))
    pred_test = np.array(simest.predict_sim_est(model, test_x))
        
    # visualize the beta of simest
    # plt.scatter(np.array(model[13]), np.array(model[15]))
    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost


def py_sim(train_x, train_y, test_x, val_ratio=0.2, rand_seed=0):

    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=rand_seed)
    val_fold = np.ones((len(indices)))
    val_fold[idx1] = -1

    param_grid = {"method": ["first", "second"],
                  "reg_lambda": [0.01, 0.05, 0.1],
                  "reg_gamma": np.logspace(-2, 2, 5)}
    grid = GridSearchCV(SIM(mu=train_x.mean(0), sigma=train_x.std(0), degree=2, knot_num=20, spline="bs", random_state=0), 
                  scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)}, refit="mse",
                  cv=PredefinedSplit(val_fold), param_grid=param_grid, n_jobs=1, verbose=0, error_score=np.nan)
    grid.fit(train_x, train_y)
    model = grid.best_estimator_
    start = time.time()
    model.fit(train_x[idx1, :], train_y[idx1].ravel())
    time_cost = time.time() - start
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])        
    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost
