Examples
===============
Here we give more example usage of this package.


Sim for Regression
--------------------------------

.. code-block::

        import numpy as np
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import make_scorer, mean_squared_error

        from pysim import SimRegressor

        ## generate simulation data
        s_star = 5
        n_features = 100
        n_samples = 10000

        np.random.seed(1)
        beta = np.zeros(n_features)
        supp_ids = np.random.choice(n_features, s_star)
        beta[supp_ids]=np.random.choice((-1, 1), s_star) / np.sqrt(s_star)

        x = np.random.normal(0, 0.3, size=(n_samples, n_features))
        y = np.sin(np.pi*(np.dot(x, beta))) + 0.1 * np.random.randn(n_samples)

        ## fit sim with hyperparameter tunning
        param_grid = {"method": ["first_order", "second_order", "first_order_thres", "marginal_regression", "ols"],
                      "knot_dist": ["uniform", "quantile"],
                      "reg_lambda": [0.1, 0.2, 0.3, 0.4, 0.5], 
                      "reg_gamma": [0.2, 0.4, 0.6, 0.8, 1.0]}
        grid = GridSearchCV(SimRegressor(spline="smoothing_spline", knot_num=20, random_state=0), iid=False,
                            cv=KFold(3, shuffle=True, random_state=0), param_grid=param_grid, n_jobs=-1, verbose=2, error_score=np.nan)
        grid.fit(x, y)
        
        ## visualize the fitted model
        clf = grid.best_estimator_
        clf.visualize()

        ## compare with the true projection indice
        plt.plot(np.abs(clf.beta_), "o")
        plt.plot(np.abs(beta), "o")
        plt.legend(["Estimated", "Ground Truth"])
        plt.show()
        
        ## fine tune the fitted sim model to improve performance
        clf.fit_inner_update(x, y, method="adam", n_inner_iter_no_change=1, batch_size=1000, verbose=True)
        clf.visualize()
        
        
Sim for Classification
-----------------------------------

.. code-block::

        import numpy as np
        from matplotlib import pylab as plt
        from sklearn.model_selection import KFold
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import make_scorer, roc_auc_score

        from pysim import SimClassifier
        
        ## generate simulation data
        s_star = 5
        n_features = 100
        n_samples = 10000

        np.random.seed(1)
        beta = np.zeros(n_features)
        supp_ids = np.random.choice(n_features, s_star)
        beta[supp_ids]=np.random.choice((-1, 1), s_star) / np.sqrt(s_star)

        x = np.random.normal(0, 0.3, size=(n_samples, n_features))
        y = 1 / (1 + np.exp(-(np.dot(x, beta)))) + 0.1 * np.random.randn(n_samples)
        y = y - np.mean(y)
        y[y <= 0] = 0
        y[y > 0] = 1
        
        ## fit sim with hyperparameter tunning
        param_grid = {"method": ["first_order", "second_order", "first_order_thres", "marginal_regression", "ols"],
                      "knot_dist": ["uniform", "quantile"],
                      "reg_lambda": [0.1, 0.2, 0.3, 0.4, 0.5], 
                      "reg_gamma": [0.2, 0.4, 0.6, 0.8, 1.0]}
        grid = GridSearchCV(SimClassifier(spline="smoothing_spline", knot_num=20, random_state=0), iid=False,
                            cv=KFold(3, shuffle=True, random_state=0), param_grid=param_grid, n_jobs=-1, verbose=2, error_score=np.nan)
        grid.fit(x, y) 
        
        ## visualize the fitted model
        clf = grid.best_estimator_
        clf.visualize()

        ## compare with the true projection indice
        plt.plot(np.abs(clf.beta_), "o")
        plt.plot(np.abs(beta), "o")
        plt.legend(["Estimated", "Ground Truth"])
        plt.show()
        
        ## fine tune the fitted sim model to improve performance
        clf.fit_inner_update(x, y, method="adam", n_inner_iter_no_change=1, batch_size=1000, verbose=True)
        clf.visualize()


Sim Boosting
-----------------------------------

.. code-block::

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import truncnorm
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, roc_auc_score
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        from pysim import SimBoostRegressor, SimBoostClassifier
        
        ## generate simulation data
        random_state = 1
        np.random.seed(random_state)
        # data generation
        beta1 = np.array([.2, 0.3, 0.5, 0, 0, 0, 0, 0, 0, 0])
        beta2 = np.array([0, .2, 0.3, 0.5, 0, 0, 0, 0, 0, 0])
        beta3 = np.array([0, 0, 0.2, 0.3, 0.5, 0, 0, 0, 0, 0])

        beta = np.vstack([beta1, beta2, beta3])
        model_list = [lambda x: 0.2 * np.exp(-4 * x), lambda x: 3 * x ** 2, lambda x: 2.5 * np.sin(1.5 * np.pi * x)]

        x = truncnorm.rvs(a=-3, b=3, loc = 0, scale=1 / 3, size=(20000, 10), random_state=random_state)
        noise = np.random.randn(20000).reshape(-1, 1)
        y = np.reshape(0.2 * np.exp(-4 * np.dot(x, beta1)) + \
                       3 * (np.dot(x, beta2)) ** 2 + 2.5 * np.sin(np.pi * 1.5 * np.dot(x, beta3)), [-1, 1]) + noise
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=random_state)
        
        clf = SimBoostRegressor(n_estimators=50, knot_num=10, knot_dist="quantile", spline="smoothing_spline", learning_rate=1,
                        reg_lambda=[0.1, 0.2, 0.3, 0.4, 0.5],
                        reg_gamma=[1e-9, 1e-6, 1e-3], inner_update="bfgs", meta_info=None, pruning=False)
        clf.fit(train_x, train_y)
        
        clf.visualize()
        clf.validation_performance()