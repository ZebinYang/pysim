# Sparse Single-index Model

**To install**:
    
```sheel
    pip install git+https://github.com/Zebinyang/pysim.git
```

Note pysim will call the R package fps (https://github.com/vqv/fps) using rpy2 interface. 

**Usage**

```python
from pysim import SimClassifier
clf = SimClassifier(method="first_order", spline="smoothing_spline_mgcv", reg_lambda=0.1, reg_gamma=10, knot_num=20, knot_dist="uniform", degree=2, random_state=0)
## clf.fit(x, y)
```

**Hyperparameters**

- method: the base method for estimating the projection coefficients in sparse SIM. default="first_order"

        "first_order": First-order Stein's Identity via sparse PCA solver

        "second_order": Second-order Stein's Identity via sparse PCA solver

        "first_order_thres": First-order Stein's Identity via hard thresholding (A simplified verison)     
        
        "marginal_regression": Marginal regression subject to hard thresholding
        
        "ols": Least squares estimation subject to hard thresholding.
        
- reg_lambda: The regularization strength of sparsity of beta. default=0.1, from 0 to 1 

- spline: The type of spline for fitting the curve. default="smoothing_spline_mgcv"
        
        "smoothing_spline_bigsplines": Smoothing spline based on bigsplines package in R

        "smoothing_spline_mgcv": Smoothing spline based on mgcv package in R

        "p_spline": P-spline

        "mono_p_spline": P-spline with monotonic constraint

        "a_spline": Adaptive B-spline

- reg_gamma: The regularization strength of the spline algorithm. default=0.1. 

        For spline="smoothing_spline_bigsplines", it ranges from 0 to 1, and the suggested tuning grid is 1e-9 to 1e-1; and it can be set to "GCV".

        For spline="smoothing_spline_mgcv", it ranges from 0 to :math:`+\infty`, and it can be set to "GCV".

        For spline="p_spline","mono_p_spline" or "a_spline", it ranges from 0 to :math:`+\infty`

- degree: The order of the spline basis. default=3; 
        
        For spline="smoothing_spline_bigsplines", possible values include 1 and 3.
    
        For spline="smoothing_spline_mgcv", possible values include 3, 4, ....

- knot_num: The number of knots. default=10

- knot_dist: The method of specifying the knots. default="quantile"

        "uniform": uniformly over the domain
        
        "quantile": uniform quantiles of the given input data (not available when spline="p_spline" or "mono_p_spline")

- random_state: the random seed. default=0
