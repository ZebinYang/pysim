# Sparse Single-index Model based on Stein's Identity

**To install**:
    
```sheel
    pip install git+https://github.com/Zebinyang/pysim.git
```

Note pysim will call the R package fps (https://github.com/vqv/fps) using rpy2 interface. 

**Usage**

```python
from pysim import SIM
clf = SIM(method="first", spline="augbs", reg_lambda=0.1, reg_gamma=0.1, knot_num=20, degree=2, random_state=0)
## clf.fit(x, y)
```

**Hyperparameters**
- task: the task type, including "Regression" and "Classification". default="Regression"
- method: the base method for estimating the projection coefficients in sparse SIM. default="first"

        "first": First-order Stein's Identity via sparse PCA solver

        "second": Second-order Stein's Identity via sparse PCA solver

        "first_thresholding": First-order Stein's Identity via hard thresholding (A simplified verison)        
    
- spline: The type of spline method. default="augbs"

        "ps": p-spline (from pygam package)
    
        "mono": p-spline with monotonic constraint (from pygam package)
    
        "augbs": adaptive spline where knots are automatically selected (see Goepp, V., Bouaziz, O. and Nuel, G., 2018. Spline regression with automatic knot selection. arXiv preprint arXiv:1808.01770.)

- reg_lambda: The regularization strength of sparsity of beta. default=0.1, from 0 to 1 

- reg_gamma: The regularization strength of the spline algorithm. default=0.1, from 0 to $+\infty$

- degree: The order of the spline basis. default=2

- knot_num: The number of knots spanned uniformly over the domain. default=20

- random_state: the random seed. default=0
