from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
import numpy as np
from sklearn.datasets import load_svmlight_file
import loss
import regularizer


def load_problem(X, y, problem_type = "classification",
                loss_type = "logistic", regularizer_type = "L2", 
                bias_term = True, scale_features = True,
                center_features = False):
    if problem_type == "classification":
        # X = X / np.max(np.abs(X), axis=0)  # X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
        # label preprocessing for some dataset whose label is not {-1, 1}
        max_v, min_v = np.max(y), np.min(y)
        idx_min = (y == min_v)
        idx_max = (y == max_v)
        y[idx_min] = -1
        y[idx_max] = 1
    elif problem_type != "regression":
        raise Exception("Unknown problem type!")

    if center_features:
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    if scale_features:
        X = X / np.max(np.abs(X), axis=0)
    if bias_term:# adding a column filling with all ones.(bias term)
        X = np.c_[X, np.ones(X.shape[0])]

    if loss_type == "L2":
        criterion = loss.L2()
    elif loss_type == "PseudoHuber":
        criterion = loss.PseudoHuberLoss(delta=1.0)
    elif loss_type == "Logistic":
        criterion = loss.LogisticLoss()
    else:
        raise Exception("Unknown loss function!")

    if regularizer_type == "L2":
        penalty = regularizer.L2()
    elif regularizer_type == "PseudoHuber":
        penalty = regularizer.PseudoHuber(delta=1.0)
    else:
        raise Exception("Unknown regularizer type!")

    return criterion, penalty, X, y
# ======================================


def get_data(data_path):
    # This function is taken from the code of Rui YUAN
    """Once datasets are downloaded, load datasets."""
    data = load_svmlight_file(data_path)
    return data[0], data[1]

def sparsity(data):
    # data, (num_samples, num_features)
    n, d = data.shape
    total_entries = n * d
    zeros_entries = np.sum(data == 0)
    return zeros_entries / total_entries

# ===============================
# These two functions to generate artificial data
# are taken from the course M2-Optimization for Data Science
# =================================


def simu_linreg(x, n, std=1., corr=0.5):
    """Simulation for the least-squares problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    b = A.dot(x) + noise
    return A, b


def simu_logreg(x, n, std=1., corr=0.5):
    """Simulation for the logistic regression problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    A, b = simu_linreg(x, n, std=1., corr=corr)
    return A, np.sign(b)


