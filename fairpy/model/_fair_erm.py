# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Union, Any

from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.multiclass import type_of_target
from sklearn.svm import SVC
# import gurobipy as gb

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, HasMethods, Interval


def linear_kernel(x1, x2):
    return np.dot(x1, np.transpose(x2))


class LinearFairERM(FairEstimator):
    """
    Empirical risk minimization under fairness constraints

    Reference:
        https://proceedings.neurips.cc/paper/2018/file/83cdcec08fbf90370fcf53bdd56604ff-Paper.pdf
    Code adopted from:
        https://github.com/jmikko/fair_ERM

    Attributes
    ----------
    s_classes_ : numpy array of shape (n_sensitive_group, )
        a list of sensitive classes known to LabelBias during training

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from fairpy.dataset import Adult
    >>> from fairpy.model import LinearFairERM
    >>> dataset = Adult()
    >>> split_data = dataset.split()
    >>> model = LinearFairERM()
    >>> model.fit(split_data.X_train, split_data.y_train, split_data.s_train)
    >>> model.predict(split_data.X_test)
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit"]), None],
    }

    def __init__(self, estimator: Any = None):
        self.estimator = estimator

    def feat_trans(self, X: np.ndarray) -> np.ndarray:
        trans_X = X - np.outer((X[:, self._max_idx] / self._u[self._max_idx]), self._u)
        trans_X = np.delete(trans_X, self._max_idx, 1)
        return trans_X

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[Sequence, np.ndarray],
            s: Union[Sequence, np.ndarray],
    ) -> LinearFairERM:
        """
        Parameters
        ----------
        """

        self._validate_params()
        if self.estimator is None:
            model = SVC(kernel="linear")
        else:
            model = self.estimator

        X, y = self._validate_data(X, y)

        y_type = type_of_target(y)
        if y_type != "binary":
            raise ValueError("LabelBias only support binary target variables, but got {0} instead".format(y_type))
        self.classes_ = np.unique(y)

        s_ind = self._validate_grp_s(s, ("binary"))

        idx_grp_0 = np.where(s_ind == 0)
        idx_grp_1 = np.where(s_ind == 1)
        idx_y_pos = np.where(y == self.classes_[1])
        idx_y_pos_grp_0 = np.intersect1d(idx_grp_0, idx_y_pos)
        idx_y_pos_grp_1 = np.intersect1d(idx_grp_1, idx_y_pos)

        feat_grp_0 = np.mean(X[idx_y_pos_grp_0], axis=0)
        feat_grp_1 = np.mean(X[idx_y_pos_grp_1], axis=0)

        self._u = feat_grp_1 - feat_grp_0
        self._max_idx = np.argmax(self._u)

        trans_X = self.feat_trans(X)
        model.fit(trans_X, y)

        return self

    def predict(self, X) -> Any:
        check_is_fitted(self)

        if not hasattr(self.estimator, "predict"):
            raise ValueError("Estimator is expected to have method 'predict'")

        X = self._validate_data(X, reset=False)
        trans_X = self.feat_trans(X)
        return self.estimator.predict(trans_X)

    def predict_proba(self, X) -> Any:
        check_is_fitted(self)

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("Estimator is expected to have method 'predict_proba'")

        X = self._validate_data(X, reset=False)
        trans_X = self.feat_trans(X)
        return self.estimator.predict_proba(trans_X)

    def predict_log_proba(self, X) -> Any:
        check_is_fitted(self)

        if not hasattr(self.estimator, "predict_log_proba"):
            raise ValueError("Estimator is expected to have method 'predict_proba'")

        X = self._validate_data(X, reset=False)
        trans_X = self.feat_trans(X)
        return self.estimator.predict_log_proba(trans_X)

    def decision_function(self, X) -> Any:
        check_is_fitted(self)

        if not hasattr(self.estimator, "decision_function"):
            raise ValueError("Estimator is expected to have method 'decision_function'")

        X = self._validate_data(X, reset=False)
        trans_X = self.feat_trans(X)
        return self.estimator.decision_function(trans_X)

    def _more_tags(self):
        return {
            'binary_only': True,
            'requires_y': True,
        }


class FairERM(FairEstimator):
    """
    Empirical risk minimization under fairness constraints

    Reference:
        https://proceedings.neurips.cc/paper/2018/file/83cdcec08fbf90370fcf53bdd56604ff-Paper.pdf
    Code adopted from:
        https://github.com/jmikko/fair_ERM

    # TODO: not working and unscalable

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from fairpy.dataset import Adult
    >>> from fairpy.model import FairERM
    >>> dataset = Adult()
    >>> split_data = dataset.split()
    >>> model = FairERM()
    >>> model.fit(split_data.X_train, split_data.y_train, split_data.s_train)
    >>> model.predict(split_data.X_test)
    """

    def __init__(
            self,
            kernel: str = "rbf",
            C: float = 1.,
            gamma: float = 1.,
    ):
        """
        Parameters
        ----------
        metric : {"rbf", "linear"}, default="rbf"
        """

        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[Sequence, np.ndarray],
            s: Union[Sequence, np.ndarray],
    ):
        if self.kernel == "rbf":
            self.fkernel = lambda x, y: rbf_kernel(x, y, self.gamma)
        elif self.kernel == "linear":
            self.fkernel = linear_kernel
        else:
            raise ValueError("FairERM does not support {0} kernel".format(self.kernel))

        # TODO: validate X and y
        y = np.where(y == 0, -1, y)

        n_samples, n_features = X.shape
        s_ind = self._validate_grp_s(s, ("binary"))

        idx_grp_0 = np.where(s_ind == 0)
        idx_grp_1 = np.where(s_ind == 1)
        idx_y_pos = np.where(y == 1)
        idx_y_pos_grp_0 = np.intersect1d(idx_grp_0, idx_y_pos)
        idx_y_pos_grp_1 = np.intersect1d(idx_grp_1, idx_y_pos)

        K = self.fkernel(X, X)
        P = np.multiply(np.outer(y, y), K)
        q = (np.ones(n_samples) * -1)

        # TODO: slight difference between these two expressions
        # tau = [(np.sum(K[idx_y_pos_grp_1, idx]) / len(idx_y_pos_grp_1)) -
        #        (np.sum(K[idx_y_pos_grp_0, idx]) / len(idx_y_pos_grp_0)) for idx in range(len(y))]
        tau = np.sum(K[idx_y_pos_grp_1, :], axis=0) / len(idx_y_pos_grp_1) - \
              np.sum(K[idx_y_pos_grp_0, :], axis=0) / len(idx_y_pos_grp_0)
        fairness_line = np.multiply(y, tau).reshape(1, -1)
        A = np.vstack([y, fairness_line])
        b = np.array([0., 0.])

        if self.C is None:
            G = np.diag(np.ones(n_samples) * -1)
            h = np.zeros(n_samples)
        else:
            G = np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples)))
            h = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))

        # Quadratic programming
        # model = gb.Model("qp")
        # x = model.addMVar(shape=(n_samples,))
        # model.setObjective(x @ P @ x * 0.5 + q @ x)
        # model.addConstr(G @ x <= h, name="inequality")
        # model.addConstr(A @ x == b, name="equality")
        # model.optimize()

        x = np.ravel(x.X)
        ind = np.where(x > 1e-7)[0]

        self.x = x[ind]
        self.sv = X[ind]
        self.sv_y = y[ind]

        self.b = np.sum(self.sv_y)
        self.b -= np.sum(self.x * self.sv_y * np.sum(K[ind, :], axis=0)[ind])
        self.b /= len(ind)

        if self.kernel == linear_kernel:
            self.w = (self.x * self.sv_y) @ self.sv
        else:
            self.w = None

        return self

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            XSV = self.fkernel(X, self.sv)
            a_sv_y = np.multiply(self.x, self.sv_y)
            y_predict = [np.sum(np.multiply(np.multiply(self.x, self.sv_y), XSV[i, :])) for i in range(len(X))]

            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

    def _more_tags(self):
        return {
            'binary_only': True,
            'requires_y': True,
        }
