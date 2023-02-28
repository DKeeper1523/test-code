# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

import numpy as np
import pandas as pd
from numbers import Real
from typing import Sequence, Union, Any

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.extmath import log_logistic
import scipy

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, HasMethods, Interval
from ..utils.encode import onehottify


class FairCstr(FairEstimator):
    """
    Fairness Constraints: Mechanisms for Fair Classification

    Support single sensitive attribute with binary values.
    # TODO: add support for multiple sensitive values and attributes

    Reference:
        http://proceedings.mlr.press/v54/zafar17a/zafar17a.pdf
    Code adopted from:
        https://github.com/mbilalzafar/fair-classification
    """

    _parameter_constraints = {
        "cstr": [StrOptions({"fair", "acc"})],
        "sep_cstr": ["boolean"],
        "max_iter": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
            self,
            cstr: str = "fair",
            sep_cstr: bool = True,
            max_iter: int = 100000,
    ):
        self.cstr = cstr
        self.sep_cstr = sep_cstr
        self.max_iter = max_iter

    def _logistic_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        yz = y * np.dot(X, w)
        loss = -np.sum(log_logistic(yz))
        return loss

    def _s_cov(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, s_one_col: np.ndarray, thresh: float = 0.):
        if w is None:
            arr = y
        else:
            arr = np.dot(w, X.T)

        cov = np.dot(s_one_col - np.mean(s_one_col), arr) / len(s_one_col)
        return thresh - abs(cov)

    def _get_cstr_cov(self, X: np.ndarray, y: np.ndarray, s_one_hot: np.ndarray):
        constraints = [{"type": "ineq", "fun": self._s_cov, "args": (X, y, s_one_hot)}]
        return constraints

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[Sequence, np.ndarray],
            s: Union[Sequence, np.ndarray],
    ) -> FairCstr:
        """
        Parameters
        ----------
        """

        self._validate_params()

        X, y = self._validate_data(X, y)
        s_ind = self._validate_grp_s(s, ("binary", "multiclass"))
        # s_one_hot = onehottify(s_ind)

        if self.cstr == "fair":
            constraints = self._get_cstr_cov(X, y, s_ind)

        w = scipy.optimize.minimize(
            fun=self._logistic_loss,
            x0=np.random.rand(X.shape[1], ),
            args=(X, y),
            method='SLSQP',
            options={"maxiter": self.max_iter},
            constraints=constraints,
        )
        self.coef_ = w.x

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self)
        return np.sign(np.dot(X, self.coef_))
