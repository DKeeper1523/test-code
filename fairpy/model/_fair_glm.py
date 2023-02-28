# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import numpy as np
import numpy.typing as npt
import pandas as pd
from copy import deepcopy
from itertools import combinations, product
from numbers import Real, Integral, Rational
from typing import Sequence, Union, Any, Optional, Dict, NewType, Tuple, Callable, Mapping, List

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import type_of_target

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, HasMethods, Interval


class FairGLM(FairEstimator):
    """
    Fair Generalized Linear Models with a Convex Penalty

    Reference:
        https://proceedings.mlr.press/v162/do22a/do22a.pdf
    Code adopted from:
        https://github.com/hyungrok-do/fair-glm-cvx

    # TODO: more family
    """

    _parameter_constraints = {
        "family": [StrOptions({"bernoulli", "multinomial"})],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "lam": [Interval(Real, 0, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "M": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
            self,
            family: str = "bernoulli",
            fit_intercept: bool = True,
            max_iter: int = 100,
            lam: float = 0.,
            tol: float = 1e-4,
            M: int = 50,
    ):
        self.family = family
        self.lam = lam
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.M = M

    def discretization(self, y: npt.NDArray, s: npt.NDArray):
        YD = None
        if self.family == "bernoulli":
            YD = y
        elif self.family == "multinomial":
            YD = y.argmax(1)

        YD = OrdinalEncoder().fit_transform(YD.reshape(-1, 1)).flatten()

        return YD

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, s: npt.ArrayLike):
        self._validate_params()

        X, y = self._validate_data(X, y)
        y_type = type_of_target(y)
        if y_type not in ("binary", "multiclass"):
            raise ValueError(
                "FairGLM only support binary or multiclass target variables, but got {0} instead".format(y_type))
        self.classes_ = np.unique(y)

        s_ind = self._validate_grp_s(s, ("binary", "multiclass"))

        N, P = X.shape
        YD = self.discretization(y, s)

        D = np.zeros((P, P))
        if self.lam > 0:
            for (a, b), yd in product(combinations(self.s_classes_, 2), np.unique(YD)):
                Xay = X[np.logical_and(s_ind == a, YD == yd)]
                Xby = X[np.logical_and(s_ind == b, YD == yd)]
                diff = (Xay[None, :, :] - Xby[:, None, :]).reshape(-1, P)

                D += diff.T @ diff / (np.float64(len(Xay)) * np.float64(len(Xby)))

            D = (self.lam * 2) / (len(np.unique(YD)) * len(self.s_classes_) * (len(self.s_classes_) - 1)) * D

        beta = [np.zeros(P)]
        if self.family == 'bernoulli':
            for i in range(self.max_iter):
                mu = 1. / (1 + np.exp(np.clip(-np.dot(X, beta[i]), -self.M, self.M)))
                grad = -X.T.dot(y - mu) / N + np.dot(D, beta[i])
                w = np.diag(mu * (1 - mu))

                # TODO: mat sometimes not invertible
                hinv = np.linalg.inv(X.T @ w @ X / N + D)
                beta.append(beta[i] - np.dot(hinv, grad))

                if np.linalg.norm(grad) < self.tol:
                    break

        self.coef_traj = beta
        self.coef_ = beta[-1]

    def predict(self, X: npt.ArrayLike):
        proba = self.predict_proba(X)
        return int(proba > 0.5)

    def predict_proba(self, X: npt.ArrayLike):
        if self.family == "bernoulli":
            Xb = np.dot(X, self.coef_)
            p = 1 / (1 + np.exp(-Xb))
            return np.column_stack([1 - p, p])
