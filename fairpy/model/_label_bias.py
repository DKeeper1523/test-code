# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

from numbers import Real, Integral
from typing import Sequence, Union, Any

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression

from ._base import FairEstimator
from ..utils.encode import onehottify
from ..utils._param_validation import StrOptions, HasMethods, Interval


class LabelBias(FairEstimator):
    """
    Identifying and Correcting Label Bias in Machine Learning

    Adaptively learn the weights for sensitive groups by fitting the sub-estimator multiple times

    Support single sensitive attribute with binary/multiple values.
    # TODO: add support for multiple sensitive attributes with multiple values
    # TODO: add support for equal odds

    Reference:
        http://proceedings.mlr.press/v108/jiang20a/jiang20a.pdf
    Code adopted from:
        https://github.com/google-research/google-research/tree/master/label_bias

    Attributes
    ----------
    s_classes_ : numpy array of shape (n_sensitive_group, )
        a list of sensitive classes known to LabelBias during training

    weights_ : numpy array of shape (n_sample, )
        weights for training samples solved by LabelBias

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from fairpy.dataset import Adult
    >>> from fairpy.model import LabelBias
    >>> dataset = Adult()
    >>> split_data = dataset.split()
    >>> model = LabelBias()
    >>> model.fit(split_data.X_train, split_data.y_train, split_data.s_train)
    >>> model.predict(split_data.X_test)
    """

    _parameter_constraints = {
        "metric": [StrOptions({"dp", "eop"})],
        "estimator": [HasMethods(["fit", "predict"]), None],
        "max_iter": [Interval(Integral, 0, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "lr": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
            self,
            metric: str = "dp",
            estimator: Any = None,
            max_iter: int = 100,
            tol: float = 1e-3,
            lr: float = 1.
    ):
        """
        Parameters
        ----------
        metric : {"dp", "eop"}, default="dp"
            The fairness notion adopted in optimization.
            dp: demographic parity
            eop: equal opportunity

        estimator : a callable object with some requirements, default=None
            Estimator is the base classifier, and LabelBias is a warpper function over this classifier.
            The estimator should implement 'fit' and 'predict' methods (and 'predict_proba' if needed) for
            model's training and prediction.
            If not specific, use Logistic Regression as the sub-estimator.

        max_iter : int, default=100
            The maximum iteration for optimization.

        tol : float, default=1e-3
            Tolerance for stopping criteria. It operates on the maximum value of violation.

        lr : float, default=1
            The learning rate for multipliers' updates.
        """

        self.metric = metric
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def debias_weights(self, y: np.ndarray, s_one_hot: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
        """ Update instance's weight based on multipliers """
        exponents = np.zeros(y.shape[0], dtype=np.float32)
        exponents = exponents - np.sum(np.multiply(s_one_hot, multipliers[np.newaxis, :]), axis=1)
        weights = np.exp(exponents) / (np.exp(exponents) + np.exp(-exponents))
        weights = np.where(y > 0, 1 - weights, weights)

        return weights

    def dp_vio(self, y_pred: np.ndarray, y: np.ndarray, s_one_hot: np.ndarray) -> np.ndarray:
        """ Violation in demographic parity """

        vio = []
        base = np.mean(y_pred)
        for i in range(s_one_hot.shape[1]):
            idx = np.where(s_one_hot[:, i] == 1)
            vio.append(base - np.mean(np.take(y_pred, idx)))
        vio = np.asarray(vio)

        return vio

    def eop_vio(self, y_pred: np.ndarray, y: np.ndarray, s_one_hot: np.ndarray) -> np.ndarray:
        """ Violation in equal opportunity """

        vio = []
        pos_idx = np.where(y == 1)
        base = np.mean(np.take(y_pred, pos_idx))
        for i in range(s_one_hot.shape[1]):
            idx = np.intersect1d(np.where(s_one_hot[:, i] == 1), pos_idx)
            vio.append(base - np.mean(np.take(y_pred, idx)))
        vio = np.asarray(vio)

        return vio

    def fit(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[Sequence, np.ndarray],
            s: Union[Sequence, np.ndarray],
    ) -> LabelBias:
        """
        Parameters
        ----------
        """
        self._validate_params()
        if self.estimator is None:
            model = LogisticRegression()
        else:
            model = self.estimator

        X, y = self._validate_data(X, y)

        y_type = type_of_target(y)
        if y_type != "binary":
            raise ValueError("LabelBias only support binary target variables, but got {0} instead".format(y_type))
        self.classes_ = np.unique(y)

        s_ind = self._validate_grp_s(s, ("binary", "multiclass"))
        s_one_hot = onehottify(s_ind)

        if self.metric == "dp":
            self._vio_func = self.dp_vio
        elif self.metric == "eop":
            self._vio_func = self.eop_vio
        else:
            raise ValueError("LabelBias does not support {0} metric".format(self.metric))

        multipliers = np.zeros(len(self.s_classes_))
        self.weights_ = np.ones(X.shape[0])

        model.fit(X, y, self.weights_)
        y_pred = model.predict(X)
        y_pred_type = type_of_target(y_pred)
        if y_pred_type != "binary":
            raise ValueError("Expected binary type for predictions from estimator, got {0} instead".format(y_pred_type))

        vio = np.nan
        for _ in range(self.max_iter):
            self.weights_ = self.debias_weights(y, s_one_hot, multipliers)
            model.fit(X, y, self.weights_)
            y_pred = model.predict(X)
            vio = self._vio_func(y_pred, y, s_one_hot)
            if np.max(vio) <= self.tol:
                break
            else:
                multipliers += vio * self.lr

        if np.max(vio) > self.tol:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            warnings.warn("LabelBias does not converged, please consider increase the maximum iteration",
                          ConvergenceWarning)

        return self

    def predict(self, X: np.ndarray) -> Any:
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> Any:
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("Estimator does not have method 'predict_proba'")

        return self.estimator.predict_proba(X)

    def _more_tags(self):
        return {
            'binary_only': True,
            'requires_y': True,
        }
