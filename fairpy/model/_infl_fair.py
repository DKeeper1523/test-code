# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

import sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.linalg import cho_solve, cho_factor
from abc import ABC, abstractmethod
from typing import Sequence, Union, Tuple, Callable

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, Interval, Real


class IFBase(ABC):
    """ Abstract base class for influence function computation """

    @staticmethod
    def set_sample_weight(N: int, sample_weight: Union[np.ndarray, Sequence[float]] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(N)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == N
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == N
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0.
            assert max(sample_weight) <= 2.

        return sample_weight

    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True) -> np.ndarray:
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)


class LogisticRegression(IFBase):
    """ Logistic regression for binary classification """

    def __init__(self, l2_reg: float, fit_intercept: bool = False):
        super(LogisticRegression, self).__init__()

        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.weight = None
        self.model = sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=(1. / l2_reg),
            fit_intercept=fit_intercept,
            tol=1e-8,
            solver="lbfgs",
            max_iter=2048,
            multi_class="ovr",
            warm_start=False,
        )

    def _log_loss(self, X: np.ndarray, y: np.ndarray, sample_weight=None, l2_reg=False, eps=1e-16):
        """ Log loss for logistic regression """

        sample_weight = self.set_sample_weight(X.shape[0], sample_weight)

        pred = self.predict_proba(X)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            log_loss += self.l2_reg * np.linalg.norm(self.weight, ord=2) / 2.

        return log_loss

    def _grad(self, X: np.ndarray, y: np.ndarray, sample_weight=None, l2_reg=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients: grad_wo_reg = (pred - y) * x
        """

        sample_weight = np.array(self.set_sample_weight(X.shape[0], sample_weight))

        pred = self.predict_proba(X)

        indiv_grad = X * (pred - y).reshape(-1, 1)
        reg_grad = self.l2_reg * self.weight
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        if self.fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)
            reg_grad = np.concatenate([reg_grad, np.zeros(1)], axis=0)

        total_grad = np.sum(weighted_indiv_grad, axis=0)

        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def _grad_pred(self, X: np.ndarray, sample_weight=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(X.shape[0], sample_weight))

        pred = self.predict_proba(X)
        indiv_grad = X * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

        return total_grad, weighted_indiv_grad

    def _hess(self, X: np.ndarray, sample_weight=None, check_pos_def=False) -> np.ndarray:
        """
        Compute hessian matrix: hessian = pred * (1 - pred) @ x^T @ x + lambda
        """

        sample_weight = np.array(self.set_sample_weight(X.shape[0], sample_weight))

        pred = self.predict_proba(X)

        factor = pred * (1. - pred)
        indiv_hess = np.einsum("a,ai,aj->aij", factor, X, X)
        reg_hess = self.l2_reg * np.eye(X.shape[1])

        if self.fit_intercept:
            off_diag = np.einsum("a,ai->ai", factor, X)
            off_diag = off_diag[:, np.newaxis, :]

            top_row = np.concatenate([indiv_hess, np.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = np.concatenate([off_diag, factor.reshape(-1, 1, 1)], axis=2)
            indiv_hess = np.concatenate([top_row, bottom_row], axis=1)

            reg_hess = np.pad(reg_hess, [[0, 1], [0, 1]], constant_values=0.)

        hess_wo_reg = np.einsum("aij,a->ij", indiv_hess, sample_weight)
        total_hess_w_reg = hess_wo_reg + reg_hess

        if check_pos_def:
            self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, verbose=False) -> LogisticRegression:
        sample_weight = self.set_sample_weight(X.shape[0], sample_weight)

        self.model.fit(X, y, sample_weight=sample_weight)
        self.weight: np.ndarray = self.model.coef_.flatten()
        if self.fit_intercept:
            self.bias: np.ndarray = self.model.intercept_

        if verbose:
            train_loss_wo_reg = self._log_loss(X, y, sample_weight)
            reg_loss = np.sum(np.power(self.weight, 2)) * self.l2_reg / 2.
            train_loss_w_reg = train_loss_wo_reg + reg_loss

            print("Train loss: %.5f + %.5f = %.5f" % (train_loss_wo_reg, reg_loss, train_loss_w_reg))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class InflFair(FairEstimator):
    """
    Achieving Fairness at No Utility Cost via Data Reweighing with Influence

    Reference:
        https://proceedings.mlr.press/v162/li22p/li22p.pdf
    Code adopted from:
        https://github.com/brandeis-machine-learning/influence-fairness

    Examples
    --------
    >>> from fairpy.dataset import Adult
    >>> from fairpy.model import InflFair
    >>> dataset = Adult()
    >>> split_data = dataset.split()
    >>> model = InflFair()
    >>> model.fit(split_data.X_train, split_data.y_train, split_data.s_train)
    >>> model.predict(split_data.X_test)
    """

    _parameter_constraints = {
        "metric": [StrOptions({"dp", "eop"})],
        "l2_reg": [Interval(Real, 0, None, closed="left")],
        "max_fit": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
            self,
            metric: str = "eop",
            l2_reg: float = 1.,
            max_fit: int = 5,
    ):
        """
        Parameters
        ----------
        metric : {"dp", "eop"}, default="dp"
            The fairness notion adopted in optimization.
            dp: demographic parity
            eop: equal opportunity

        l2_reg: float, default=1.
            L2 regularization strength for logistic regression
        """

        self.metric = metric
        self.l2_reg = l2_reg
        self.max_fit = max_fit

    def _grad_ferm(
            self,
            grad_fn: Callable,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.int16],
            s: npt.NDArray[np.int16],
    ) -> npt.NDArray[np.float64]:
        """
        Gradients of fair empirical risk minimization for binary sensitive attribute
        Exp(L|grp_0) - Exp(L|grp_1)
        """

        N = X.shape[0]

        idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
        idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

        grad_grp_0_y_1, _ = grad_fn(X=X[idx_grp_0_y_1], y=y[idx_grp_0_y_1])
        grad_grp_1_y_1, _ = grad_fn(X=X[idx_grp_1_y_1], y=y[idx_grp_1_y_1])

        return (grad_grp_0_y_1 / len(idx_grp_0_y_1)) - (grad_grp_1_y_1 / len(idx_grp_1_y_1))

    def _loss_ferm(
            self,
            loss_fn: Callable,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.int16],
            s: npt.NDArray[np.int16],
    ) -> float:
        """
        Loss of fair empirical risk minimization for binary sensitive attribute
        Exp(L|grp_0) - Exp(L|grp_1)
        """

        N = X.shape[0]

        idx_grp_0_y_1 = [i for i in range(N) if s[i] == 0 and y[i] == 1]
        idx_grp_1_y_1 = [i for i in range(N) if s[i] == 1 and y[i] == 1]

        loss_grp_0_y_1 = loss_fn(X[idx_grp_0_y_1], y[idx_grp_0_y_1])
        loss_grp_1_y_1 = loss_fn(X[idx_grp_1_y_1], y[idx_grp_1_y_1])

        return (loss_grp_0_y_1 / len(idx_grp_0_y_1)) - (loss_grp_1_y_1 / len(idx_grp_1_y_1))

    def _lp(self, fair_infl: Sequence[float], util_infl: Sequence[float], fair_loss: float) -> npt.NDArray[np.float64]:
        """ Linear programming """

        num_sample = len(fair_infl)
        max_fair = sum([v for v in fair_infl if v < 0.])
        max_util = sum([v for v in util_infl if v < 0.])

        print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f" % (max_fair, max_util))

        all_one = np.array([1. for _ in range(num_sample)])
        fair_infl = np.array(fair_infl)
        util_infl = np.array(util_infl)
        model = gp.Model()
        x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

        if fair_loss >= -max_fair:
            print("=====> Fairness loss exceeds the maximum availability")
            model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
            model.addConstr(all_one @ x <= 0.10 * num_sample, name="amount")
            model.setObjective(fair_infl @ x)
            model.optimize()
        else:
            model.addConstr(fair_infl @ x <= 0. * -fair_loss, name="fair")
            model.addConstr(util_infl @ x <= 0. * max_util, name="util")
            model.setObjective(all_one @ x)
            model.optimize()

        print("Total removal: %.5f; Ratio: %.3f%%" % (sum(x.X), (sum(x.X) / num_sample) * 100))

        return 1 - x.X

    def fit(
            self,
            X_train: npt.ArrayLike,
            y_train: npt.ArrayLike,
            X_val: npt.ArrayLike,
            y_val: npt.ArrayLike,
            s_val: npt.ArrayLike,
    ) -> InflFair:
        """
        Parameters
        ----------

        #TODO: add repeatedly reweighing
        """

        self.model = LogisticRegression(self.l2_reg)
        self.model.fit(X_train, y_train)

        ori_fair_loss_val = self._loss_ferm(self.model._log_loss, X_val, y_val, s_val)
        ori_util_loss_val = self.model._log_loss(X_val, y_val)

        train_total_grad, train_indiv_grad = self.model._grad(X_train, y_train)
        util_loss_total_grad, acc_loss_indiv_grad = self.model._grad(X_val, y_val)
        fair_loss_total_grad = self._grad_ferm(self.model._grad, X_val, y_val, s_val)

        hess = self.model._hess(X_train)
        util_grad_hvp = self.model.get_inv_hvp(hess, util_loss_total_grad)
        fair_grad_hvp = self.model.get_inv_hvp(hess, fair_loss_total_grad)

        util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
        fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)

        sample_weight = self._lp(fair_pred_infl, util_pred_infl, ori_fair_loss_val)

        self.model.fit(X_train, y_train, sample_weight=sample_weight)

        return self
