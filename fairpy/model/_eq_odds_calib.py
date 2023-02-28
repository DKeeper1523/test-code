# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Sequence, Union

import cvxpy as cvx

from ._base import FairEstimator


class EqOddsCalib(FairEstimator):
    """
    Equality of Opportunity in Supervised Learning

    Reference:
        https://arxiv.org/pdf/1610.02413.pdf
    Code adopted from:
        https://github.com/gpleiss/equalized_odds_and_calibration
    """

    def __init__(self):
        self.sp2p = None
        self.sn2p = None
        self.op2p = None
        self.on2p = None

    @staticmethod
    def base_rate(label):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(label)

    @staticmethod
    def tpr(pred, label):
        """
        True positive rate
        """
        return np.mean(np.logical_and(pred.round() == 1, label == 1))

    @staticmethod
    def fpr(pred, label):
        """
        False positive rate
        """
        return np.mean(np.logical_and(pred.round() == 1, label == 0))

    @staticmethod
    def tnr(pred, label):
        """
        True negative rate
        """
        return np.mean(np.logical_and(pred.round() == 0, label == 0))

    @staticmethod
    def fnr(pred, label):
        """
        False negative rate
        """
        return np.mean(np.logical_and(pred.round() == 0, label == 1))

    @staticmethod
    def fn_cost(pred, label):
        """
        Generalized false negative cost
        """
        return 1 - pred[label == 1].mean()

    @staticmethod
    def fp_cost(pred, label):
        """
        Generalized false positive cost
        """
        return pred[label == 0].mean()

    def fit(
            self,
            pred: Union[np.ndarray, pd.DataFrame, Sequence[Union[int, float]]],
            y: Union[np.ndarray, pd.DataFrame, Sequence[Union[int, float]]],
            s: Union[np.ndarray, pd.DataFrame, Sequence[Union[int, float, str]]],
    ) -> EqOddsCalib:
        # TODO: validate pred and y have the same size
        # TODO: validate pred in [0, 1]
        # TODO: check the implementation in AIF360

        s_ind = self._validate_grp_s(s, ("binary"))
        grp_0_idx, grp_1_idx = np.where(s_ind == 0)[0], np.where(s_ind == 1)[0]
        pred_grp_0, pred_grp_1 = np.take(pred, grp_0_idx), np.take(pred, grp_1_idx)
        y_grp_0, y_grp_1 = np.take(y, grp_0_idx), np.take(y, grp_1_idx)

        sbr = self.base_rate(y_grp_0)
        obr = self.base_rate(y_grp_1)

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr(pred_grp_0, y_grp_0) * sp2p + self.tnr(pred_grp_0, y_grp_0) * sn2p
        sfnr = self.fnr(pred_grp_0, y_grp_0) * sn2n + self.tpr(pred_grp_0, y_grp_0) * sp2n
        ofpr = self.fpr(pred_grp_1, y_grp_1) * op2p + self.tnr(pred_grp_1, y_grp_1) * on2p
        ofnr = self.fnr(pred_grp_1, y_grp_1) * on2n + self.tpr(pred_grp_1, y_grp_1) * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - pred_grp_0
        sconst = pred_grp_0
        oflip = 1 - pred_grp_1
        oconst = pred_grp_1

        sm_tn = np.logical_and(pred_grp_0 == 0, y_grp_0 == 0)
        sm_fn = np.logical_and(pred_grp_0 == 0, y_grp_0 == 1)
        sm_tp = np.logical_and(pred_grp_0 == 1, y_grp_0 == 1)
        sm_fp = np.logical_and(pred_grp_0 == 1, y_grp_0 == 0)

        om_tn = np.logical_and(pred_grp_1 == 0, y_grp_1 == 0)
        om_fn = np.logical_and(pred_grp_1 == 0, y_grp_1 == 1)
        om_tp = np.logical_and(pred_grp_1 == 1, y_grp_1 == 1)
        om_fp = np.logical_and(pred_grp_1 == 1, y_grp_1 == 0)

        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        self.sp2p = sp2p.value
        self.sn2p = sn2p.value
        self.op2p = op2p.value
        self.on2p = on2p.value

        return self

    def transform(
            self,
            pred: Union[np.ndarray, pd.DataFrame, Sequence[Union[int, float]]],
            s: Union[np.ndarray, pd.DataFrame, Sequence[Union[int, float, str]]],
    ) -> np.ndarray:
        # TODO: align train and test s
        # TODO: check fitted

        s_ind = self._validate_grp_s(s, ("binary"))
        grp_0_idx, grp_1_idx = np.where(s_ind == 0)[0], np.where(s_ind == 1)[0]

        fair_pred = copy.deepcopy(pred)

        pp_indices = np.intersect1d(np.where(pred == 1)[0], grp_0_idx)
        pn_indices = np.intersect1d(np.where(pred == 0)[0], grp_0_idx)
        np.random.shuffle(pp_indices)
        np.random.shuffle(pn_indices)

        n2p_indices = pn_indices[:int(len(pn_indices) * self.sn2p)]
        fair_pred[n2p_indices] = 1 - fair_pred[n2p_indices]
        p2n_indices = pp_indices[:int(len(pp_indices) * (1 - self.sp2p))]
        fair_pred[p2n_indices] = 1 - fair_pred[p2n_indices]

        othr_pp_indices = np.intersect1d(np.where(pred == 1)[0], grp_1_idx)
        othr_pn_indices = np.intersect1d(np.where(pred == 0)[0], grp_1_idx)
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * self.on2p)]
        fair_pred[n2p_indices] = 1 - fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - self.op2p))]
        fair_pred[p2n_indices] = 1 - fair_pred[p2n_indices]

        return fair_pred
