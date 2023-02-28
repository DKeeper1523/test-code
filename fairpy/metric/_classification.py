# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import numpy as np
from typing import Sequence, Union

from sklearn.metrics._classification import column_or_1d, confusion_matrix
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length


def binary_dp(
        y_pred: Union[Sequence, np.ndarray],
        s: Union[Sequence, np.ndarray],
        abs: bool = True,
) -> float:
    """
    Demographic parity for binary sensitive attribute and binary classification

    Parameters
    ----------
    y_pred : 1d array-like
        Predicted labels, as returned by a classifier.

    s : 1d array-like
        Sensitive attributes.

    abs : bool, default=True
        If ``True``, return the absolute value of dp.
        Otherwise, treat sensitive attribute with value 0 as the privileged group

    Returns
    -------
    dp : float
        Return the gap in demographic parity between two sensitive groups.
        The best performance is 0.

    Examples
    --------

    """

    check_consistent_length(y_pred, s)
    y_type = type_of_target(y_pred)
    s_type = type_of_target(s)

    if y_type != "binary":
        raise ValueError("binary_dp does not support {0} predictions".format(y_type))
    if s_type != "binary":
        raise ValueError("binary_dp does not support {0} sensitive attribute".format(y_type))
    if len(np.unique(s)) == 1:
        raise ValueError("Demographic Parity does not support only one sensitive group")

    y_pred = column_or_1d(y_pred)
    s = column_or_1d(s)

    s_val = np.unique(s)
    grp_0_idx = np.where(s == s_val[0])
    grp_1_idx = np.where(s == s_val[1])

    dp = np.mean(y_pred.take(grp_0_idx)) - np.mean(y_pred.take(grp_1_idx))
    if abs:
        dp = np.abs(dp)

    return dp


def binary_eop(
        y_true: Union[Sequence, np.ndarray],
        y_pred: Union[Sequence, np.ndarray],
        s: Union[Sequence, np.ndarray],
        abs: bool = True,
) -> float:
    """
    Equal Opportunity for binary sensitive attribute and binary classification

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) labels.

    y_pred : 1d array-like
        Predicted labels, as returned by a classifier.

    s : 1d array-like
        Sensitive attributes.

    abs : bool, default=True
        If ``True``, return the absolute value of dp.
        Otherwise, treat sensitive attribute with value 0 as the privileged group

    Returns
    -------
    dp : float
        Return the gap in demographic parity between two sensitive groups.
        The best performance is 0.

    Examples
    --------

    """

    check_consistent_length(y_true, y_pred, s)
    y_true_type = type_of_target(y_true)
    y_pred_type = type_of_target(y_pred)
    s_type = type_of_target(s)

    if y_true_type != "binary":
        raise ValueError("binary_eop does not support {0} targets".format(y_true_type))
    if y_pred_type not in ("binary", "multiclass"):
        raise ValueError("binary_eop does not support {0} predictions".format(y_pred_type))
    elif y_pred_type == "multiclass":
        raise Warning("Expected binary classification, but predictions contain more than two classes")
    if s_type != "binary":
        raise ValueError("binary_eop does not support {0} sensitive attribute".format(s_type))
    if len(np.unique(s)) == 1:
        raise ValueError("Equal Opportunity does not support only one sensitive group")

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    s = column_or_1d(s)

    s_val = np.unique(s)
    grp_0_idx = np.where(s == s_val[0])
    grp_1_idx = np.where(s == s_val[1])

    cm_grp_0 = confusion_matrix(y_true[grp_0_idx].squeeze(), y_pred[grp_0_idx].squeeze())
    cm_grp_1 = confusion_matrix(y_true[grp_1_idx].squeeze(), y_pred[grp_1_idx].squeeze())

    tpr_grp_0 = cm_grp_0[1][1] / (cm_grp_0[1][1] + cm_grp_0[1][0])
    tpr_grp_1 = cm_grp_1[1][1] / (cm_grp_1[1][1] + cm_grp_1[1][0])
    eop = tpr_grp_0 - tpr_grp_1

    if abs:
        eop = np.abs(eop)

    return eop
