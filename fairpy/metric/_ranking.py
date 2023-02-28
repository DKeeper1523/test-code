# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import math
import numpy as np
import numpy.typing as npt
from typing import Sequence, Union, List, Tuple

from sklearn.metrics._classification import column_or_1d, confusion_matrix
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length


def _fast_auc(y_true, y_pred):
    N = len(y_true)

    y_true = y_true[np.argsort(y_pred)]
    N_false = 0
    auc = 0

    for i in range(N):
        y_i = y_true[i]
        N_false += (1 - y_i)
        auc += y_i * N_false

    auc /= (N_false * (N - N_false))

    return auc


def _cross_auc(pred_1, pred_2):
    scores = np.hstack(np.asarray([pred_1, pred_2]))
    y_true = np.zeros(len(pred_1) + len(pred_2))
    y_true[0:len(pred_1)] = 1

    return _fast_auc(y_true, scores)


def xAUC(
        y_true: Union[Sequence, np.ndarray],
        y_pred: Union[Sequence, np.ndarray],
        s: Union[Sequence, np.ndarray],
        abs: bool = False,
) -> float:
    """
    The Fairness of Risk Scores Beyond Classification: Bipartite Ranking and the xAUC Metric

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
    xAUC : float
        Disparity in the probabilities of ranking a positive example from one group above a negative one
        from another group.
        The best performance is 0.

    Examples
    --------
    from fairpy.metric import xAUC

    y_true = [0, 0, 1, 0, 1, 1, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.8, 0.9, 1.5, 0, 1]
    s = ["male", "female", "male", "female", "male", "male", "male", "female"]

    xauc = xAUC(y_true, y_pred, s)
    """

    check_consistent_length(y_true, y_pred, s)
    y_true_type = type_of_target(y_true)
    y_pred_type = type_of_target(y_pred)
    s_type = type_of_target(s)

    if y_true_type != "binary":
        raise ValueError("xAUC does not support {0} targets".format(y_true_type))
    if y_pred_type not in ("binary", "continuous"):
        raise ValueError("xAUC does not support {0} predictions".format(y_pred_type))

    if s_type != "binary":
        raise ValueError("xAUC does not support {0} sensitive attribute".format(s_type))
    if len(np.unique(s)) == 1:
        raise ValueError("xAUC does not support only one sensitive group")

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    s = column_or_1d(s)

    s_val = np.unique(s)
    grp_0_idx = np.where(s == s_val[0])
    grp_1_idx = np.where(s == s_val[1])
    y_0_idx = np.where(y_true == 0)
    y_1_idx = np.where(y_true == 1)

    grp_0_y_0_cross_grp_1_y_1 = _cross_auc(
        y_pred[np.intersect1d(grp_0_idx, y_0_idx)],
        y_pred[np.intersect1d(grp_1_idx, y_1_idx)]
    )

    grp_1_y_0_cross_grp_0_y_1 = _cross_auc(
        y_pred[np.intersect1d(grp_1_idx, y_0_idx)],
        y_pred[np.intersect1d(grp_0_idx, y_1_idx)]
    )

    dp = grp_0_y_0_cross_grp_1_y_1 - grp_1_y_0_cross_grp_0_y_1
    if abs:
        dp = np.abs(dp)

    return dp


def dcg(ranking: List[int], rel_vec: npt.ArrayLike, k: int = 0) -> Tuple[float, float]:
    """ Discounted cumulative gain (dcg) and normalized discounted cumulative gain (ndcg) """

    dcgmax = 0.
    dcg = 0.
    sorted_rel = -np.sort(-rel_vec)
    N = len(rel_vec)

    if k == 0:
        k = N

    for i, rel in enumerate(sorted_rel[:min((k, N))]):
        dcgmax += float(2.0 ** rel - 1) / math.log2(2 + i)
    for i, doc in enumerate(ranking[:min(k, N)]):
        dcg += float(2 ** rel_vec[doc] - 1) / math.log2(2 + i)

    if dcgmax == 0:
        return 1.0, 1.0
    else:
        return dcg / dcgmax, dcg
