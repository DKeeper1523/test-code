# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

""" https://link.springer.com/content/pdf/10.1007/s10115-011-0463-8.pdf """

import numpy as np
from collections import Counter
from typing import Sequence, Union


def reweigh(y: Union[Sequence, np.ndarray], s: Union[Sequence, np.ndarray], alg="reweighing"):
    assert alg in ("massaging", "reweighing", "sampling")

    y = np.asarray(y)
    s = np.asarray(s)

    if alg == "reweighing":
        w = _reweighing(y, s)
    else:
        raise NotImplementedError

    return w


def _reweighing(y: np.ndarray, s: np.ndarray) -> Sequence:
    # TODO: matrix computation

    N = y.shape[0]
    y = y.tolist()
    s = s.tolist()
    y_cnt = Counter(y)
    s_cnt = Counter(s)

    stat = {(e1, e2): 0. for e1 in y_cnt.keys() for e2 in s_cnt.keys()}
    for e1, e2 in zip(y, s):
        stat[(e1, e2)] += 1

    sample_weight = []
    for i in range(N):
        weight = (y_cnt[y[i]] * s_cnt[s[i]]) / (N * stat[(y[i], s[i])])
        sample_weight.append(weight)

    return sample_weight
