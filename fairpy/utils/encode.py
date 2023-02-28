# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import numpy as np
from typing import Sequence, Union


def onehottify(x: Union[np.ndarray, Sequence], n=None, dtype=np.float32):
    """ 1-hot encode x with the max value n (computed from data if n is None) """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    return np.eye(n, dtype=dtype)[x]
