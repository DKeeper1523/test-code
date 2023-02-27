# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Sequence, Union, Tuple

from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

from ..utils._param_validation import validate_parameter_constraints


class FairEstimator(BaseEstimator):
    """
    Abstract class for all algorithms in FairPy
    """

    def _validate_params(self) -> None:
        """
        Validate types and values of constructor parameters
        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """

        assert hasattr(self, "_parameter_constraints")
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

    def _validate_idx(self, idx: Union[int, npt.ArrayLike], max_idx: int) -> Tuple[int, ...]:
        """ Validate feature indexes """

        if isinstance(idx, int):
            idx = tuple([idx])

        if not all(isinstance(i, int) for i in idx):
            raise ValueError("expects all indexes an integer")
        elif min(idx) < 0 or max(idx) >= max_idx:
            raise ValueError("indexes out of range [0, Feat. Dim. - 1]")
        else:
            idx = tuple(i for i in sorted(set(idx)))

        return idx

    def _validate_grp_s(self, s: npt.ArrayLike, avail_s_type: Sequence[str]) -> npt.NDArray:
        """ Validate input sensitive features for group fairness """

        s_type = type_of_target(s)
        if s_type not in avail_s_type:
            raise ValueError("Does not support {0} sensitive attributes".format(s_type))

        enc = LabelEncoder()
        s_ind = enc.fit_transform(s)
        self.s_classes_ = enc.classes_
        if len(self.s_classes_) < 2:
            raise ValueError(
                "Need at least two sensitive groups, but the data only contains only one group: %r" %
                self.s_classes_[0])

        return s_ind
