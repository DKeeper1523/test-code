# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

from __future__ import annotations

import math
import random
import numpy as np
import numpy.typing as npt
import pandas as pd
from numbers import Real, Integral
from typing import Sequence, Union, Any, Optional, Dict, NewType, Tuple, Callable, Mapping, List

from sklearn.utils.multiclass import type_of_target

import torch
from torch import nn

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, HasMethods, Interval
from ..metric import dcg


class FairPGRank(FairEstimator):
    """
    Policy Learning for Fairness in Ranking

    Reference:
        https://papers.nips.cc/paper/2019/file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf

    Code adopted from:
        https://github.com/ashudeep/Fair-PGRank

    Examples
    --------
    """

    _parameter_constraints = {
        "num_epochs": [Interval(Integral, 0, None, closed="left")],
        "lr": [Interval(Real, 0, None, closed="left")],
        "weight_decay": [Interval(Real, 0, None, closed="left")],
        "sample_size": [Interval(Integral, 0, None, closed="left")],
        "reward_type": [StrOptions({"ndcg", "dcg", "avrank"})],
        "baseline_type": [StrOptions({"value", "max"})],
        "lambda_grp_fair": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
            self,
            model: Optional[nn.Module] = None,
            num_epochs: int = 100,
            lr: float = 1e-3,
            weight_decay: float = 1e-3,
            sample_size: int = 10,
            reward_type: str = "ndcg",
            baseline_type: str = "value",
            lambda_grp_fair: float = 1.,
            device: str = "cpu",
    ):
        """
        Parameters
        ----------
        """

        self.model = model
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.sample_size = sample_size
        self.reward_type = reward_type
        self.baseline_type = baseline_type
        self.lambda_grp_fair = lambda_grp_fair
        self.device = device

    def _sample_ranking(self, probs: npt.NDArray[np.float32]) -> List[int]:
        ranking = np.random.choice(probs.shape[0], size=probs.shape[0], p=probs, replace=False)
        return ranking

    def _log_model_prob(self, scores, ranking):
        pass

    def fit(self, X, y, s=None) -> FairPGRank:
        # y is a matrix not a vector in ranking
        X, y = torch.from_numpy(X).to(self.device), torch.from_numpy(y).to(self.device)

        if self.lambda_grp_fair > 0:
            if s is None:
                raise ValueError("Sensitive attributes 's' are required for group fairness")
            else:
                # TODO: verify for query and s
                pass

        # TODO: lr scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.num_epochs):
            scores = self.model(X)
            probs = torch.nn.functional.softmax(scores, dim=0)

            ranking_list, reward_list = [], []
            for _ in range(self.sample_size):
                ranking = self._sample_ranking(probs)
                ranking_list.append(ranking)

                # utility
                if self.reward_type == "ndcg":
                    reward_list.append(dcg(ranking, y)[0])
                elif self.reward_type == "dcg":
                    reward_list.append(dcg(ranking, y)[1])

                # baseline
                if self.baseline_type == "value":
                    baseline = np.mean(reward_list)
                elif self.baseline_type == "max":
                    # TODO: figure out
                    pass

                # group fairness
                if self.lambda_grp_fair > 0:
                    if np.sum(rel_labels[group_identities == 0]) == 0 or np.sum(
                            rel_labels[group_identities == 1]) == 0:
                        skip_this_query = True

                    pass

            optimizer.zero_grad()
            for i in range(self.sample_size):
                ranking = ranking_list[i]
                reward = reward_list[i]

        return self
