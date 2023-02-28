# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import math
import copy
import random
import numpy as np
import numpy.typing as npt
import pandas as pd
from copy import deepcopy
from numbers import Real
import networkx as nx
from collections import OrderedDict, Counter
from typing import Sequence, Union, Any, Optional, Dict, NewType, Tuple, Callable, Mapping, List

from sklearn.utils.multiclass import type_of_target

from ._base import FairEstimator
from ..utils._param_validation import StrOptions, HasMethods, Interval

SenFeat = NewType('SenFeat', Union[int, float, str, Sequence])  # sensitive feature
ColFeat = NewType('ColFeat', Union[int, float, str])  # feature at current column


def get_median(input: Sequence) -> Any:
    sorted_input = sorted(deepcopy(input))

    if len(sorted_input) % 2 == 0:
        return sorted_input[len(sorted_input) // 2 - 1]
    else:
        return sorted_input[len(sorted_input) // 2]


class CategoricalFeat():
    """ Modeling a categorical feature """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.bin_data = dict(Counter(self.data))
        self.val2idx = {val: np.where(self.data == val)[0].tolist() for val in self.bin_data.keys()}
        self.N_bin = len(self.bin_data)

    def create_graph(self, generator: Callable) -> nx.classes.digraph.DiGraph:
        dg = nx.DiGraph()
        dg.add_node("s")
        dg.add_node("t")

        for i, (col_val, cnt) in enumerate(self.bin_data.items()):
            dg.add_node(i)
            dg.add_node(i + self.N_bin)
            dg.add_edge("s", i, capacity=cnt, weight=0)
            dg.add_edge(i + self.N_bin, "t", capacity=generator(col_val), weight=0)

        dg.add_node(2 * self.N_bin)
        dg.add_edge(2 * self.N_bin, "t", capacity=len(self), weight=0)

        for i in range(self.N_bin):
            for j in range(self.N_bin, 2 * self.N_bin):
                if i + self.N_bin == j:
                    dg.add_edge(i, j, weight=0)
                else:
                    dg.add_edge(i, j, weight=1)

            dg.add_edge(i, 2 * self.N_bin, weight=2)

        self.dg = dg

        return dg

    def repair(self) -> Tuple[np.ndarray, int]:
        min_cost_flow = nx.max_flow_min_cost(self.dg, "s", "t")

        val2idx_ = copy.deepcopy(self.val2idx)
        repair_data = np.asarray(["PLACEHOLDER_DIREMOVER_FAIRPY" for _ in range(len(self))])
        repair_bin_dict = {}
        candidates = list(self.bin_data.keys())
        overflow = 0

        for i, (col_val, cnt) in enumerate(self.bin_data.items()):
            overflow += min_cost_flow[i][2 * self.N_bin]

            for j in range(self.N_bin, 2 * self.N_bin):
                edgeflow = min_cost_flow[i][j]
                group = random.sample(val2idx_[col_val], int(edgeflow))

                q = j - self.N_bin

                for e in group:
                    val2idx_[col_val].remove(e)
                    repair_data[e] = candidates[q]

                if q in repair_bin_dict:
                    repair_bin_dict[q].extend(group)
                else:
                    repair_bin_dict[q] = group

        return repair_data, overflow

    def __len__(self):
        return len(self.data)


class DIRemover(FairEstimator):
    """
    Certifying and Removing Disparate Impact

    Reference:
        https://dl.acm.org/doi/pdf/10.1145/2783258.2783311
    Code adopted from:
        https://github.com/algofairness/BlackBoxAuditing

    # TODO: bugs on data types in numpy

    Examples
    --------
    >>> from fairpy.dataset import Adult
    >>> from fairpy.model import DIRemover
    >>> dataset = Adult()
    >>> split_data = dataset.split()
    >>> model = DIRemover(s_idx=dataset.feat_idx.sen_idx, num_feat_idx=0)
    >>> new_X_train = model.fit_transform(split_data.X_train)
    """

    _parameter_constraints = {
        "s_idx": [None, Interval(Real, 0, None, closed="left"), "array-like"],
        "repair_feat_idx": [None, Interval(Real, 0, None, closed="left"), "array-like"],
        "num_feat_idx": [None, Interval(Real, 0, None, closed="left"), "array-like"],
        "repair_level": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
            self,
            s_idx: Union[int, Sequence[int]],
            repair_feat_idx: Optional[Union[int, Sequence[int]]] = None,
            num_feat_idx: Optional[Union[int, Sequence[int]]] = None,
            repair_level: float = 1.,
    ):
        """
        Parameters
        ----------
        """

        self.s_idx = s_idx
        self.repair_feat_idx = repair_feat_idx
        self.num_feat_idx = num_feat_idx
        self.repair_level = repair_level

    def _validate_idx(self, X):
        """ Validate indexes in s_idx, repair_feat_idx, and num_feat_idx """

        if isinstance(self.s_idx, int):
            self.s_idx = tuple([self.s_idx])
        if isinstance(self.repair_feat_idx, int):
            self.repair_feat_idx = tuple([self.repair_feat_idx])
        if isinstance(self.num_feat_idx, int):
            self.num_feat_idx = tuple([self.num_feat_idx])

        if not all(isinstance(idx, int) for idx in self.s_idx):
            raise ValueError("s_idx in DIRemover expects all indexes an integer")
        elif min(self.s_idx) < 0 or max(self.s_idx) >= X.shape[1]:
            raise ValueError("s_idx in DIRemover out of range [0, Feat. Dim. - 1]")
        else:
            self.s_idx = tuple(i for i in sorted(set(self.s_idx)))

        if self.repair_feat_idx is None:
            self.repair_feat_idx = (i for i in range(X.shape[1]))
        else:
            if not all(isinstance(idx, int) for idx in self.repair_feat_idx):
                raise ValueError("repair_feat_idx in DIRemover expects all indexes an integer")
            elif min(self.num_feat_idx) < 0 or max(self.num_feat_idx) >= X.shape[1]:
                raise ValueError("repair_feat_idx in DIRemover out of range [0, Feat. Dim. - 1]")
            else:
                self.repair_feat_idx = tuple(i for i in sorted(set(self.repair_feat_idx)))

        if self.num_feat_idx is not None:
            if not all(isinstance(idx, int) for idx in self.num_feat_idx):
                raise ValueError("num_feat_idx in DIRemover expects all indexes an integer")
            elif min(self.num_feat_idx) < 0 or max(self.num_feat_idx) >= X.shape[1]:
                raise ValueError("num_feat_idx in DIRemover out of range [0, Feat. Dim. - 1]")
            else:
                self.num_feat_idx = tuple(i for i in sorted(set(self.num_feat_idx)))

    def _repair_num_feat(
            self,
            x: npt.NDArray,
            unique_s: Sequence[SenFeat],
            s2val2idx: Mapping[SenFeat, Mapping[ColFeat, Sequence[int]]],
    ) -> np.ndarray:
        """ Repair a numerical feature x """

        # minimal number of unique column values for all sensitive attributes
        N_quantiles = min(len(set_) for set_ in s2val2idx.values())
        quantile_unit = 1. / N_quantiles

        s2offsets = {s: 0 for s in unique_s}

        ranked_col_val = sorted(np.unique(x))
        for quantile in range(N_quantiles):
            median_at_curr_quantile = []
            s2idx_curr_quantile = {}

            for s in unique_s:
                N_col_vals = len(s2val2idx[s])
                offset = int(round(s2offsets[s] * N_col_vals))
                number_to_get = int(round((s2offsets[s] + quantile_unit) * N_col_vals) - offset)
                s2offsets[s] += quantile_unit

                if number_to_get > 0:
                    # select column values at current quantile
                    select_col_val = list(s2val2idx[s].keys())[offset:offset + number_to_get]
                    s2idx_curr_quantile[s] = [idx for val in select_col_val for idx in s2val2idx[s][val]]
                    median_at_curr_quantile.append(get_median(select_col_val))

            median = get_median(median_at_curr_quantile)
            median_rank = ranked_col_val.index(median)

            for s in unique_s:
                for row_idx in s2idx_curr_quantile[s]:
                    # repair column values at current quantile
                    curr_val = x[row_idx]
                    curr_val_rank = ranked_col_val.index(curr_val)
                    distance = median_rank - curr_val_rank
                    distance_to_repair = int(round(distance * self.repair_level))
                    repaired_val = ranked_col_val[curr_val_rank + distance_to_repair]
                    x[row_idx] = repaired_val

        return x

    def _gen_desired_dist(
            self,
            s: SenFeat,
            col_val: ColFeat,
            cnt_norm: Mapping[ColFeat, Mapping[SenFeat, float]],
            median: Mapping[ColFeat, float],
            mode: ColFeat,
            is_sen_feat: bool,
    ) -> float:
        if is_sen_feat:
            return 1 if col_val == mode else (1 - self.repair_level) * cnt_norm[col_val][s]
        else:
            return (1 - self.repair_level) * cnt_norm[col_val].get(s, 0) + (self.repair_level * median[col_val])

    def _gen_desired_cnt(
            self,
            s: SenFeat,
            col_val: ColFeat,
            s2cat_feat: Mapping[SenFeat, CategoricalFeat],
            cnt_dict: Mapping[ColFeat, Mapping[SenFeat, int]],
            median: Mapping[ColFeat, float],
    ) -> int:
        N = len(s2cat_feat[s])
        cnt = cnt_dict[col_val].get(s, 0)
        des_cnt = math.floor(((1 - self.repair_level) * cnt) + (self.repair_level) * median[col_val] * N)

        return des_cnt

    def _flow(
            self,
            unique_s: Tuple,
            s2cat_feat: Mapping[SenFeat, CategoricalFeat],
            repair_generator: Callable,
    ) -> Mapping[SenFeat, Tuple[npt.NDArray, int]]:

        s2flow = {}
        for i, s in enumerate(unique_s):
            cat_feat = s2cat_feat[s]
            generator = lambda col_val: repair_generator(s, col_val)
            cat_feat.create_graph(generator)
            new_feat, overflow = cat_feat.repair()
            s2flow[s] = (new_feat, overflow)

        return s2flow

    def _assign_overflow(
            self,
            unique_s: Sequence[SenFeat],
            unique_col_val: Sequence[ColFeat],
            s2flow: Mapping[SenFeat, Tuple[np.ndarray, int]],
            repair_generator: Callable,
    ) -> Mapping[SenFeat, npt.NDArray]:

        assigned_overflow = {}
        s2repaired = {s: flow[0] for s, flow in s2flow.items()}

        print(s2flow)

        for s in unique_s:
            dist_generator = lambda col_val: repair_generator(s, col_val)
            col_val_props = list(map(dist_generator, unique_col_val))

            if all(e == 0 for e in col_val_props):
                col_val_props = [1. / len(col_val_props)] * len(col_val_props)

            col_val_props = [e / float(sum(col_val_props)) for e in col_val_props]

            assigned_overflow[s] = {}
            for i in range(int(s2flow[s][1])):
                N = random.uniform(0, 1)
                cat_index = 0
                tally = 0

                for j in range(len(col_val_props)):
                    val = col_val_props[j]
                    if N < (tally + val):
                        cat_index = j
                        break
                    tally += val

                assigned_overflow[s][i] = unique_col_val[cat_index]

            cnt = 0
            for i, val in enumerate(s2flow[s][0]):
                if val == "PLACEHOLDER_DIREMOVER_FAIRPY":
                    s2repaired[s][i] = assigned_overflow[s][cnt]
                    cnt += 1

        return s2repaired

    def _repair_cat_feat(
            self,
            x: np.ndarray,
            unique_s: Tuple,
            s2row: Mapping[SenFeat, Sequence[int]],
            s2val2idx: Mapping[SenFeat, Mapping[ColFeat, List[int]]],
    ):
        """ Repair a categorical feature """

        unique_col_val = np.unique(x)
        mode = max(s2row, key=lambda k: len(s2row[k]))

        # Construct CatFeat for each sensitive attribute
        s2cat_feat = {}
        for s, row_idx in s2row.items():
            val = x[row_idx]
            s2cat_feat[s] = CategoricalFeat(val)

        # Calculate and normalize the joint distribution over column value and sensitive attribute
        cnt_dict: Dict[ColFeat, OrderedDict[SenFeat, int]] = {}
        cnt_norm_dict: Dict[ColFeat, OrderedDict[SenFeat, float]] = {}
        for col_val in unique_col_val:
            temp_cnt = {}
            temp_cnt_norm = {}
            for s in unique_s:
                if col_val in s2val2idx[s]:
                    temp_cnt[s] = len(s2val2idx[s][col_val])
                    temp_cnt_norm[s] = len(s2val2idx[s][col_val]) / len(s2row[s])

            cnt_dict[col_val] = OrderedDict(data=sorted(temp_cnt.items(), key=lambda x: x[1]))
            cnt_norm_dict[col_val] = OrderedDict(sorted(temp_cnt_norm.items(), key=lambda x: x[1]))

        # TODO: computational error in median
        def list_pad(col_val):
            temp = list(cnt_norm_dict[col_val].values())
            if len(temp) == 1:
                temp.insert(0, 0)
            return temp

        median = {col_val: get_median(list_pad(col_val)) for col_val in unique_col_val}
        median = {'A': 0.3333333333333333, 'B': 0.30000000000000004, 'C': 0}

        print("median", median)

        dist_generator = lambda s, col_val: self._gen_desired_dist(s, col_val, cnt_norm_dict, median, mode, False)
        cnt_generator = lambda s, col_val: self._gen_desired_cnt(s, col_val, s2cat_feat, cnt_dict, median)

        s2flow = self._flow(
            unique_s=unique_s,
            s2cat_feat=s2cat_feat,
            repair_generator=cnt_generator,
        )

        s2repaired = self._assign_overflow(
            unique_s=unique_s,
            unique_col_val=unique_col_val,
            s2flow=s2flow,
            repair_generator=dist_generator,
        )

        for s in unique_s:
            for i, idx in enumerate(s2row[s]):
                x[idx] = s2repaired[s][i]

        print(x)

        return x

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self._validate_params()
        X = self._validate_data(X, dtype=None)
        self._validate_idx(X)
        X = np.copy(X)

        s_cols = X[:, self.s_idx].astype('U')
        unique_s = np.unique(s_cols, axis=0)
        unique_s = tuple(tuple(s) for s in unique_s)

        # sensitive attribute -> indexes of rows containing the sensitive attribute
        s2row: Dict[SenFeat, Sequence[int]] = {
            s: np.where(np.prod(X[:, self.s_idx] == s, axis=1) == 1)[0] for s in unique_s
        }

        # repair data on a column basis
        for col_idx in self.repair_feat_idx:

            # sensitive attribute -> unique values in the current column -> corresponding row indexes
            s2val2idx = {}
            for s in unique_s:
                unique_val = np.unique(X[s2row[s], col_idx])
                curr_dict = {
                    val: np.intersect1d(np.where(X[:, self.s_idx] == s)[0], np.where(X[:, col_idx] == val)[0]).tolist()
                    for val in unique_val
                }
                s2val2idx[s] = curr_dict

            # sort column values in a ascending order for numerical features
            s2val2idx = {s: OrderedDict(sorted(dict_.items())) for s, dict_ in s2val2idx.items()}

            if self.num_feat_idx is not None and col_idx in self.num_feat_idx:
                X[:, col_idx] = self._repair_num_feat(X[:, col_idx], unique_s, s2val2idx)
            else:
                X[:, col_idx] = self._repair_cat_feat(X[:, col_idx], unique_s, s2row, s2val2idx)

            break

        return X
