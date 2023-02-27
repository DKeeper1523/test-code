# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

import abc
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import binom
from numbers import Real, Integral
from typing import List

from sklearn.utils._param_validation import Interval

from ._base import FairEstimator


def compute_aux_mtable(mtable: pd.DataFrame) -> pd.DataFrame:
    """
    Stores the inverse of an mTable entry and the size of the block with respect to the inverse
    """

    if not (isinstance(mtable, pd.DataFrame)):
        raise TypeError("Internal mtable must be a DataFrame")

    aux_mtable = pd.DataFrame(columns=["inv", "block"])
    last_m_seen = 0
    last_position = 0
    for position in range(1, len(mtable)):
        if position % 2000 == 0:
            print("Computing m inverse: {:.0f} of {:.0f}".format(position, len(mtable)))
        if mtable.at[position, "m"] == last_m_seen + 1:
            last_m_seen += 1
            aux_mtable.loc[position] = [position, position - last_position]
            last_position = position
        elif mtable.at[position, "m"] != last_m_seen:
            raise RuntimeError("Inconsistent mtable")

    return aux_mtable


class MTableFailProbPair():
    """
    Encapsulation of all parameters for the interim mtables
    """

    def __init__(self, k, p, alpha, fail_prob, mtable):
        self.k = k
        self.p = p
        self.alpha = alpha
        self.fail_prob = fail_prob
        self.mtable = mtable

    def mass_of_mtable(self):
        return self.mtable['m'].sum()


class LegalAssignmentKey():
    """
    Utility class for the recursive fail prob
    """

    def __init__(self, remaining_candidates, remaining_block_sizes, current_block_number, candidates_assigned_so_far):
        self.remaining_candidates = remaining_candidates
        self.remaining_block_sizes = remaining_block_sizes
        self.current_block_number = current_block_number
        self.candidates_assigned_so_far = candidates_assigned_so_far

    def __eq__(self, other):
        if self.remaining_candidates != other.remaining_candidates:
            return False
        if self.current_block_number != other.current_block_number:
            return False
        if self.candidates_assigned_so_far != other.candidates_assigned_so_far:
            return False
        if self.remaining_block_sizes != other.remaining_block_sizes:
            return False
        return True

    def __hash__(self):
        return int(self.remaining_candidates + len(
            self.remaining_block_sizes) << 16) + self.current_block_number + self.candidates_assigned_so_far


class FailProbabilityCalculator(abc.ABC):
    """
    Base class for the fail probability calculation
    """

    def __init__(self, k, p, alpha):
        self.k = k
        self.p = p
        self.alpha = alpha

        self.pmf_cache = {}

    @abc.abstractmethod
    def calculate_fail_probability(self, mtable):
        raise NotImplementedError("This is an abstract method. Implement it!")

    def get_from_pmf_cache(self, trials, successes):
        key = (trials, successes)
        if not key in self.pmf_cache:
            # TODO: Check the documentation if this is fine
            self.pmf_cache[key] = binom.pmf(k=successes, n=trials, p=self.p)
        return self.pmf_cache[key]


class RecursiveNumericFailProbabilityCalculator(FailProbabilityCalculator):
    """
    Recursive calculation of fail probability
    """

    EPS = 1e-12

    def __init__(self, k, p, alpha):
        super().__init__(k, p, alpha)

        self.legal_assignment_cache = {}

    def adjust_alpha(self):
        a_min = 0
        a_max = self.alpha
        a_mid = (a_min + a_max) / 2

        minb = self._compute_boundary(a_min)
        maxb = self._compute_boundary(a_max)
        midb = self._compute_boundary(a_mid)

        while minb.mass_of_mtable() < maxb.mass_of_mtable() and midb.fail_prob != self.alpha:
            if midb.fail_prob < self.alpha:
                a_min = a_mid
                minb = self._compute_boundary(a_min)
            elif midb.fail_prob > self.alpha:
                a_max = a_mid
                maxb = self._compute_boundary(a_max)

            a_mid = (a_min + a_max) / 2
            midb = self._compute_boundary(a_mid)

            max_mass = maxb.mass_of_mtable()
            min_mass = minb.mass_of_mtable()
            mid_mass = midb.mass_of_mtable()

            if max_mass - min_mass == 1 or maxb.alpha - minb.alpha <= RecursiveNumericFailProbabilityCalculator.EPS:
                min_diff = abs(minb.fail_prob - self.alpha)
                max_diff = abs(maxb.fail_prob - self.alpha)

                if min_diff <= max_diff:
                    return minb
                else:
                    return maxb

            if max_mass - mid_mass == 1 and mid_mass - min_mass == 1:
                min_diff = abs(minb.fail_prob - self.alpha)
                max_diff = abs(maxb.fail_prob - self.alpha)
                mid_diff = abs(midb.fail_prob - self.alpha)

                if mid_diff <= max_diff and mid_diff <= min_diff:
                    return midb
                if min_diff <= mid_diff and min_diff <= max_diff:
                    return minb
                else:
                    return maxb

        return midb

    def calculate_fail_probability(self, mtable):
        """
        Analytically calculates the fail probability of the mtable
        """
        aux_mtable = compute_aux_mtable(mtable)
        max_protected = aux_mtable['block'].sum()
        block_sizes = aux_mtable['block'].tolist()  # [1:]
        success_prob = self._find_legal_assignments(max_protected, block_sizes)
        return 0 if success_prob == 0 else (1 - success_prob)

    def _compute_boundary(self, alpha):
        """
        Returns a tuple of (k, p, alpha, fail_prob, mtable)
        """
        mtable = MTableGenerator(self.k, self.p, alpha, False).mtable_as_dataframe()
        fail_prob = self.calculate_fail_probability(mtable)
        return MTableFailProbPair(self.k, self.p, alpha, fail_prob, mtable)

    def _find_legal_assignments(self, number_of_candidates, block_sizes):
        return self._find_legal_assignments_aux(number_of_candidates, block_sizes, 1, 0)

    def _find_legal_assignments_aux(
            self,
            number_of_candidates,
            block_sizes,
            current_block_number,
            candidates_assigned_so_far,
    ):
        if len(block_sizes) == 0:
            return 1

        min_needed_this_block = current_block_number - candidates_assigned_so_far
        if min_needed_this_block < 0:
            min_needed_this_block = 0

        max_possible_this_block = min(block_sizes[0], number_of_candidates)

        assignments = 0
        new_remaining_block_sizes = block_sizes[1:]
        for items_this_block in range(min_needed_this_block, max_possible_this_block + 1):
            new_remaining_candidates = number_of_candidates - items_this_block

            suffixes = self._calculate_legal_assignments_aux(
                new_remaining_candidates,
                new_remaining_block_sizes,
                current_block_number + 1,
                candidates_assigned_so_far + items_this_block,
            )

            assignments += self.get_from_pmf_cache(max_possible_this_block, items_this_block) * suffixes

        return assignments

    def _calculate_legal_assignments_aux(
            self,
            remaining_candidates,
            remaining_block_sizes,
            current_block_number,
            candidates_assigned_so_far,
    ):
        key = LegalAssignmentKey(
            remaining_candidates,
            remaining_block_sizes,
            current_block_number,
            candidates_assigned_so_far,
        )

        if not key in self.legal_assignment_cache:
            self.legal_assignment_cache[key] = self._find_legal_assignments_aux(
                remaining_candidates,
                remaining_block_sizes,
                current_block_number,
                candidates_assigned_so_far,
            )

        return self.legal_assignment_cache[key]


class MTableGenerator():
    def __init__(self, K, P, alpha, adjust_alpha):
        self.k = K
        self.p = P
        self.alpha = alpha
        self.adjust_alpha = adjust_alpha

        if self.adjust_alpha:
            fail_prob_pair = RecursiveNumericFailProbabilityCalculator(K, P, alpha).adjust_alpha()
            self.adjusted_alpha = fail_prob_pair.alpha
            self._mtable = fail_prob_pair.mtable
        else:
            self.adjusted_alpha = alpha
            self._mtable = self._compute_mtable()

    def mtable_as_list(self):
        return [int(i) for i in self._mtable.m.tolist()]

    def mtable_as_dataframe(self):
        return self._mtable

    def m(self, k):
        if k < 1:
            raise ValueError("Parameter k must be at least 1")
        elif k > self.k:
            raise ValueError("Parameter k must be at most {0}".format(self.k))

        result = binom.ppf(self.adjusted_alpha if self.adjust_alpha else self.alpha, k, self.p)
        return 0 if result < 0 else result

    def _compute_mtable(self):
        """ Computes a table containing the minimum number of protected elements
            required at each position
        """
        mtable = pd.DataFrame(columns=["m"])
        for i in range(1, self.k + 1):
            if i % 2000 == 0:
                print("Computing m: {:.0f} of {:.0f}".format(i, self.k))
            mtable.loc[i] = [self.m(i)]
        return mtable


class FairRank(FairEstimator):
    """
    FA*IR: A Fair Top-k Ranking Algorithm
    Reference:
        https://dl.acm.org/doi/pdf/10.1145/3132847.3132938
    Code adopted from:
        https://github.com/fair-search/fairsearch-fair-python
    Attributes
    ----------
    s_classes_ : ndarray of shape (n_sensitive_group,)
        A list of sensitive classes known to LabelBias during training.
    Examples
    --------
    >>> from fairpy.model import FairRank
    >>> model = FairRank(K=5, P=0.5, alpha=0.10)
    >>> scores = [0.98, 0.97, 0.85, 0.84, 0.83, 0.55]
    >>> s = ["male", "male", "male", "female", "female", "female"]
    >>> fair_rank = model.transform(scores=scores, s=s)
    """

    _parameter_constraints = {
        "K": [Interval(Integral, 10, 400, closed="both")],
        "P": [Interval(Real, 0.02, 0.98, closed="both")],
        "alpha": [Interval(Real, 0.01, 0.15, closed="both")],
    }

    def __init__(self, K: int, P: float, alpha: float):
        """
        Parameters
        ----------
        K : int
            Number of Top-K elements returned, should be in range [10, 400].
        P : float
            Proportion of sensitive attributes in the Top-K elements, should be in range [0.02, 0.98].
        alpha : float
            Significance level, should be in range [0.01, 0.15].
        """

        self.K = K
        self.P = P
        self.alpha = alpha

    def _create_mtable(self, alpha: float, adjust_alpha: bool) -> List:
        fc = MTableGenerator(self.K, self.P, alpha, adjust_alpha)
        return fc.mtable_as_list()

    def transform(self, scores: npt.ArrayLike, s: npt.ArrayLike) -> List[int]:
        """
        Transform the ranking to be fair based on scores.
        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Scores for ranking samples, where `n_samples` is the number of samples.
        s : array-like of shape (n_samples,)
            Sensitive attributes relative to scores.
        Returns
        -------
        rank : array-like of shape (n_samples,)
            Fair rank with indexes corresponding to scores.
        """

        self._validate_params()
        scores, s = self._validate_data(scores, s)
        s = self._validate_grp_s(s)

        mtable = self._create_mtable(self.alpha, True)

        protected_idx_list = np.where(s == 0)[0]
        non_protected_idx_list = np.where(s == 1)[0]
        N_protected = len(protected_idx_list)
        N_non_protected = len(non_protected_idx_list)

        idx_protected = 0
        idx_non_protected = 0
        res = []
        for i in range(self.K):
            if idx_protected >= N_protected and idx_non_protected >= N_non_protected:
                # no more candidate available
                return res

            elif idx_protected >= N_protected:
                # no more protected candidate available, take non-protected candidate
                res.append(non_protected_idx_list[idx_non_protected])
                idx_non_protected += 1

            elif idx_non_protected >= N_non_protected:
                # no more non-protected candidate available, take protected candidate
                res.append(protected_idx_list[idx_protected])
                idx_protected += 1

            elif idx_protected < mtable[i]:
                # add a protected candidate
                res.append(protected_idx_list[idx_protected])
                idx_protected += 1
            else:
                # find the best candidate available
                if scores[protected_idx_list[idx_protected]] >= scores[non_protected_idx_list[idx_non_protected]]:
                    res.append(protected_idx_list[idx_protected])
                    idx_protected += 1
                else:
                    res.append(non_protected_idx_list[idx_non_protected])
                    idx_non_protected += 1

        return res

    def _more_tags(self):
        return {
            'requires_s': True,
            "s_types": ["binary"],
        }