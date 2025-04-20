# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from copy import deepcopy
from string import ascii_uppercase as letters

from castle.common import BaseLearner, Tensor
from castle.algorithms.ges.score.local_scores import (BICScore, BDeuScore, DecomposableScore)

from castle.metrics import MetricsDAG

from castle.algorithms.ges.functional import graph
from castle.algorithms.ges.functional.utils import subset_generator
from castle.algorithms.ges.operators.inserter import insert, insert_validity
from castle.algorithms.ges.operators.deleter import delete, delete_validity

# Dennis Farmer - University of Michigan
# based on castle/algorithms/ges/ges.py implementation
# modified to visualize the progress of the GES algorithm
# into multiple "progress frames"

# pip install gcastle

class GES(BaseLearner):
    """
    Greedy equivalence search for causal discovering

    References
    ----------
    [1]: https://www.sciencedirect.com/science/article/pii/S0888613X12001636
    [2]: https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

    Parameters
    ----------
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

        Notes:
            1. 'bdeu' just for discrete variable.
            2. if you want to customize criterion, you must create a class
            and inherit the base class `DecomposableScore` in module
            `ges.score.local_scores`
    method: str
        effective when `criterion='bic'`, one of ['r2', 'scatter'].
    k: float, default: 0.001
        structure prior, effective when `criterion='bdeu'`.
    N: int, default: 10
        prior equivalent sample size, effective when `criterion='bdeu'`
    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset

    >>> X, true_dag, _ = load_dataset(name='IID_Test')
    >>> algo = GES()
    >>> algo.learn(X)
    >>> GraphDAG(algo.causal_matrix, true_dag, save_name='result_pc')
    >>> met = MetricsDAG(algo.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, criterion='bic', method='scatter', k=0.001, N=10):
        super(GES, self).__init__()
        if isinstance(criterion, str):
            if criterion not in ['bic', 'bdeu']:
                raise ValueError(f"if criterion is str, it must be one of "
                                 f"['bic', 'bdeu'], but got {criterion}.")
        else:
            if not isinstance(criterion, DecomposableScore):
                raise TypeError(f"The criterion is not instance of "
                                f"DecomposableScore.")
        self.criterion = criterion
        self.method = method
        self.k = k
        self.N = N
        self.progress_frames = list() # graph, label, info
        self.progress_scores = list() # MetricDAG

    def learn(self, data, columns=None, **kwargs):
        ground_truth = kwargs.pop("ground_truth")

        d = data.shape[1]
        e = np.zeros((d, d), dtype=int)

        if self.criterion == 'bic':
            self.criterion = BICScore(data=data,
                                      method=self.method)
        elif self.criterion == 'bdeu':
            self.criterion = BDeuScore(data=data, k=self.k, N=self.N)

        c, self.progress_frames, self.progress_scores = fes(C=e, 
                                    criterion=self.criterion, progress_frames=self.progress_frames, 
                                    progress_scores=self.progress_scores, ground_truth=ground_truth)
        c, self.progress_frames, self.progress_scores = bes(C=c, 
                                    criterion=self.criterion, progress_frames=self.progress_frames, 
                                    progress_scores=self.progress_scores, ground_truth=ground_truth)

        self._causal_matrix = Tensor(c, index=columns, columns=columns)

def fes(C, criterion, progress_frames, progress_scores, ground_truth):
    """
    Forward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, t = forward_search(C, criterion)
        if edge is None:
            break
        x, y = edge
        C = insert(x, y, t, C)
        C = graph.pdag_to_cpdag(C)
        progress_frames.append([deepcopy(C), "Forward Equivalence Search", f"(({letters[x]}, {letters[y]}), T={str(set([letters[z] for z in t]))})\nT is a subset of the neighbors of Y that are not adjacent to X"])
        progress_scores.append(MetricsDAG(C, ground_truth))

    return C, progress_frames, progress_scores


def forward_search(C, criterion):
    """
    forward search

    starts with an empty (i.e., no-edge) CPDAG and greedily applies GES
    insert operators until no operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: tuple
        ((X, Y), T), the edge (X, Y) denotes X->Y is valid and T is a subset of
        the neighbors of Y that are not adjacent to X,
    """

    d = C.shape[0]
    edge = None
    subset = {}
    best = 0
    V = np.arange(d)
    for x in V:
        Vy = graph.connect(x, C, relation=None)
        for y in Vy:
            T0 = subset_generator(graph.neighbors(y, C) - graph.adjacent(x, C))
            for T in T0:
                if not insert_validity(x, y, T, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {X} ∪ T ∪ NAY,X ) − f (Y, PaPC (Y) ∪ T ∪ NAY,X ).
                na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
                pa_y = graph.parent(y, C)
                pa1 = pa_y | {x} | T | na_yx
                pa2 = pa_y | T | na_yx
                try:
                    det = (criterion.local_score(y, pa1)
                           - criterion.local_score(y, pa2))
                except AttributeError:
                    raise AttributeError(f"The criterion has no attribute named "
                                         f"`local_score`, you can create a class inherit"
                                         f"`DecomposableScore` and implement `local_score`"
                                         f" method.")

                if det > best:
                    best = det
                    edge = (x, y)
                    subset = T
    return edge, subset


def bes(C, criterion, progress_frames, progress_scores, ground_truth):
    """
    Backward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, h = backward_search(C, criterion)
        if edge is None:
            break
        x, y = edge
        C = delete(x, y, h, C)
        C = graph.pdag_to_cpdag(C)
        progress_frames.append([deepcopy(C), "Backward Equivalence Search", f"(({letters[x]}, {letters[y]}), H={str(set([letters[z] for z in h]))})\nH is a subset of the neighbors of Y that are adjacent to X"])
        progress_scores.append(MetricsDAG(C, ground_truth))

    return C, progress_frames, progress_scores


def backward_search(C, criterion):
    """
    backward search

    starts with a CPDAG and greedily applies GES delete operators until no
    operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: tuple
        ((X, Y), H), the edge (X, Y) denotes X->Y is valid and H is a subset of
        the neighbors of Y that are adjacent to X,
    """

    d = criterion.d
    edge = None
    subset = {}
    best = 0
    V = np.arange(d)
    for x in V:
        Vy = graph.adjacent(x, C)
        for y in Vy:
            H0 = subset_generator(graph.neighbors(y, C) - graph.adjacent(x, C))
            for H in H0:
                if not delete_validity(x, y, H, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {NAY,X \ H} \ X) − f (Y, PaPC (Y) ∪ {NAY,X \ H}).
                na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
                pa_y = graph.parent(y, C)
                pa1 = pa_y | ((na_yx - H) - {x})
                pa2 = pa_y | (na_yx - H)
                det = (criterion.local_score(y, pa1)
                       - criterion.local_score(y, pa2))
                if det > best:
                    best = det
                    edge = (x, y)
                    subset = H

    return edge, subset