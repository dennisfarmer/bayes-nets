
from copy import deepcopy
from itertools import combinations, permutations
import numpy as np
import joblib
from string import ascii_uppercase as letters

from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import CITest
from castle.common.priori_knowledge import orient_by_priori_knowledge
from castle.metrics import MetricsDAG

# Dennis Farmer - University of Michigan
# based on castle/algorithms/pc/pc.py implementation
# modified to visualize the progress of the PC algorithm
# into multiple "progress frames"

# pip install gcastle

class PC(BaseLearner):
    def __init__(self, variant='original', alpha=0.05, ci_test='fisherz',
                 priori_knowledge=None):
        super(PC, self).__init__()
        self.variant = variant
        self.alpha = alpha
        self.ci_test = ci_test
        self._causal_matrix = None
        self.priori_knowledge = priori_knowledge
        self.progress_frames = list() # graph, label, info
        self.progress_scores = list() # MetricDAG
        self.str_sep_set=""

    def learn(self, data, columns=None, **kwargs):
        ground_truth = kwargs.pop("ground_truth")
        data = Tensor(data, columns=columns)

        skeleton, sep_set, self.progress_frames, self.progress_scores = find_skeleton(data,
                                          alpha=self.alpha,
                                          ci_test=self.ci_test,
                                          variant=self.variant,
                                          priori_knowledge=self.priori_knowledge,
                                          progress_frames=self.progress_frames,
                                          progress_scores=self.progress_scores,
                                          ground_truth=ground_truth,
                                          **kwargs)


        self.str_sep_set = str(sep_set)
        list_letters = list(letters)
        list_letters.reverse()
        for num, letter in enumerate(list_letters):
            self.str_sep_set=self.str_sep_set.replace(str(np.abs(num-25)), letter)
        print("SEP_SET:\n", self.str_sep_set, sep="")
        equivalence_class, self.progress_frames, self.progress_scores = orient(skeleton, 
                                                         sep_set, 
                                                         self.priori_knowledge, 
                                                         self.progress_frames,
                                                         self.progress_scores,
                                                         ground_truth=ground_truth,
                                                         )
        equivalence_class = equivalence_class.astype(int)
        self._causal_matrix = Tensor(
            equivalence_class,
            index=data.columns,
            columns=data.columns
        )


def _loop(G, d):
    """
    Check if |adj(x, G)\{y}| < d for every pair of adjacency vertices in G

    Parameters
    ----------
    G: numpy.ndarray
        The undirected graph  G
    d: int
        depth of conditional vertices

    Returns
    -------
    out: bool
        if False, denote |adj(i, G)\{j}| < d for every pair of adjacency
        vertices in G, then finished loop.
    """

    assert G.shape[0] == G.shape[1]

    pairs = [(x, y) for x, y in combinations(set(range(G.shape[0])), 2)]
    less_d = 0
    for i, j in pairs:
        adj_i = set(np.argwhere(G[i] != 0).reshape(-1, ))
        z = adj_i - {j}  # adj(C, i)\{j}
        if len(z) < d:
            less_d += 1
        else:
            break
    if less_d == len(pairs):
        return False
    else:
        return True


def orient(skeleton, sep_set, priori_knowledge=None, progress_frames=[], progress_scores=[], ground_truth=None):
    """Extending the Skeleton to the Equivalence Class

    it orients the undirected edges to form an equivalence class of DAGs.

    Parameters
    ----------
    skeleton : array
        The undirected graph
    sep_set : dict
        separation sets
        if key is (x, y), then value is a set of other variables
        not contains x and y

    Returns
    -------
    out : array
        An equivalence class of DAGs can be uniquely described
        by a completed partially directed acyclic graph (CPDAG)
        which includes both directed and undirected edges.
    """

    if priori_knowledge is not None:
        skeleton = orient_by_priori_knowledge(skeleton, priori_knowledge)

    columns = list(range(skeleton.shape[1]))
    cpdag = deepcopy(abs(skeleton))
    # pre-processing
    for ij in sep_set.keys():
        i, j = ij
        all_k = [x for x in columns if x not in ij]
        for k in all_k:
            if cpdag[i, k] + cpdag[k, i] != 0 \
                    and cpdag[k, j] + cpdag[j, k] != 0:
                if k not in sep_set[ij]:
                    if cpdag[i, k] + cpdag[k, i] == 2:
                        cpdag[k, i] = 0
                        #progress_frames.append([deepcopy(cpdag), "orient", f"pre:i={letters[i]},j={letters[j]},k={letters[k]}"])
                        progress_frames.append([deepcopy(cpdag), "orient", f"pre:({letters[i]}, {letters[j]}, {letters[k]})"])
                        progress_scores.append(MetricsDAG(cpdag, ground_truth))
                    if cpdag[j, k] + cpdag[k, j] == 2:
                        cpdag[k, j] = 0
                        #progress_frames.append([deepcopy(cpdag), "orient", f"pre:i={letters[i]},j={letters[j]},k={letters[k]}"])
                        progress_frames.append([deepcopy(cpdag), "orient", f"pre:({letters[i]}, {letters[j]}, {letters[k]})"])
                        progress_scores.append(MetricsDAG(cpdag, ground_truth))
    while True:
        old_cpdag = deepcopy(cpdag)
        pairs = list(combinations(columns, 2))
        for ij in pairs:
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 1:
                # rule1
                for i, j in permutations(ij, 2):
                    all_k = [x for x in columns if x not in ij]
                    for k in all_k:
                        if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                                and cpdag[k, j] + cpdag[j, k] == 0:
                            cpdag[j, i] = 0
                            progress_frames.append([deepcopy(cpdag), "orient", f"rule1:({letters[i]}, {letters[j]}, {letters[k]})"])
                            progress_scores.append(MetricsDAG(cpdag, ground_truth))
                # rule2
                for i, j in permutations(ij, 2):
                    all_k = [x for x in columns if x not in ij]
                    for k in all_k:
                        if (cpdag[i, k] == 1 and cpdag[k, i] == 0) \
                            and (cpdag[k, j] == 1 and cpdag[j, k] == 0):
                            cpdag[j, i] = 0
                            progress_frames.append([deepcopy(cpdag), "orient", f"rule2:({letters[i]}, {letters[j]}, {letters[k]})"])
                            progress_scores.append(MetricsDAG(cpdag, ground_truth))
                # rule3
                for i, j in permutations(ij, 2):
                    for kl in sep_set.keys():  # k and l are nonadjacent.
                        k, l = kl
                        # if i——k——>j and  i——l——>j
                        if cpdag[i, k] == 1 \
                                and cpdag[k, i] == 1 \
                                and cpdag[k, j] == 1 \
                                and cpdag[j, k] == 0 \
                                and cpdag[i, l] == 1 \
                                and cpdag[l, i] == 1 \
                                and cpdag[l, j] == 1 \
                                and cpdag[j, l] == 0:
                            cpdag[j, i] = 0
                            progress_frames.append([deepcopy(cpdag), "orient", f"rule3:({letters[i]}, {letters[j]}, {letters[k]})"])
                            progress_scores.append(MetricsDAG(cpdag, ground_truth))
                # rule
                for i, j in permutations(ij, 2):
                    for kj in sep_set.keys():  # k and j are nonadjacent.
                        if j not in kj:
                            continue
                        else:
                            kj = list(kj)
                            kj.remove(j)
                            k = kj[0]
                            ls = [x for x in columns if x not in [i, j, k]]
                            for l in ls:
                                if cpdag[k, l] == 1 \
                                        and cpdag[l, k] == 0 \
                                        and cpdag[i, k] == 1 \
                                        and cpdag[k, i] == 1 \
                                        and cpdag[l, j] == 1 \
                                        and cpdag[j, l] == 0:
                                    cpdag[j, i] = 0
                                    progress_frames.append([deepcopy(cpdag), "orient", f"rule4:({letters[i]}, {letters[j]}, {letters[k]})"])
                                    progress_scores.append(MetricsDAG(cpdag, ground_truth))
        if np.all(cpdag == old_cpdag):
            break

    progress_frames.append([deepcopy(cpdag), "final", ""])
    progress_scores.append(MetricsDAG(cpdag, ground_truth))
    return cpdag, progress_frames, progress_scores


def find_skeleton(data, alpha, ci_test, variant='original',
                  priori_knowledge=None, base_skeleton=None,
                  p_cores=1, s=None, batch=None, progress_frames=[], progress_scores=[], ground_truth=None):
    """Find skeleton graph from G using PC algorithm

    It learns a skeleton graph which contains only undirected edges
    from data.

    Parameters
    ----------
    data : array, (n_samples, n_features)
        Dataset with a set of variables V
    alpha : float, default 0.05
        significant level
    ci_test : str, callable
        ci_test method, if str, must be one of [`fisherz`, `g2`, `chi2`].
        if callable, must return a tuple that  the last element is `p_value` ,
        like (_, _, p_value) or (chi2, dof, p_value).
        See more: `castle.common.independence_tests.CITest`
    variant : str, default 'original'
        variant of PC algorithm, contains [`original`, `stable`, `parallel`].
        If variant == 'parallel', need to provide the flowing 3 parameters.
    base_skeleton : array, (n_features, n_features)
        prior matrix, must be undirected graph.
        The two conditionals `base_skeleton[i, j] == base_skeleton[j, i]`
        and `and base_skeleton[i, i] == 0` must be satisified which i != j.
    p_cores : int
        Number of CPU cores to be used
    s : bool, default False
        memory-efficient indicator
    batch : int
        number of edges per batch

    if s is None or False, or without batch, batch=|J|.
    |J| denote number of all pairs of adjacency vertices (X, Y) in G.

    Returns
    -------
    skeleton : array
        The undirected graph
    seq_set : dict
        Separation sets
        Such as key is (x, y), then value is a set of other variables
        not contains x and y.

    Examples
    --------
    >>> from castle.algorithms.pc.pc import find_skeleton
    >>> from castle.datasets import load_dataset

    >>> true_dag, X = load_dataset(name='iid_test')
    >>> skeleton, sep_set = find_skeleton(data, 0.05, 'fisherz')
    >>> print(skeleton)
    [[0. 0. 1. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 1. 1. 1. 1. 0. 1. 0.]
     [1. 0. 0. 0. 1. 0. 0. 1. 0. 0.]
     [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
     [0. 1. 1. 1. 0. 0. 0. 0. 0. 1.]
     [1. 1. 0. 0. 0. 0. 0. 1. 1. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 0. 1. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 1. 0. 0. 0. 1.]
     [0. 0. 0. 1. 1. 1. 0. 1. 1. 0.]]
    """

    def test(x, y):

        K_x_y = 1
        sub_z = None
        # On X's neighbours
        adj_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
        z_x = adj_x - {y}  # adj(X, G)\{Y}
        if len(z_x) >= d:
            # |adj(X, G)\{Y}| >= d
            for sub_z in combinations(z_x, d):
                sub_z = list(sub_z)
                _, _, p_value = ci_test(data, x, y, sub_z)
                if p_value >= alpha:
                    K_x_y = 0
                    # sep_set[(x, y)] = sub_z
                    break
            if K_x_y == 0:
                return K_x_y, sub_z

        return K_x_y, sub_z

    def parallel_cell(x, y):

        # On X's neighbours
        K_x_y, sub_z = test(x, y)
        if K_x_y == 1:
            # On Y's neighbours
            K_x_y, sub_z = test(y, x)

        return (x, y), K_x_y, sub_z

    if ci_test == 'fisherz':
        ci_test = CITest.fisherz_test
    elif ci_test == 'g2':
        ci_test = CITest.g2_test
    elif ci_test == 'chi2':
        ci_test = CITest.chi2_test
    elif callable(ci_test):
        ci_test = ci_test
    else:
        raise ValueError(f'The type of param `ci_test` expect callable,'
                         f'but got {type(ci_test)}.')

    n_feature = data.shape[1]
    if base_skeleton is None:
        skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
    else:
        row, col = np.diag_indices_from(base_skeleton)
        base_skeleton[row, col] = 0
        skeleton = base_skeleton
    nodes = set(range(n_feature))

    # update skeleton based on priori knowledge
    for i, j in combinations(nodes, 2):
        if priori_knowledge is not None and (
                priori_knowledge.is_forbidden(i, j)
                and priori_knowledge.is_forbidden(j, i)):
            skeleton[i, j] = skeleton[j, i] = 0

    sep_set = {}
    d = -1
    progress_frames.append([deepcopy(skeleton), "skeleton", "initial state"])
    progress_scores.append(MetricsDAG(skeleton, ground_truth))
    while _loop(skeleton, d):  # until for each adj(C,i)\{j} < l
        d += 1
        if variant == 'stable':
            C = deepcopy(skeleton)
        else:
            C = skeleton
        if variant != 'parallel':
            for i, j in combinations(nodes, 2):
                if skeleton[i, j] == 0:
                    continue

                # Causation_Prediction_and_Search.pdf:
                # Let Adjacencies(C, A) be the set of vertices adjacent to A
                # in directed acyclic graph C.

                # adj(C, i): set of variables that i'th variable is directed towards
                # (columns where 1's/edges exist in i'th row)
                adj_i = set(np.argwhere(C[i] == 1).reshape(-1, ))

                # removal of the arc we are testing
                z = adj_i - {j}  # adj(C, i)\{j}

                if len(z) >= d:
                    # |adj(C, i)\{j}| >= l
                    for sub_z in combinations(z, d):
                        # sub_z: set of variables we are conditioning on
                        sub_z = list(sub_z)
                        # testing for independence, conditioning on sub_z
                        _, _, p_value = ci_test(data, i, j, sub_z)
                        if p_value >= alpha:
                            skeleton[i, j] = skeleton[j, i] = 0
                            progress_frames.append([deepcopy(skeleton), "skeleton", f"{letters[i]}||{letters[j]} | {str(set([letters[z] for z in sub_z]))}"])
                            progress_scores.append(MetricsDAG(skeleton, ground_truth))
                            sep_set[(i, j)] = sub_z
                            break
        else:
            J = [(x, y) for x, y in combinations(nodes, 2)
                 if skeleton[x, y] == 1]
            if not s or not batch:
                batch = len(J)
            if batch < 1:
                batch = 1
            if not p_cores or p_cores == 0:
                raise ValueError(f'If variant is parallel, type of p_cores '
                                 f'must be int, but got {type(p_cores)}.')
            for i in range(int(np.ceil(len(J) / batch))):
                each_batch = J[batch * i: batch * (i + 1)]
                parallel_result = joblib.Parallel(n_jobs=p_cores,
                                                  max_nbytes=None)(
                    joblib.delayed(parallel_cell)(x, y) for x, y in
                    each_batch
                )
                # Synchronisation Step
                for (x, y), K_x_y, sub_z in parallel_result:
                    if K_x_y == 0:
                        skeleton[x, y] = skeleton[y, x] = 0
                        sep_set[(x, y)] = sub_z

    return skeleton, sep_set, progress_frames, progress_scores
