from functools import partial
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import math
import copy


def compute_distances(
    X,
    n_neighbors,
    range_scaling=None,
    Y=None,
    working_memory=1024,
    n_jobs=1,
    argsort=False,
):
    """
    Description:
        adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
        It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
        instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
        up to neighbors' rank 'range scaling'.
        For big datasets it avoids out of memory errors

    Args:
        range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

    Returns:
        dist (np.ndarray(float)): the FULL distance matrix sorted in increasing order of neighbor distances up to maxk
        neighb_ind np.ndarray(int)): the FULL matrix of the indices of the nearest neighbors up to maxk
        mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
        rs np.ndarray(float)): the FULL matrix of the distances of the neighbors involved in the mu estimates
    """

    reduce_func = partial(
        _kneighbors_reduce_func,
        n_neighbors=n_neighbors,
        range_scaling=range_scaling,
        argsort=argsort,
    )

    kwds = {"squared": True}
    if Y is None:
        Y = X
    chunked_results = list(
        pairwise_distances_chunked(
            X,
            Y,
            reduce_func=reduce_func,
            metric="euclidean",
            n_jobs=n_jobs,
            working_memory=working_memory,
            **kwds,
        )
    )

    if range_scaling is not None:
        neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)
        return (
            np.vstack(neigh_dist),
            np.vstack(neigh_ind).astype("int32"),
            np.vstack(mus),
            np.vstack(rs),
        )
    else:
        neigh_dist, neigh_ind = zip(*chunked_results)
        return (
            np.vstack(neigh_dist),
            np.vstack(neigh_ind).astype("int32"),
        )


def _kneighbors_reduce_func(
    dist, start, n_neighbors, range_scaling=None, argsort=False
):
    """Reduce a chunk of distances to the nearest neighbors.
    Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`
    Parameters
    ----------
    dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.
    start : int
            The index in X which the first row of dist corresponds to.
    n_neighbors : int
            Number of neighbors required for each sample.
    return_distance : bool
            Whether or not to return the distances.
    Returns
    -------
    dist : array of shape (n_samples_chunk, n_neighbors)
            Returned only if `return_distance=True`.
    neigh : array of shape (n_samples_chunk, n_neighbors)
            The neighbors indices.
    """

    if range_scaling is not None:
        "maybe faster without argpartition"
        max_step = int(math.log(range_scaling, 2))
        steps = np.array([2**i for i in range(max_step)])
        sample_range = np.arange(dist.shape[0])[:, None]
        if argsort:
            neigh_ind = np.argsort(dist, axis=1)[:, :range_scaling]
        else:
            neigh_ind = np.argpartition(dist, steps[-1], axis=1)
            neigh_ind = neigh_ind[:, : steps[-1] + 1]
            # argpartition doesn't guarantee sorted order, so we sort again
            neigh_ind = neigh_ind[
                sample_range, np.argsort(dist[sample_range, neigh_ind])
            ]

        "compute mus and rs"
        dist = np.sqrt(dist[sample_range, neigh_ind])

        # find all points with any zero distance
        # find all distances which are 0
        indices = np.nonzero(dist[:, 1:] < np.finfo(dist.dtype).eps)
        # set distance to epsilon
        dist[indices] = np.finfo(dist.dtype).eps

        # indx_ = np.nonzero(dist[:, 1:] < np.finfo(dist.dtype).eps)[0]
        # # set nearest distance to eps:
        # dist[indx_, 1] = 10*np.finfo(dist.dtype).eps

        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
        rs = dist[:, np.array([steps[:-1], steps[1:]])]
        dist = copy.deepcopy(dist[:, :n_neighbors])
        neigh_ind = copy.deepcopy(neigh_ind[:, :n_neighbors])
        return dist, neigh_ind, mus, rs

    else:
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
        neigh_ind1 = copy.deepcopy(neigh_ind[:, :n_neighbors])

    return np.sqrt(dist[sample_range, neigh_ind1[:, :]]), neigh_ind1
