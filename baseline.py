import csv
from functools import reduce
from itertools import combinations
import math
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import torch


def l2_distance(grads):
    grads_num = len(grads)
    distances = np.zeros([grads_num, grads_num])
    for i in range(grads_num):
        for j in range(grads_num):
            if i < j:
                distances[i, j] = torch.dist(grads[i], grads[j], 2).cpu().numpy()
            elif i > j:
                distances[i, j] = distances[j, i]
            # else : 0
    return distances


# some aggregation methods
# mean: average the grads
def mean(grads):
    return reduce(lambda x, y: x + y, grads) / len(grads)


def krum_initializer(b_workers_num=5):
    """
    Krum defense, see Blanchard. Machine learning with adversaries: Byzantine tolerant gradient descent.NIPS17
    """

    def krum(grads):
        group_size = len(grads) - b_workers_num - 2
        distances = l2_distance(grads)
        sorted_dis = np.sort(distances, axis=-1)
        keep_group = sorted_dis[
            :, 1 : group_size + 1
        ]  # exclude this first one that is itself
        total_dis = np.sum(keep_group, axis=-1)
        return grads[np.argmin(total_dis)]

    return krum


def bulyan_initializer(b_workers_num=5, use_cuda=True):
    """
    see 'The Hidden Vulnerability of Distributed Learning in Byzantium'
    """

    def bulyan_median(arr):
        arr_len = len(arr)
        distances = np.zeros([arr_len, arr_len])
        for i in range(arr_len):
            for j in range(arr_len):
                if i < j:
                    distances[i, j] = abs(arr[i] - arr[j])
                elif i > j:
                    distances[i, j] = distances[j, i]
        total_dis = np.sum(distances, axis=-1)
        median_index = np.argmin(total_dis)
        return median_index, distances[median_index]

    def bulyan_one_coordinate(arr, beta):
        _, distances = bulyan_median(arr)
        median_beta_neighbors = arr[np.argsort(distances)[:beta]]
        return np.mean(median_beta_neighbors)

    def bulyan(grads):
        grads_num = len(grads)
        theta = grads_num - 2 * b_workers_num
        # bulyan cannot do the work here when theta <= 0. Actually, it assumes n >= 4 * f + 3
        selected_grads = []
        # here, we use krum as sub algorithm
        krum = krum_initializer(b_workers_num)
        for i in range(theta):
            krum_grad = krum(grads)
            selected_grads.append(krum_grad)
            for j in range(len(grads)):
                if grads[j] is krum_grad:
                    del grads[j]
                    break

        beta = theta - 2 * b_workers_num
        if use_cuda:
            np_grads = np.array(
                [g.cpu().numpy().flatten().tolist() for g in selected_grads]
            )
        else:
            np_grads = np.array([g.numpy().flatten().tolist() for g in selected_grads])

        grads_dim = len(np_grads[0])
        selected_grads_by_cod = np.zeros([grads_dim, 1])  # shape of torch grads
        for i in range(grads_dim):
            selected_grads_by_cod[i, 0] = bulyan_one_coordinate(np_grads[:, i], beta)

        if use_cuda:
            cuda_tensor = torch.from_numpy(
                selected_grads_by_cod.astype(np.float32)
            ).cuda()
            return cuda_tensor
        else:
            return torch.from_numpy(selected_grads_by_cod.astype(np.float32))

    return bulyan


def brute_initializer(b_workers_num=5):
    """
    see 'The Hidden Vulnerability of Distributed Learning in Byzantium'. one baseline of bulyan
    """

    def brute(grads):
        workers_num = len(grads)
        all_indices = range(workers_num)
        comb_indices = combinations(all_indices, workers_num - b_workers_num)
        most_clumped_grads = None
        minimum = sys.float_info.max
        for g in comb_indices:
            g_grads = [grads[i] for i in g]
            distances = l2_distance(g_grads)
            max_distance = np.max(distances)
            if max_distance < minimum:
                minimum = max_distance
                most_clumped_grads = g_grads

        return mean(most_clumped_grads)

    return brute


def geomed_initializer(use_cuda=True):
    """
    see 'The Hidden Vulnerability of Distributed Learning in Byzantium'. one baseline of bulyan
    """

    def geomed(grads):
        # simply krum with f = -1
        # then krum will select n - f - 2 = n - 1 neighbours
        krum_fn = krum_initializer(b_workers_num=-1)
        return krum_fn(grads)

    return geomed


def classical_initializer():
    """
    classical GAR
    """

    def classic_GAR(grads):
        return mean(grads)

    return classic_GAR


def minimize_method(points):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist(np.asarray([x.tolist()]), points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)
    optimize_result = minimize(aggregate_distance, centroid, method="COBYLA")

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """

    default_options = {"maxiter": 1000, "tol": 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist(np.asarray([x.tolist()]), points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options["maxiter"]:
        distances = distance_func(guess).T

        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points / distances).sum(axis=0) / (1.0 / distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next) ** 2).sum())

        guess = guess_next

        if guess_movement <= options["tol"]:
            break

        iters += 1

    return guess


def geomed_plus_initializer(use_cuda=False, b_workers_num=5, store_path=None):
    """
    since it only needs to run for one loop, I think there is no need
    """
    f = open(store_path, "w+")
    writer = csv.writer(f)

    def geomed_plus(grads):
        selection = []
        mask = torch.ones(len(grads), len(grads))
        if use_cuda:
            mask = mask.cuda()

        ## do the top-k selection without replacement
        for k in range(1, b_workers_num + 1):
            distances = l2_distance(grads)
            distances *= mask
            total_dis = np.sum(distances, axis=-1)
            victim = np.argmax(total_dis)
            mask[victim, :] = 0
            mask[:, victim] = 0
            selection.append(victim)
        writer.writerow(selection)
        return grads[0]

    return geomed_plus
