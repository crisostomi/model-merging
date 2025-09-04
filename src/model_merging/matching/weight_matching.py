import copy
import logging
from enum import auto
from typing import List, Tuple, Union

import numpy as np
from model_merging.matching.utils import (
    LayerIterationOrder,
    compute_weights_similarity,
    get_layer_iteration_order,
    solve_linear_assignment_problem,
)
from model_merging.permutations.permutation_spec import PermutationSpec
from model_merging.permutations.utils import get_permuted_param
import torch
from backports.strenum import StrEnum
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm


pylogger = logging.getLogger(__name__)


def weight_matching(
    ps: PermutationSpec,
    fixed,
    permutee,
    max_iter=100,
    init_perm=None,
    alternate_diffusion_params=None,
    layer_iteration_order: LayerIterationOrder = LayerIterationOrder.RANDOM,
    verbose=False,
):
    """
    Find a permutation of params_b to make them match params_a.

    :param ps: PermutationSpec
    :param target: the parameters to match
    :param to_permute: the parameters to permute
    """

    if not verbose:
        pylogger.setLevel(logging.WARNING)

    params_a, params_b = fixed, permutee

    # # For a MLP of 4 layers it would be something like {'P_0': 512, 'P_1': 512, 'P_2': 512, 'P_3': 256}. Input and output dim are never permuted.
    perm_sizes = {}

    for p, params_and_axes in ps.perm_to_layers_and_axes.items():

        # p is the permutation matrix name, e.g. P_0, P_1, ..
        # params_and_axes is a list of tuples, each tuple contains the name of the parameter and the axis on which the permutation matrix acts

        # it is enough to select a single parameter and axis, since all the parameters permuted by the same matrix have the same shape
        ref_tuple = params_and_axes[0]
        ref_param_name = ref_tuple[0]
        ref_axis = ref_tuple[1]

        perm_sizes[p] = params_a[ref_param_name].shape[ref_axis]

    # initialize with identity permutation if none given
    all_perm_indices = (
        {p: torch.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    # e.g. P0, P1, ..
    perm_names = list(all_perm_indices.keys())

    num_layers = len(perm_names)

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        progress = False

        perm_order = get_layer_iteration_order(layer_iteration_order, num_layers)

        for p_ix in perm_order:
            p = perm_names[p_ix]
            num_neurons = perm_sizes[p]

            sim_matrix = torch.zeros((num_neurons, num_neurons))
            dist_aa = torch.zeros((num_neurons, num_neurons))
            dist_bb = torch.zeros((num_neurons, num_neurons))

            # all the params that are permuted by this permutation matrix, together with the axis on which it acts
            # e.g. ('layer_0.weight', 0), ('layer_0.bias', 0), ('layer_1.weight', 0)..
            params_and_axes: List[Tuple[str, int]] = ps.perm_to_layers_and_axes[p]

            for params_name, axis in params_and_axes:

                w_a = copy.deepcopy(params_a[params_name])
                w_b = copy.deepcopy(params_b[params_name])

                assert w_a.shape == w_b.shape

                perms_to_apply = ps.layer_and_axes_to_perm[params_name]

                w_b = get_permuted_param(
                    w_b, perms_to_apply, all_perm_indices, except_axis=axis
                )

                assert w_a.shape == w_b.shape

                w_a = torch.moveaxis(w_a, axis, 0).reshape((num_neurons, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((num_neurons, -1))

                sim_matrix += w_a @ w_b.T

                dist_aa += torch.cdist(w_a, w_a)
                dist_bb += torch.cdist(w_b, w_b)

            perm_indices = solve_linear_assignment_problem(sim_matrix)

            old_similarity = compute_weights_similarity(sim_matrix, all_perm_indices[p])

            all_perm_indices[p] = perm_indices

            new_similarity = compute_weights_similarity(sim_matrix, all_perm_indices[p])

            pylogger.info(
                f"Iteration {iteration}, Permutation {p}: {new_similarity - old_similarity}"
            )

            progress = progress or new_similarity > old_similarity + 1e-12

        if not progress:
            break

    return all_perm_indices
