from enum import auto
from model_merging.permutations.utils import perm_indices_to_perm_matrix
import torch
from torch import Tensor
from typing import Union
from scipy.optimize import linear_sum_assignment
from backports.strenum import StrEnum
import logging
import numpy as np


class LayerIterationOrder(StrEnum):
    RANDOM = auto()
    FORWARD = auto()
    BACKWARD = auto()


def compute_weights_similarity(similarity_matrix, perm_indices):
    """
    similarity_matrix: matrix s.t. S[i, j] = w_a[i] @ w_b[j]

    we sum over the cells identified by perm_indices, i.e. S[i, perm_indices[i]] for all i
    """

    n = len(perm_indices)

    similarity = torch.sum(similarity_matrix[torch.arange(n), perm_indices.long()])

    return similarity


def solve_linear_assignment_problem(
    sim_matrix: Union[torch.Tensor, np.ndarray], return_matrix=False
):
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().detach().numpy()

    ri, ci = linear_sum_assignment(sim_matrix, maximize=True)

    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    indices = torch.tensor(ci)
    return indices if not return_matrix else perm_indices_to_perm_matrix(indices)


def get_layer_iteration_order(
    layer_iteration_order: LayerIterationOrder, num_layers: int
):
    if layer_iteration_order == LayerIterationOrder.RANDOM:
        return torch.randperm(num_layers)
    elif layer_iteration_order == LayerIterationOrder.FORWARD:
        return torch.arange(num_layers)
    elif layer_iteration_order == LayerIterationOrder.BACKWARD:
        return range(num_layers)[num_layers:0:-1]
    else:
        raise NotImplementedError(
            f"Unknown layer iteration order {layer_iteration_order}"
        )
