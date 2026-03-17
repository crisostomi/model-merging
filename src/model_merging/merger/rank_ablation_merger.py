import copy
import logging

import torch
from tqdm import tqdm

from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.merging.structured import aggregate_decomposed_task_vectors
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    is_matrix,
    print_memory,
)

pylogger = logging.getLogger(__name__)


@torch.no_grad()
def decompose_with_max_rank(task_dicts, max_rank):
    """Like decompose_task_vectors but keeps exactly min(max_rank, full_rank) singular values per layer."""
    svd_dict = {}

    for dataset, task_dict in tqdm(
        task_dicts.items(), desc="Computing SVD with max rank"
    ):
        svd_dict[dataset] = {}

        for key, layer in task_dict.items():

            if is_matrix(layer):
                U, S, V = torch.linalg.svd(layer.float(), full_matrices=False)
                k = min(max_rank, S.shape[0])

                svd_dict[dataset][key] = {
                    "u": U[:, :k].detach().cpu(),
                    "s": S[:k].detach().cpu(),
                    "v": V[:k, :].detach().cpu(),
                }

            else:
                svd_dict[dataset][key] = {"dim1": layer.detach().cpu()}

    return svd_dict


class RankAblationMerger(TaskVectorBasedMerger):

    def __init__(self, max_rank_per_task, non_matrix_params_aggregation="mean"):
        super().__init__()

        self.max_rank_per_task = max_rank_per_task
        self.non_matrix_params_aggregation = non_matrix_params_aggregation

    def merge(self, base_model, finetuned_models):

        task_dicts = {}

        datasets = list(finetuned_models.keys())

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        pylogger.info(f"Decomposing task vectors with max rank per task: {self.max_rank_per_task}")
        svd_dict = decompose_with_max_rank(task_dicts, self.max_rank_per_task)

        multi_task_vector = aggregate_decomposed_task_vectors(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            decomposed_task_vectors=svd_dict,
            non_matrix_params_aggregation=self.non_matrix_params_aggregation,
        )

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
        )

        return merged_encoder
