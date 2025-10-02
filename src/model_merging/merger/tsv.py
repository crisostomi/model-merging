import copy
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.merging.structured import aggregate_decomposed_task_vectors, get_svd_dict
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    is_matrix,
    print_memory,
)
import torch
import copy
from hmac import new
import os
from typing import Tuple

import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
import copy
import os
import torch
import logging
from tqdm import tqdm
from typing import Tuple

pylogger = logging.getLogger(__name__)


class TaskSingularVectorsMerger(TaskVectorBasedMerger):

    def __init__(self, svd_path, svd_compress_factor, non_matrix_params_aggregation):
        super().__init__()

        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
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

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

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

