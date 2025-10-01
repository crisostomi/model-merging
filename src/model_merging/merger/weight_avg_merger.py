import copy
import logging
from typing import Dict, List
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    sum_task_dict,
)

pylogger = logging.getLogger(__name__)


class WeightAverageMerger(TaskVectorBasedMerger):

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, Dict]
    ) -> ImageEncoder:

        merged_model = copy.deepcopy(base_model)

        # Collect model keys
        datasets = list(finetuned_models.keys())
        num_models = len(datasets)

        # Initialize accumulator
        avg_state = {}
        for key in finetuned_models[datasets[0]].keys():
            # sum the same key across all finetuned models
            avg_state[key] = (
                sum(finetuned_models[ds][key] for ds in datasets) / num_models
            )

        # Load averaged weights
        merged_model.load_state_dict(avg_state, strict=True)

        return merged_model
