from typing import OrderedDict
import logging
import torch

pylogger = logging.getLogger(__name__)


@torch.no_grad()
def compute_task_vector(pretrained, finetuned) -> OrderedDict:
    new_state_dict = OrderedDict()

    for key in pretrained:
        if pretrained[key].dtype in [torch.int64, torch.uint8]:
            pylogger.info(f"Skipping key {key}")
            continue

        difference = finetuned[key].cuda() - pretrained[key].cuda()
        new_state_dict[key] = difference

    return new_state_dict
