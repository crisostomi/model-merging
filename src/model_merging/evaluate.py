import copy
import logging
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import omegaconf
from model_merging.permutations.utils import get_model
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import EMNIST
from tqdm import tqdm
import torch

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from model_merging.utils import linear_interpolate

pylogger = logging.getLogger(__name__)


def evaluate_pair_of_models(
    models,
    fixed_id,
    permutee_id,
    updated_params,
    train_loader,
    test_loader,
    lambdas,
    cfg,
):
    fixed_model = models[fixed_id]
    permutee_model = models[permutee_id]

    get_model(permutee_model).load_state_dict(updated_params[fixed_id][permutee_id])

    results = evaluate_interpolated_models(
        fixed_model,
        permutee_model,
        train_loader,
        test_loader,
        lambdas,
        cfg.matching,
        repair=False,
    )

    return results


def evaluate_interpolated_models(
    fixed, permutee, train_loader, test_loader, lambdas, cfg, repair=False
):
    fixed = fixed.cuda()
    permutee = permutee.cuda()

    fixed_dict = copy.deepcopy(get_model(fixed).state_dict())
    permutee_dict = copy.deepcopy(get_model(permutee).state_dict())

    results = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }
    trainer = instantiate(
        cfg.trainer, enable_progress_bar=False, enable_model_summary=False
    )

    for lam in tqdm(lambdas):

        interpolated_model = copy.deepcopy(permutee)

        interpolated_params = linear_interpolate(lam, fixed_dict, permutee_dict)
        get_model(interpolated_model).load_state_dict(interpolated_params)

        if repair and lam > 1e-5 and lam < 1 - 1e-5:
            print("Repairing model")
            interpolated_model = repair_model(
                interpolated_model, {"a": fixed, "c": permutee}, train_loader
            )

        train_results = trainer.test(interpolated_model, train_loader)

        results["train_acc"].append(train_results[0]["acc/test"])
        results["train_loss"].append(train_results[0]["loss/test"])

        if test_loader:
            test_results = trainer.test(interpolated_model, test_loader)

            results["test_acc"].append(test_results[0]["acc/test"])
            results["test_loss"].append(test_results[0]["loss/test"])

    train_loss_barrier = compute_loss_barrier(results["train_loss"])
    results["train_loss_barrier"] = train_loss_barrier

    if test_loader:
        test_loss_barrier = compute_loss_barrier(results["test_loss"])
        results["test_loss_barrier"] = test_loss_barrier

    return results


def log_results(results, lambdas):

    for metric in ["acc", "loss"]:
        for mode in ["train", "test"]:
            for step, lam in enumerate(lambdas):
                wandb.log(
                    {
                        f"{mode}_{metric}": results[f"{mode}_{metric}"][step],
                        "lambda": lam,
                    }
                )

    for mode in ["train", "test"]:
        wandb.log({f"{mode}_loss_barrier": results[f"{mode}_loss_barrier"]})


def compute_loss_barrier(losses):
    """
    max_{lambda in [0,1]} loss(alpha * model_a + (1 - alpha) * model_b) - 0.5 * (loss(model_a) + loss(model_b))
    """
    model_a_loss = losses[0]
    model_b_loss = losses[-1]

    avg_loss = (model_a_loss + model_b_loss) / 2

    return max(losses) - avg_loss
