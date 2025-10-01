from collections import OrderedDict
import copy
import logging
import os
import pickle
import psutil
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import ListConfig
from pytorch_lightning import Callback

pylogger = logging.getLogger(__name__)


def print_memory(context):
    process = psutil.Process(os.getpid())
    pylogger.warning(
        f"{context} -- memory in MB: { process.memory_info().rss / 1024**2}",
    )


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def add_normalized_accuracy(results, finetuning_accuracies):
    for dataset_name, metrics in results.items():
        if dataset_name in finetuning_accuracies:
            normalized_acc = (
                metrics[0][f"acc/test/{dataset_name}"]
                / finetuning_accuracies[dataset_name]
            )
            results[dataset_name][0][
                f"acc/test_normalized/{dataset_name}"
            ] = normalized_acc

    return results


def get_finetuning_accuracies(path):
    with open(path, "rb") as f:
        finetuning_accuracies = json.load(f)
    return finetuning_accuracies


def compute_avg_accuracy(results) -> Dict:
    total_acc = 0
    total_normalized_acc = 0
    count = 0

    for dataset_name, metrics in results.items():
        for m in metrics:
            total_acc += m[f"acc/test/{dataset_name}"]
            total_normalized_acc += m[f"normalized_acc/test/{dataset_name}"]
            count += 1

    average_acc = total_acc / count if count > 0 else 0
    average_normalized_acc = total_normalized_acc / count if count > 0 else 0

    return {
        "acc/test/avg": average_acc,
        "normalized_acc/test/avg": average_normalized_acc,
    }

def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)

# TODO: use this in WeMoE
def print_params_summary(model: torch.nn.Module):
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, ({sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()) * 100}%)"
    )

class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def build_callbacks(cfg: ListConfig, *args: Callback, verbose=False) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        if verbose:
            pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks

# TODO: unify with the below
def pad_unbatched_output(outputs, output_classes):
    """
    Trims a list of unbatched output tensors to match the specified number of output classes,
    then stacks them into a batch.

    Args:
        outputs (list of torch.Tensor): List of tensors with shape (num_classes,) - one per sample.
        output_classes (int): The fixed number of classes to retain in each tensor.

    Returns:
        torch.Tensor: Stacked tensor with shape (batch_size, output_classes).
    """
    trimmed_outputs = []

    for out in outputs:
        num_classes = out.shape[0]

        if num_classes > output_classes:
            out = out[:output_classes]

        elif num_classes < output_classes:
            pad_size = output_classes - num_classes
            pad = torch.zeros(pad_size, device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=0)

        trimmed_outputs.append(out)

    return torch.stack(trimmed_outputs, dim=0)

# TODO: unify with the above
def pad_output(outputs, output_classes):
    """
    Trims a list of output tensors to match the specified number of output classes.

    Args:
        outputs (list of torch.Tensor): List of tensors with shape (batch_size, num_classes).
        output_classes (int): The fixed number of classes to retain in each tensor.

    Returns:
        torch.Tensor: Concatenated tensor with shape (batch_size, output_classes).
    """
    trimmed_outputs = []

    for out in outputs:
        num_classes = out.shape[1]

        if num_classes > output_classes:
            out = out[:, :output_classes]  # Trim exceeding classes

        elif num_classes < output_classes:
            pad_size = output_classes - num_classes
            pad = torch.zeros(
                (out.shape[0], pad_size), device=out.device, dtype=out.dtype
            )
            out = torch.cat([out, pad], dim=1)  # Pad with zeros if necessary

        trimmed_outputs.append(out)

    return torch.cat(trimmed_outputs, dim=0)


def get_hook_fn_impact(model, name):
    """
    Hook function to capture both input and output for impact logging.
    It extracts the token embeddings from both the input and output
    and computes the average L2 norm difference across all tokens.
    """

    def hook_fn(module, input, output):
        # Extract the main tensor from input and output (handle tuple cases)
        inp = input[0] if isinstance(input, tuple) else input
        out = output[0] if isinstance(output, tuple) else output

        # Assuming shape (B, seq_len, hidden); compute per-token L2 norm
        diff = torch.norm(out - inp, p=2, dim=-1)  # Shape: (B, seq_len)

        # Compute the mean impact over all tokens
        avg_diff = diff.mean(dim=1)  # Shape: (B,)

        # Log the results
        model.layer_impact_log[name].append(avg_diff.detach().cpu().numpy())

    return hook_fn


def get_hook_fn(model, name, input_or_output="input"):
    """
    Register a hook to store intermediate features.
    """

    def hook_fn_output(module, input, output):
        if isinstance(output, torch.Tensor):
            model.middle_features[name] = output.cpu().detach()
        elif isinstance(output, tuple):
            model.middle_features[name] = output[0].cpu().detach()

    def hook_fn_input(module, input, output):
        if isinstance(input, torch.Tensor):
            model.middle_features[name] = input.cpu().detach()
        elif isinstance(input, tuple):
            model.middle_features[name] = input[0].cpu().detach()

    hook_fn = hook_fn_output if input_or_output == "output" else hook_fn_input

    return hook_fn


def reconstruct_tv_from_svddict(svd_dict, device="cuda"):
    with torch.no_grad():
        tv = {
            key: (
                (
                    svd_dict[key]["u"]
                    @ torch.diag_embed(svd_dict[key]["s"])
                    @ svd_dict[key]["v"]
                ).to(device)
                if "u" in svd_dict[key]
                else svd_dict[key]["dim1"].to(device)
            )
            for key in svd_dict.keys()
        }

    return tv


def apply_dict_to_model(task_vector_dict, model, coefficient: float = 1.0):
    """
    Applies a task vector dictionary to a model. The resulting model is the deep copy of the input model
    on the GPU with the task vector applied to the weights.
    """
    with torch.no_grad():
        model.cuda()
        new_state_dict = (
            model.state_dict()
        )  # Get model's state_dict (reference, not a copy)

        for key, value in task_vector_dict.items():
            new_key = key.replace("encoder.", "")
            if new_key not in new_state_dict:
                pylogger.warning(
                    f"Key {new_key} is present in the task vector but not in the model"
                )
                continue
            else:
                new_state_dict[new_key] += coefficient * value.cuda()  # Update weight

        model.load_state_dict(new_state_dict, strict=False)  # Load updated parameters
    return model.cuda()


def sum_task_dict(task_vector_dict_1, task_vector_dict_2):
    """
    Sums two task vector dictionaries. It sums task_vector_dict_2 into task_vector_dict_1.
    """
    for key, value in task_vector_dict_2.items():
        if key in task_vector_dict_1:
            task_vector_dict_1[key] += value
        else:
            task_vector_dict_1[key] = value
    return task_vector_dict_1


def is_matrix(layer):
    return len(layer.shape) == 2


def is_matrix_dict(layer):
    return isinstance(layer, dict) and "u" in layer


def get_routing_weights(svd_dict, layer, get_sigma=False, get_u=False):
    """
    Returns the right singular vectors
    """

    vs = []
    sigma = []
    us = []

    for dt in svd_dict.keys():
        layer_key = layer.replace("encoder.", "")

        vs.append(svd_dict[dt][layer_key]["v"].cuda())
        sigma.append(svd_dict[dt][layer_key]["s"].cuda())
        us.append(svd_dict[dt][layer_key]["u"].cuda())

    return (
        torch.stack(vs),
        torch.stack(sigma) if get_sigma else None,
        torch.stack(us) if get_u else None,
    )

def is_supported_layer(layer_key: str) -> bool:
    """
    Check if layer_key contains 'mlp' or 'attn' and 'resblocks.'
    """

    return (
        ("resblocks." in layer_key)
        and (("attn" in layer_key) or ("mlp" in layer_key))
        and not ("ln" in layer_key)
        and not ("gelu" in layer_key)
        and not ("c_proj" in layer_key)
        and not ("c_fc" in layer_key)
    )

def router_key_from_layer(key, index):
    return f"encoder.model.visual.transformer.resblocks.{index}.{key}"


def svd_key_from_layer(key, index):
    base = router_key_from_layer(key, index)
    if "attn" in key:
        return base + ".in_proj_weight"
    elif "mlp" in key:
        return base + ".c_fc.weight"

def from_router_to_svd_dict_key(key):
    key = key.replace("model.encoder.", "")
    if "attn" in key:
        return key + ".in_proj_weight"
    if "mlp" in key:
        return key + ".c_fc.weight"


@torch.no_grad()
def compute_task_dict(pretrained, finetuned):
    new_state_dict = OrderedDict()

    for key in pretrained:
        if pretrained[key].dtype in [torch.int64, torch.uint8]:
            pylogger.info(f"Skipping key {key}")
            continue

        difference = finetuned[key] - pretrained[key]
        new_state_dict[key] = difference

    return new_state_dict


def unzip_all_in_folder(folder_path):
    """
    Unzips all .zip files in the given folder.

    Args:
        folder_path (str): The path to the folder containing zip files.

    Returns:
        None
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    for file in os.listdir(folder_path):
        if file.endswith(".zip"):  # Check if the file is a ZIP archive
            zip_path = os.path.join(folder_path, file)

            # Remove all extensions from the filename
            folder_name = file.split(".")[0]
            extract_path = os.path.join(folder_path, folder_name)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)  # Extract files

            print(f"Extracted: {zip_path} → {extract_path}")


def is_all_zeros(tensor: torch.Tensor | List[torch.Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): A tensor or a list of tensors.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(is_all_zeros(t) for t in tensor)


import logging
import os
import random
from contextlib import contextmanager
from typing import Optional, Union, Dict

import dotenv
import numpy as np
import pytorch_lightning as pl
import torch
import copy
import time

from functools import wraps


pylogger = logging.getLogger(__name__)


def linear_interpolate(
    lambd: float,
    model_a: Union[pl.LightningModule, Dict],
    model_b: Union[pl.LightningModule, Dict],
):
    """
    Linearly interpolate models given as LightningModules or as StateDicts.
    """
    pylogger.info(f"Evaluating interpolated model with lambda: {lambd}")

    if isinstance(model_a, torch.Tensor) and isinstance(model_b, torch.Tensor):
        # flat model parameters, interpolate them as vectors
        return (1 - lambd) * model_a + lambd * model_b

    if is_torch_or_lightning_module(model_a) and is_torch_or_lightning_module(model_b):
        model_a = model_a.state_dict()
        model_b = model_b.state_dict()

    interpolated_model = copy.deepcopy(model_a)

    for param_name in model_a:
        interpolated_model[param_name] = (1 - lambd) * model_a[param_name] + (
            lambd
        ) * model_b[param_name]

    return interpolated_model


def is_torch_or_lightning_module(obj):
    """
    Check if the object is a PyTorch or PyTorch Lightning module.
    """
    return isinstance(obj, (torch.nn.Module, pl.LightningModule))


def to_np(tensor):
    if tensor.nelement() == 1:  # Check if the tensor is a scalar
        return tensor.item()  # Convert a scalar tensor to a Python number
    else:
        return tensor.cpu().detach().numpy()  # Convert a tensor to a numpy array


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """Safely read an environment variable.

    Raises errors if it is not defined or is empty.

    :param env_name: Name of the environment variable.
    :param default: Optional default value if the variable is not set.
    :return: The environment variable's value.
    """
    if env_name not in os.environ:
        if default is None:
            message = f"{env_name} not defined and no default value is present!"
            pylogger.error(message)
            raise KeyError(message)
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            message = (
                f"{env_name} has yet to be configured and no default value is present!"
            )
            pylogger.error(message)
            raise ValueError(message)
        return default

    return env_value





@contextmanager
def environ(**kwargs):
    """Temporarily set the process environment variables.

    https://stackoverflow.com/a/34333710

    >>> with environ(PLUGINS_DIR=u'test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type kwargs: dict[str, unicode]
    :param kwargs: Environment variables to set
    """
    # Use a copy of os.environ to ensure we have an independent backup.
    old_environ = os.environ.copy()
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


# https://github.com/Lightning-AI/lightning/blob/f6a36cf2204b8a6004b11cf0e21879872a63f414/src/lightning/fabric/utilities/seed.py#L19
def _select_seed_randomly(
    min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value
) -> int:
    """Select a random seed within the provided bounds."""
    return random.randint(min_seed_value, max_seed_value)  # noqa: S3


def seed_everything(seed: Optional[int] = None) -> int:
    r"""Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random.

    In addition, sets the following environment variables:
    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, will read seed from ``PL_GLOBAL_SEED`` env variable
            or select it randomly.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly()
            pylogger.warning(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly()
                pylogger.warning(
                    f"Invalid seed found: {env_seed!r}, seed set to {seed}"
                )
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        pylogger.warning(
            f"{seed} is out of bounds (must be between {min_seed_value} and {max_seed_value}). Selecting a new seed."
        )
        seed = _select_seed_randomly()

    pylogger.info(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pylogger.info("PyTorch not installed; skipping torch seeding.")

    return seed


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")

        return result

    return timeit_wrapper
