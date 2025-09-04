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


def load_envs(env_file: Optional[str] = None) -> None:
    """Load environment variables from a file.

    This is equivalent to sourcing the file in a shell.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: The file that defines the environment variables to use. If None,
                     it searches for a `.env` file in the project.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


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
