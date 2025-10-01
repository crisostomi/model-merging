import logging
import os
from pathlib import Path
from typing import Optional

import dotenv
import git

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

from rich.logging import RichHandler

# from rich.traceback import install

# # Enable pretty tracebacks
# install(show_locals=True)

# Set up Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

load_envs()

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

__all__ = ["PROJECT_ROOT"]



from omegaconf import OmegaConf


def dataset_resolver(dataset_name: str):  # NOQA
    dataset_cfg = OmegaConf.load(
        PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    )

    return dataset_cfg


OmegaConf.register_new_resolver("dataset", dataset_resolver)
