import logging
import os

from omegaconf import OmegaConf
import open_clip
import torch
from tqdm import tqdm

from model_merging.data.templates import (
    get_templates,
    dataset_descriptions,
)
from model_merging import PROJECT_ROOT
from model_merging.model.encoder import ClassificationHead, ImageEncoder

from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)


def build_task_classification_head(
    model, task_names, data_location, cache_dir, openclip_cachedir, device
):
    model = ImageEncoder(
        model, cache_dir=cache_dir, openclip_cachedir=openclip_cachedir, keep_lang=True
    ).model

    logit_scale = model.logit_scale

    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []

        for task_name in tqdm(task_names):

            dataset_cfg = OmegaConf.load(
                PROJECT_ROOT / "conf" / "dataset" / f"{task_name}.yaml"
            )

            dataset = instantiate(dataset_cfg, preprocess_fn=None)

            # template = lambda x: f"An image of a {x} from the {task_name} dataset."

            texts = [dataset_descriptions[task_name]]

            texts = open_clip.tokenize(texts).to(device)  # tokenize

            embeddings = model.encode_text(texts)  # embed with text encoder

            if type(embeddings) is tuple:
                embeddings = embeddings[0]

            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def build_classification_head(model, dataset_name, template, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale

    dataset_cfg = OmegaConf.load(
        PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    )

    dataset = instantiate(dataset_cfg, preprocess_fn=None)

    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []

        for classname in tqdm(dataset.classnames):
            # get templates for the class
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device)  # tokenize

            embeddings = model.encode_text(texts)  # embed with text encoder

            if type(embeddings) is tuple:
                embeddings = embeddings[0]

            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(
    model, dataset, ckpt_path, openclip_cachedir, device="cuda"
):
    filename = os.path.join(ckpt_path, f"head_{dataset}.pt")

    try:
        classification_head = load_classification_head(dataset, ckpt_path)
        return classification_head
    except FileNotFoundError:
        pylogger.info(f"Building classification head for {dataset}")

    model = ImageEncoder(
        model, openclip_cachedir=openclip_cachedir, keep_lang=True
    ).model
    template = get_templates(dataset)
    classification_head = build_classification_head(model, dataset, template, device)
    os.makedirs(ckpt_path, exist_ok=True)
    classification_head.save(filename)

    return classification_head


def load_classification_head(dataset, ckpt_path):
    filename = os.path.join(ckpt_path, f"head_{dataset}.pt")
    classification_head = ClassificationHead.load(filename)
    return classification_head
