"""
Activation Space Analysis: How EuroSAT Disrupts SVHN Representations

Compares intermediate representations on SVHN (and EuroSAT) data between:
1. Pretrained CLIP model (baseline)
2. SVHN fine-tuned model (ideal for SVHN)
3. All-8 merged model (MP-edge + isotropic)
4. No-EuroSAT merged model (7 tasks, excl. EuroSAT)

Key metrics at each transformer block:
- Linear CKA (centered kernel alignment)
- Per-sample cosine similarity
- Activation L2 norm statistics
- Representation shift (L2 distance)

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_activations.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_activations.py --quick
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.model.encoder import ImageEncoder
from model_merging.merger.interference_aware_merger import InterferenceAwareMerger
from model_merging.utils.utils import print_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
MODEL_NAME = "ViT-B-32"
OUTPUT_DIR = PROJECT_ROOT / "results" / "activation_analysis"
N_SAMPLES = 500
BATCH_SIZE = 64

# HF dataset specs: (hf_name, config_name, image_column)
DATASET_HF_MAP = {
    "SVHN": ("ufldl-stanford/svhn", "cropped_digits", "image"),
    "EuroSAT": ("tanganke/eurosat", None, "image"),
    "MNIST": ("ylecun/mnist", None, "image"),
}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    """Load a state dict from the HuggingFace cache."""
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def create_encoder(model_name=MODEL_NAME):
    """Create a fresh ImageEncoder (pretrained CLIP weights)."""
    return ImageEncoder(model_name)


def load_finetuned_encoder(dataset_name, model_name=MODEL_NAME):
    """Load a fine-tuned encoder by applying fine-tuned weights."""
    encoder = create_encoder(model_name)
    ft_sd = load_state_dict_from_hf_cache(model_name, dataset_name)
    encoder.load_state_dict(ft_sd, strict=False)
    return encoder


def create_merged_model(datasets, model_name=MODEL_NAME):
    """Create a merged model using MP-edge + isotropic merger."""
    logger.info(f"Creating merged model with {len(datasets)} tasks: {datasets}")
    encoder = create_encoder(model_name)
    finetuned = {}
    for ds in datasets:
        logger.info(f"  Loading finetuned: {ds}")
        finetuned[ds] = load_state_dict_from_hf_cache(model_name, ds)

    merger = InterferenceAwareMerger(
        use_mp_edge=True,
        use_isotropic=True,
        mp_min_rank=4,
        mp_max_rank=128,
    )
    merged = merger.merge(encoder, finetuned)
    return merged


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset_samples(dataset_name, preprocess, n_samples=N_SAMPLES):
    """Load n_samples from a dataset's test split."""
    from datasets import load_dataset as hf_load_dataset

    hf_name, config, img_col = DATASET_HF_MAP[dataset_name]

    logger.info(f"Loading {dataset_name} data ({hf_name}, config={config})...")
    if config:
        ds = hf_load_dataset(hf_name, config, split="test")
    else:
        ds = hf_load_dataset(hf_name, split="test")

    images = []
    labels = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        img = preprocess(sample[img_col].convert("RGB"))
        images.append(img)
        labels.append(sample["label"])

    logger.info(f"  Loaded {len(images)} samples from {dataset_name}")
    return torch.stack(images), torch.tensor(labels)


# --------------------------------------------------------------------------- #
# Activation extraction
# --------------------------------------------------------------------------- #

def register_hooks(model):
    """Register forward hooks on all transformer resblocks and ln_post.

    Returns (activations_dict, hook_handles) where activations_dict is populated
    during forward passes.
    """
    activations = {}
    handles = []

    visual = model.model.visual

    for i, block in enumerate(visual.transformer.resblocks):
        name = f"block_{i}"

        def hook(module, input, output, name=name):
            # Resblock output: (seq_len, batch, hidden_dim) — seq-first format
            activations[name] = output.detach().cpu()

        handles.append(block.register_forward_hook(hook))

    # ln_post receives the CLS token after pooling: (batch, hidden_dim)
    def ln_post_hook(module, input, output):
        activations["ln_post"] = output.detach().cpu()

    handles.append(visual.ln_post.register_forward_hook(ln_post_hook))

    return activations, handles


@torch.no_grad()
def extract_all_activations(model, images, device, batch_size=BATCH_SIZE):
    """Run images through model and return per-layer activations.

    Returns dict: {layer_name: tensor} where tensor shapes are:
      - block_*: (n_samples, hidden_dim) — CLS token only
      - ln_post: (n_samples, hidden_dim)
      - output: (n_samples, output_dim)
    """
    model.eval()
    model.to(device)

    all_cls = defaultdict(list)
    all_outputs = []

    activations_dict, handles = register_hooks(model)

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        out = model(batch)  # (batch, output_dim)
        all_outputs.append(out.cpu())

        for name, act in activations_dict.items():
            if act.dim() == 3:
                # (seq_len, batch, hidden) → CLS token is at seq index 0
                all_cls[name].append(act[0])  # (batch, hidden)
            else:
                # (batch, hidden) for ln_post
                all_cls[name].append(act)
        activations_dict.clear()

    for h in handles:
        h.remove()

    model.cpu()
    torch.cuda.empty_cache()

    result = {}
    for name, tensors in all_cls.items():
        result[name] = torch.cat(tensors, dim=0).float()  # (n_samples, hidden)
    result["output"] = torch.cat(all_outputs, dim=0).float()

    return result


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def linear_cka(X, Y):
    """Compute linear CKA between X (n, d1) and Y (n, d2)."""
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    hsic_xy = torch.norm(Y.T @ X, "fro") ** 2
    hsic_xx = torch.norm(X.T @ X, "fro")
    hsic_yy = torch.norm(Y.T @ Y, "fro")

    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return float(hsic_xy / (hsic_xx * hsic_yy))


def pairwise_cosine_sim(X, Y):
    """Per-sample cosine similarity between X and Y (n, d). Returns mean, std, all."""
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-10)
    cos = (X_norm * Y_norm).sum(dim=1)
    return float(cos.mean()), float(cos.std()), cos.numpy()


def activation_norm_stats(X):
    """Mean and std of per-sample L2 norms."""
    norms = X.norm(dim=1)
    return float(norms.mean()), float(norms.std())


def l2_distance_stats(X, Y):
    """Per-sample L2 distance statistics."""
    diff_norms = (X - Y).norm(dim=1)
    ref_norms = X.norm(dim=1)
    relative = diff_norms / (ref_norms + 1e-10)
    return {
        "mean": float(diff_norms.mean()),
        "std": float(diff_norms.std()),
        "relative_mean": float(relative.mean()),
        "relative_std": float(relative.std()),
    }


# --------------------------------------------------------------------------- #
# Comparison logic
# --------------------------------------------------------------------------- #

def compare_models(all_activations, model_names, layer_names, probe_dataset):
    """Compare activations across models at each layer.

    Args:
        all_activations: {model_name: {layer_name: (n, d) tensor}}
        model_names: list of model names to compare
        layer_names: ordered list of layer names
        probe_dataset: name of dataset used for probing

    Returns:
        dict of results per layer
    """
    logger.info(f"Comparing {len(model_names)} models on {probe_dataset} data across {len(layer_names)} layers")
    results = {}

    for layer in layer_names:
        reps = {m: all_activations[m][layer] for m in model_names}

        # Pairwise CKA
        cka = {}
        for m1, m2 in combinations(model_names, 2):
            key = f"{m1}_vs_{m2}"
            cka[key] = linear_cka(reps[m1], reps[m2])

        # Pairwise cosine similarity
        cos_sim = {}
        for m1, m2 in combinations(model_names, 2):
            key = f"{m1}_vs_{m2}"
            mean_c, std_c, _ = pairwise_cosine_sim(reps[m1], reps[m2])
            cos_sim[key] = {"mean": mean_c, "std": std_c}

        # Per-model activation norms
        norms = {}
        for m in model_names:
            mean_n, std_n = activation_norm_stats(reps[m])
            norms[m] = {"mean": mean_n, "std": std_n}

        # Pairwise L2 distances
        l2_dist = {}
        for m1, m2 in combinations(model_names, 2):
            key = f"{m1}_vs_{m2}"
            l2_dist[key] = l2_distance_stats(reps[m1], reps[m2])

        results[layer] = {
            "cka": cka,
            "cosine_similarity": cos_sim,
            "activation_norms": norms,
            "l2_distances": l2_dist,
        }

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

PAIR_COLORS = {
    "merged_8_vs_merged_7": "tab:red",
    "merged_8_vs_svhn_ft": "tab:blue",
    "merged_7_vs_svhn_ft": "tab:green",
    "pretrained_vs_merged_8": "tab:orange",
    "pretrained_vs_merged_7": "tab:purple",
    "pretrained_vs_svhn_ft": "tab:brown",
}

PAIR_LABELS = {
    "merged_8_vs_merged_7": "All8 vs No-EuroSAT",
    "merged_8_vs_svhn_ft": "All8 vs SVHN-FT",
    "merged_7_vs_svhn_ft": "No-EuroSAT vs SVHN-FT",
    "pretrained_vs_merged_8": "Pretrained vs All8",
    "pretrained_vs_merged_7": "Pretrained vs No-EuroSAT",
    "pretrained_vs_svhn_ft": "Pretrained vs SVHN-FT",
}


def plot_cka_curves(results, layer_names, probe_dataset, output_dir):
    """Plot CKA across layers for key model pairs."""
    block_layers = [l for l in layer_names if l.startswith("block_")]
    block_indices = list(range(len(block_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Focus on the most informative pairs
    focus_pairs = [
        "merged_8_vs_merged_7",
        "merged_8_vs_svhn_ft",
        "merged_7_vs_svhn_ft",
        "pretrained_vs_merged_8",
    ]

    for pair in focus_pairs:
        values = [results[l]["cka"].get(pair, None) for l in block_layers]
        if values[0] is None:
            continue
        color = PAIR_COLORS.get(pair, "gray")
        label = PAIR_LABELS.get(pair, pair)
        ax.plot(block_indices, values, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Transformer Block", fontsize=11)
    ax.set_ylabel("Linear CKA", fontsize=11)
    ax.set_title(f"Representation Similarity Across Layers ({probe_dataset} data)", fontsize=12)
    ax.set_xticks(block_indices)
    ax.set_xticklabels([l.replace("block_", "") for l in block_layers])
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / f"cka_curves_{probe_dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_cosine_sim_curves(results, layer_names, probe_dataset, output_dir):
    """Plot mean cosine similarity across layers."""
    block_layers = [l for l in layer_names if l.startswith("block_")]
    block_indices = list(range(len(block_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))

    focus_pairs = [
        "merged_8_vs_merged_7",
        "merged_8_vs_svhn_ft",
        "merged_7_vs_svhn_ft",
    ]

    for pair in focus_pairs:
        values = [results[l]["cosine_similarity"].get(pair, {}).get("mean", None) for l in block_layers]
        stds = [results[l]["cosine_similarity"].get(pair, {}).get("std", 0) for l in block_layers]
        if values[0] is None:
            continue
        color = PAIR_COLORS.get(pair, "gray")
        label = PAIR_LABELS.get(pair, pair)
        ax.plot(block_indices, values, "o-", color=color, label=label, linewidth=2, markersize=5)
        ax.fill_between(
            block_indices,
            [v - s for v, s in zip(values, stds)],
            [v + s for v, s in zip(values, stds)],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Transformer Block", fontsize=11)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=11)
    ax.set_title(f"Per-Sample Representation Similarity ({probe_dataset} data)", fontsize=12)
    ax.set_xticks(block_indices)
    ax.set_xticklabels([l.replace("block_", "") for l in block_layers])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"cosine_sim_curves_{probe_dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_activation_norms(results, layer_names, probe_dataset, output_dir):
    """Plot activation norm evolution across layers for each model."""
    block_layers = [l for l in layer_names if l.startswith("block_")]
    block_indices = list(range(len(block_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))

    model_colors = {
        "pretrained": "gray",
        "svhn_ft": "tab:brown",
        "merged_8": "tab:red",
        "merged_7": "tab:green",
    }

    model_labels = {
        "pretrained": "Pretrained",
        "svhn_ft": "SVHN Fine-tuned",
        "merged_8": "Merged All-8",
        "merged_7": "Merged No-EuroSAT",
    }

    models = list(results[block_layers[0]]["activation_norms"].keys())
    for model_name in models:
        means = [results[l]["activation_norms"][model_name]["mean"] for l in block_layers]
        stds = [results[l]["activation_norms"][model_name]["std"] for l in block_layers]
        color = model_colors.get(model_name, "gray")
        label = model_labels.get(model_name, model_name)
        ax.plot(block_indices, means, "o-", color=color, label=label, linewidth=2, markersize=5)
        ax.fill_between(
            block_indices,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Transformer Block", fontsize=11)
    ax.set_ylabel("Mean CLS Token L2 Norm", fontsize=11)
    ax.set_title(f"Activation Norm Evolution ({probe_dataset} data)", fontsize=12)
    ax.set_xticks(block_indices)
    ax.set_xticklabels([l.replace("block_", "") for l in block_layers])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"activation_norms_{probe_dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_l2_shift(results, layer_names, probe_dataset, output_dir):
    """Plot the relative L2 distance (representation shift) across layers."""
    block_layers = [l for l in layer_names if l.startswith("block_")]
    block_indices = list(range(len(block_layers)))

    fig, ax = plt.subplots(figsize=(10, 5))

    focus_pairs = [
        "merged_8_vs_merged_7",
        "merged_8_vs_svhn_ft",
        "merged_7_vs_svhn_ft",
    ]

    for pair in focus_pairs:
        values = [results[l]["l2_distances"].get(pair, {}).get("relative_mean", None) for l in block_layers]
        if values[0] is None:
            continue
        color = PAIR_COLORS.get(pair, "gray")
        label = PAIR_LABELS.get(pair, pair)
        ax.plot(block_indices, values, "o-", color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel("Transformer Block", fontsize=11)
    ax.set_ylabel("Relative L2 Shift (||diff|| / ||ref||)", fontsize=11)
    ax.set_title(f"Representation Shift by Layer ({probe_dataset} data)", fontsize=12)
    ax.set_xticks(block_indices)
    ax.set_xticklabels([l.replace("block_", "") for l in block_layers])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"l2_shift_{probe_dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_cosine_histogram(all_activations, model_names, layer_name, probe_dataset, output_dir):
    """Plot histogram of per-sample cosine similarities at a specific layer."""
    reps = {m: all_activations[m][layer_name] for m in model_names}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    pairs = [
        ("merged_8", "merged_7", "All8 vs No-EuroSAT"),
        ("merged_8", "svhn_ft", "All8 vs SVHN-FT"),
        ("merged_7", "svhn_ft", "No-EuroSAT vs SVHN-FT"),
    ]

    for ax, (m1, m2, title) in zip(axes, pairs):
        _, _, cos_vals = pairwise_cosine_sim(reps[m1], reps[m2])
        ax.hist(cos_vals, bins=50, alpha=0.8, color=PAIR_COLORS.get(f"{m1}_vs_{m2}", "gray"), edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\n(mean={cos_vals.mean():.4f})", fontsize=10)
        ax.axvline(cos_vals.mean(), color="red", linestyle="--", linewidth=1.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Per-Sample Cosine Similarity at {layer_name} ({probe_dataset} data)", fontsize=12)
    plt.tight_layout()
    path = output_dir / f"cosine_hist_{layer_name}_{probe_dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_eurosat_vs_svhn_comparison(svhn_results, eurosat_results, layer_names, output_dir):
    """Side-by-side comparison: how EuroSAT removal affects SVHN vs EuroSAT data."""
    block_layers = [l for l in layer_names if l.startswith("block_")]
    block_indices = list(range(len(block_layers)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pair = "merged_8_vs_merged_7"

    # CKA
    svhn_cka = [svhn_results[l]["cka"].get(pair, 0) for l in block_layers]
    euro_cka = [eurosat_results[l]["cka"].get(pair, 0) for l in block_layers]
    axes[0].plot(block_indices, svhn_cka, "o-", color="tab:red", label="SVHN data", linewidth=2)
    axes[0].plot(block_indices, euro_cka, "o-", color="tab:blue", label="EuroSAT data", linewidth=2)
    axes[0].set_ylabel("CKA (All8 vs No-EuroSAT)")
    axes[0].set_title("CKA: How much does removing EuroSAT change representations?")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Cosine similarity
    svhn_cos = [svhn_results[l]["cosine_similarity"].get(pair, {}).get("mean", 0) for l in block_layers]
    euro_cos = [eurosat_results[l]["cosine_similarity"].get(pair, {}).get("mean", 0) for l in block_layers]
    axes[1].plot(block_indices, svhn_cos, "o-", color="tab:red", label="SVHN data", linewidth=2)
    axes[1].plot(block_indices, euro_cos, "o-", color="tab:blue", label="EuroSAT data", linewidth=2)
    axes[1].set_ylabel("Mean Cosine Sim")
    axes[1].set_title("Cosine Similarity: Representation alignment")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Relative L2 shift
    svhn_l2 = [svhn_results[l]["l2_distances"].get(pair, {}).get("relative_mean", 0) for l in block_layers]
    euro_l2 = [eurosat_results[l]["l2_distances"].get(pair, {}).get("relative_mean", 0) for l in block_layers]
    axes[2].plot(block_indices, svhn_l2, "o-", color="tab:red", label="SVHN data", linewidth=2)
    axes[2].plot(block_indices, euro_l2, "o-", color="tab:blue", label="EuroSAT data", linewidth=2)
    axes[2].set_ylabel("Relative L2 Shift")
    axes[2].set_title("L2 Shift: Magnitude of representation change")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Transformer Block")
        ax.set_xticks(block_indices)
        ax.set_xticklabels([l.replace("block_", "") for l in block_layers])

    fig.suptitle("Effect of Removing EuroSAT: SVHN Data vs EuroSAT Data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "eurosat_vs_svhn_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def make_serializable(obj):
    """Recursively convert numpy/torch types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Activation space analysis for model merging interference")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer samples, fewer layers")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help=f"Number of samples per dataset (default: {N_SAMPLES})")
    parser.add_argument("--skip-eurosat-data", action="store_true", help="Skip EuroSAT data probing (faster)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_samples = 100 if args.quick else args.n_samples

    # ======================================================================= #
    # Phase 1: Create / load all models
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 1: Creating models")
    logger.info("=" * 70)

    # Pretrained encoder
    logger.info("Loading pretrained encoder...")
    pretrained = create_encoder(MODEL_NAME)
    print_memory("after pretrained")

    # Get preprocessing transform (same for all models since all are ViT-B-32)
    preprocess = pretrained.val_preprocess

    # SVHN fine-tuned encoder
    logger.info("Loading SVHN fine-tuned encoder...")
    svhn_ft = load_finetuned_encoder("SVHN", MODEL_NAME)
    print_memory("after svhn_ft")

    # Merged all-8
    merged_8 = create_merged_model(N8_DATASETS, MODEL_NAME)
    print_memory("after merged_8")

    # Merged no-EuroSAT (7 tasks)
    no_eurosat_datasets = [d for d in N8_DATASETS if d != "EuroSAT"]
    merged_7 = create_merged_model(no_eurosat_datasets, MODEL_NAME)
    print_memory("after merged_7")

    models = {
        "pretrained": pretrained,
        "svhn_ft": svhn_ft,
        "merged_8": merged_8,
        "merged_7": merged_7,
    }
    model_names = list(models.keys())

    # ======================================================================= #
    # Phase 2: Load probe data
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 2: Loading probe data")
    logger.info("=" * 70)

    svhn_images, svhn_labels = load_dataset_samples("SVHN", preprocess, n_samples)
    if not args.skip_eurosat_data:
        eurosat_images, eurosat_labels = load_dataset_samples("EuroSAT", preprocess, n_samples)

    # ======================================================================= #
    # Phase 3: Extract activations
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting activations")
    logger.info("=" * 70)

    # Process each model one at a time to save GPU memory
    svhn_activations = {}
    eurosat_activations = {}

    for model_name, model in models.items():
        logger.info(f"Extracting activations for: {model_name}")
        svhn_activations[model_name] = extract_all_activations(model, svhn_images, device)
        if not args.skip_eurosat_data:
            eurosat_activations[model_name] = extract_all_activations(model, eurosat_images, device)
        print_memory(f"after extracting {model_name}")

    # Determine layer names from the first model's activations
    layer_names = sorted(svhn_activations[model_names[0]].keys(), key=lambda x: (
        0 if x.startswith("block_") else 1 if x == "ln_post" else 2,
        int(x.split("_")[1]) if x.startswith("block_") else 999,
    ))
    logger.info(f"Layer names: {layer_names}")

    # ======================================================================= #
    # Phase 4: Compare activations
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 4: Computing comparison metrics")
    logger.info("=" * 70)

    svhn_results = compare_models(svhn_activations, model_names, layer_names, "SVHN")
    if not args.skip_eurosat_data:
        eurosat_results = compare_models(eurosat_activations, model_names, layer_names, "EuroSAT")

    # ======================================================================= #
    # Phase 5: Print summary
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)

    # Focus: Where does EuroSAT's effect concentrate?
    block_layers = [l for l in layer_names if l.startswith("block_")]
    key_pair = "merged_8_vs_merged_7"

    logger.info("\nEffect of removing EuroSAT on SVHN representations (All8 vs No-EuroSAT):")
    logger.info(f"{'Layer':<12} {'CKA':>8} {'Cos Sim':>10} {'Rel L2':>10}")
    logger.info("-" * 45)
    for layer in block_layers:
        r = svhn_results[layer]
        cka_val = r["cka"].get(key_pair, 0)
        cos_val = r["cosine_similarity"].get(key_pair, {}).get("mean", 0)
        l2_val = r["l2_distances"].get(key_pair, {}).get("relative_mean", 0)
        logger.info(f"{layer:<12} {cka_val:>8.4f} {cos_val:>10.4f} {l2_val:>10.4f}")

    # Where does removing EuroSAT bring the merged model CLOSER to SVHN-FT?
    logger.info("\nDoes removing EuroSAT bring merged model closer to SVHN-FT?")
    logger.info(f"{'Layer':<12} {'CKA(All8,FT)':>14} {'CKA(No-E,FT)':>14} {'Delta':>8}")
    logger.info("-" * 55)
    for layer in block_layers:
        cka_8_ft = svhn_results[layer]["cka"].get("merged_8_vs_svhn_ft", 0)
        cka_7_ft = svhn_results[layer]["cka"].get("merged_7_vs_svhn_ft", 0)
        delta = cka_7_ft - cka_8_ft
        marker = " ***" if delta > 0.01 else ""
        logger.info(f"{layer:<12} {cka_8_ft:>14.4f} {cka_7_ft:>14.4f} {delta:>+8.4f}{marker}")

    if not args.skip_eurosat_data:
        logger.info("\nAsymmetry: EuroSAT removal effect on SVHN data vs EuroSAT data:")
        logger.info(f"{'Layer':<12} {'SVHN CKA':>10} {'EuroSAT CKA':>12} {'Asym':>8}")
        logger.info("-" * 45)
        for layer in block_layers:
            s_cka = svhn_results[layer]["cka"].get(key_pair, 0)
            e_cka = eurosat_results[layer]["cka"].get(key_pair, 0)
            asym = s_cka - e_cka
            logger.info(f"{layer:<12} {s_cka:>10.4f} {e_cka:>12.4f} {asym:>+8.4f}")

    # ======================================================================= #
    # Phase 6: Generate plots
    # ======================================================================= #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Generating plots")
    logger.info("=" * 70)

    plot_cka_curves(svhn_results, layer_names, "SVHN", OUTPUT_DIR)
    plot_cosine_sim_curves(svhn_results, layer_names, "SVHN", OUTPUT_DIR)
    plot_activation_norms(svhn_results, layer_names, "SVHN", OUTPUT_DIR)
    plot_l2_shift(svhn_results, layer_names, "SVHN", OUTPUT_DIR)

    # Histogram at last block (where effects likely strongest)
    last_block = block_layers[-1]
    plot_cosine_histogram(svhn_activations, model_names, last_block, "SVHN", OUTPUT_DIR)

    if not args.skip_eurosat_data:
        plot_cka_curves(eurosat_results, layer_names, "EuroSAT", OUTPUT_DIR)
        plot_cosine_sim_curves(eurosat_results, layer_names, "EuroSAT", OUTPUT_DIR)
        plot_activation_norms(eurosat_results, layer_names, "EuroSAT", OUTPUT_DIR)
        plot_l2_shift(eurosat_results, layer_names, "EuroSAT", OUTPUT_DIR)
        plot_cosine_histogram(eurosat_activations, model_names, last_block, "EuroSAT", OUTPUT_DIR)
        plot_eurosat_vs_svhn_comparison(svhn_results, eurosat_results, layer_names, OUTPUT_DIR)

    # ======================================================================= #
    # Phase 7: Save results
    # ======================================================================= #
    all_results = {
        "metadata": {
            "model": MODEL_NAME,
            "n_samples": n_samples,
            "probe_datasets": ["SVHN"] + (["EuroSAT"] if not args.skip_eurosat_data else []),
            "merge_tasks_all8": N8_DATASETS,
            "merge_tasks_no_eurosat": no_eurosat_datasets,
            "merger": "InterferenceAwareMerger(mp_edge=True, isotropic=True)",
            "layer_names": layer_names,
        },
        "svhn_results": svhn_results,
    }
    if not args.skip_eurosat_data:
        all_results["eurosat_results"] = eurosat_results

    json_path = OUTPUT_DIR / "activation_analysis.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to: {json_path}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info("Activation analysis complete.")


if __name__ == "__main__":
    main()
