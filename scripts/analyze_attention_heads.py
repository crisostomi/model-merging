"""
Per-Head Attention Routing Analysis in Block 11

Block 11's attention mechanism creates ~99% of task-specific features in the
merged model (+39.4 out of +39.7 probe accuracy points; MLP adds almost nothing).
This script investigates whether different attention heads specialize for
different tasks.

For each of the 12 attention heads in block 11, we compute:
  1. CLS attention entropy — how spread is the head's attention? (per task)
  2. CLS self-attention weight — how much does CLS attend to itself? (per task)
  3. Task selectivity — pairwise cosine similarity of mean attention patterns
     across tasks.  Low similarity = task-selective head.
  4. Per-head output contribution norm — L2 norm of each head's CLS output
     vector (before out_proj), measuring how much the head contributes to the
     final representation.

Models analysed:
  - SVHN fine-tuned
  - Merged all-8 (MP-edge + isotropic)

Data: 200 test samples from each of the 8 N8 tasks (configurable).

Outputs (results/activation_analysis/):
  - attention_heads_analysis.json
  - attention_entropy_heatmap.png
  - head_output_contribution.png
  - task_selectivity_per_head.png

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_attention_heads.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_attention_heads.py --quick
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
MODEL_NAME = "ViT-B-32"
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "activation_analysis"
BATCH_SIZE = 64
BLOCK_IDX = 11
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN_DIM = 768  # NUM_HEADS * HEAD_DIM

# HuggingFace dataset specs: (hf_name, config_name, image_column, label_column)
DATASET_HF_MAP = {
    "SUN397":   ("tanganke/sun397",         None,             "image", "label"),
    "Cars":     ("tanganke/stanford_cars",   None,             "image", "label"),
    "RESISC45": ("tanganke/resisc45",        None,             "image", "label"),
    "EuroSAT":  ("tanganke/eurosat",         None,             "image", "label"),
    "SVHN":     ("ufldl-stanford/svhn",      "cropped_digits", "image", "label"),
    "GTSRB":    ("tanganke/gtsrb",           None,             "image", "label"),
    "MNIST":    ("ylecun/mnist",             None,             "image", "label"),
    "DTD":      ("tanganke/dtd",             None,             "image", "label"),
}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def create_encoder(model_name=MODEL_NAME):
    return ImageEncoder(model_name)


def load_finetuned_encoder(dataset_name, model_name=MODEL_NAME):
    encoder = create_encoder(model_name)
    ft_sd = load_state_dict_from_hf_cache(model_name, dataset_name)
    encoder.load_state_dict(ft_sd, strict=False)
    return encoder


def create_merged_model(datasets, model_name=MODEL_NAME):
    logger.info(f"Creating merged model with {len(datasets)} tasks: {datasets}")
    encoder = create_encoder(model_name)
    finetuned = {}
    for ds in datasets:
        logger.info(f"  Loading finetuned: {ds}")
        finetuned[ds] = load_state_dict_from_hf_cache(model_name, ds)
    merger = InterferenceAwareMerger(
        use_mp_edge=True, use_isotropic=True, mp_min_rank=4, mp_max_rank=128
    )
    merged = merger.merge(encoder, finetuned)
    return merged


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset_samples(dataset_name, preprocess, n_samples=200):
    """Load n_samples from a dataset's test split via HuggingFace."""
    from datasets import load_dataset as hf_load_dataset

    hf_name, config, img_col, lbl_col = DATASET_HF_MAP[dataset_name]
    logger.info(f"Loading {dataset_name} ({hf_name}, config={config}, n={n_samples})...")

    if config:
        ds = hf_load_dataset(hf_name, config, split="test")
    else:
        ds = hf_load_dataset(hf_name, split="test")

    images, labels = [], []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        images.append(preprocess(sample[img_col].convert("RGB")))
        labels.append(sample[lbl_col])

    logger.info(f"  Loaded {len(images)} samples, {len(set(labels))} classes")
    return torch.stack(images), np.array(labels)


# --------------------------------------------------------------------------- #
# Per-head attention extraction (manual QKV computation)
# --------------------------------------------------------------------------- #

def register_block10_hook(model):
    """Hook block 10 output to get block 11 input."""
    storage = {}
    visual = model.model.visual

    def hook(module, input, output):
        storage["block_10_out"] = output.detach()  # (seq, batch, hidden) — keep on device

    handle = visual.transformer.resblocks[10].register_forward_hook(hook)
    return storage, handle


@torch.no_grad()
def extract_per_head_attention(model, images, device, batch_size=BATCH_SIZE):
    """Extract per-head attention weights and output contributions for block 11.

    For every image we compute:
      - attn_weights: (batch, num_heads, seq_len, seq_len) — softmax(QK^T / sqrt(d))
      - head_outputs: (batch, num_heads, seq_len, head_dim)  — attn_weights @ V

    We aggregate across the dataset and return CLS-centric summaries:
      - cls_attn_weights: (n_samples, num_heads, seq_len) — A[:, :, 0, :]
      - cls_head_outputs: (n_samples, num_heads, head_dim)  — head_outputs[:, :, 0, :]
    """
    model.eval()
    model.to(device)

    block11 = model.model.visual.transformer.resblocks[BLOCK_IDX]

    # Get in_proj weight/bias for manual QKV computation
    W_qkv = block11.attn.in_proj_weight  # (3*hidden, hidden)
    b_qkv = block11.attn.in_proj_bias    # (3*hidden,)

    storage, handle = register_block10_hook(model)

    all_cls_attn = []
    all_cls_head_out = []

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        bsz = batch.shape[0]

        # Forward pass to populate hook
        _ = model(batch)

        # block 10 output = block 11 input: (seq_len, batch, hidden)
        x = storage["block_10_out"]
        seq_len = x.shape[0]

        # Apply layer norm (ln_1) before attention
        x_norm = block11.ln_1(x)  # (seq_len, batch, hidden)

        # Manual QKV projection
        qkv = F.linear(x_norm, W_qkv, b_qkv)  # (seq_len, batch, 3*hidden)
        q, k, v = qkv.chunk(3, dim=-1)  # each (seq_len, batch, hidden)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        q = q.permute(1, 0, 2).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.permute(1, 0, 2).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.permute(1, 0, 2).reshape(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        # Compute attention weights: (batch, heads, seq_len, seq_len)
        scale = math.sqrt(HEAD_DIM)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Per-head output: (batch, heads, seq_len, head_dim)
        head_outputs = torch.matmul(attn_weights, v)

        # Extract CLS token (index 0 in seq dim)
        cls_attn = attn_weights[:, :, 0, :]       # (batch, heads, seq_len)
        cls_head_out = head_outputs[:, :, 0, :]    # (batch, heads, head_dim)

        all_cls_attn.append(cls_attn.cpu())
        all_cls_head_out.append(cls_head_out.cpu())

        storage.clear()

    handle.remove()
    model.cpu()
    torch.cuda.empty_cache()

    return {
        "cls_attn_weights": torch.cat(all_cls_attn, dim=0),       # (n, heads, seq_len)
        "cls_head_outputs": torch.cat(all_cls_head_out, dim=0),    # (n, heads, head_dim)
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_attention_entropy(cls_attn_weights):
    """Compute entropy of CLS attention distribution for each head.

    Args:
        cls_attn_weights: (n_samples, num_heads, seq_len) — attention probs

    Returns:
        (num_heads,) mean entropy per head
    """
    # Clamp to avoid log(0)
    p = cls_attn_weights.clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=-1)  # (n_samples, num_heads)
    return entropy.mean(dim=0).numpy()  # (num_heads,)


def compute_cls_self_attention(cls_attn_weights):
    """CLS self-attention: attention weight A[cls, cls] = A[:, :, 0].

    Args:
        cls_attn_weights: (n_samples, num_heads, seq_len)

    Returns:
        (num_heads,) mean CLS self-attention weight per head
    """
    return cls_attn_weights[:, :, 0].mean(dim=0).numpy()  # (num_heads,)


def compute_head_output_norms(cls_head_outputs):
    """L2 norm of each head's CLS output vector.

    Args:
        cls_head_outputs: (n_samples, num_heads, head_dim)

    Returns:
        (num_heads,) mean L2 norm per head
    """
    norms = cls_head_outputs.norm(dim=-1)  # (n_samples, num_heads)
    return norms.mean(dim=0).numpy()  # (num_heads,)


def compute_mean_attention_pattern(cls_attn_weights):
    """Mean CLS attention pattern per head (averaged over samples).

    Args:
        cls_attn_weights: (n_samples, num_heads, seq_len)

    Returns:
        (num_heads, seq_len) mean attention pattern
    """
    return cls_attn_weights.mean(dim=0)  # (num_heads, seq_len)


def compute_task_selectivity(mean_patterns_by_task):
    """Compute task selectivity for each head.

    For each head, compute pairwise cosine similarity of mean attention
    patterns across tasks.  Task selectivity = 1 - mean pairwise cosine.
    High selectivity means the head attends to different positions for
    different tasks.

    Args:
        mean_patterns_by_task: dict {task_name: (num_heads, seq_len) tensor}

    Returns:
        (num_heads,) selectivity scores
    """
    task_names = list(mean_patterns_by_task.keys())
    n_tasks = len(task_names)
    patterns = torch.stack([mean_patterns_by_task[t] for t in task_names])  # (n_tasks, heads, seq)

    selectivity = np.zeros(NUM_HEADS)
    for h in range(NUM_HEADS):
        # (n_tasks, seq_len) patterns for this head
        h_patterns = patterns[:, h, :]
        # Normalize
        h_norm = h_patterns / (h_patterns.norm(dim=1, keepdim=True) + 1e-10)
        # Pairwise cosine: (n_tasks, n_tasks)
        cos_matrix = h_norm @ h_norm.T

        # Average off-diagonal
        total = 0.0
        count = 0
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                total += cos_matrix[i, j].item()
                count += 1
        mean_cos = total / max(count, 1)
        selectivity[h] = 1.0 - mean_cos

    return selectivity


def compute_pairwise_task_cosine(mean_patterns_by_task):
    """Pairwise cosine similarity between tasks for each head.

    Returns: dict of {head_idx: {(task_a, task_b): cosine_sim}}
    """
    task_names = list(mean_patterns_by_task.keys())
    patterns = torch.stack([mean_patterns_by_task[t] for t in task_names])  # (n_tasks, heads, seq)

    result = {}
    for h in range(NUM_HEADS):
        h_patterns = patterns[:, h, :]
        h_norm = h_patterns / (h_patterns.norm(dim=1, keepdim=True) + 1e-10)
        cos_matrix = h_norm @ h_norm.T

        head_result = {}
        for i, t1 in enumerate(task_names):
            for j, t2 in enumerate(task_names):
                if i < j:
                    head_result[f"{t1}_vs_{t2}"] = float(cos_matrix[i, j])
        result[h] = head_result

    return result


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

TASK_COLORS = {
    "SUN397": "#1f77b4",
    "Cars": "#ff7f0e",
    "RESISC45": "#2ca02c",
    "EuroSAT": "#d62728",
    "SVHN": "#9467bd",
    "GTSRB": "#8c564b",
    "MNIST": "#e377c2",
    "DTD": "#7f7f7f",
}


def plot_entropy_heatmap(entropy_by_task, model_name, output_dir):
    """Heatmap: per-head attention entropy across tasks (heads x tasks)."""
    tasks = list(entropy_by_task.keys())
    n_tasks = len(tasks)

    # Build matrix: (num_heads, n_tasks)
    matrix = np.zeros((NUM_HEADS, n_tasks))
    for j, task in enumerate(tasks):
        matrix[:, j] = entropy_by_task[task]

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.9), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean CLS Attention Entropy (nats)", fontsize=10)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(NUM_HEADS))
    ax.set_yticklabels([f"Head {i}" for i in range(NUM_HEADS)], fontsize=9)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Attention Head", fontsize=11)
    ax.set_title(f"Block 11 Per-Head CLS Attention Entropy — {model_name}", fontsize=12)

    # Annotate cells
    for i in range(NUM_HEADS):
        for j in range(n_tasks):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    plt.tight_layout()
    path = output_dir / f"attention_entropy_heatmap_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_head_output_contribution(norms_by_task, model_name, output_dir,
                                  highlight_tasks=("SVHN", "EuroSAT")):
    """Bar chart: per-head output contribution norm for highlighted tasks."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(NUM_HEADS)
    n_tasks = len(highlight_tasks)
    width = 0.8 / n_tasks

    for i, task in enumerate(highlight_tasks):
        if task not in norms_by_task:
            continue
        norms = norms_by_task[task]
        offset = (i - n_tasks / 2 + 0.5) * width
        color = TASK_COLORS.get(task, f"C{i}")
        bars = ax.bar(x + offset, norms, width, label=task, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {i}" for i in range(NUM_HEADS)], fontsize=9)
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Mean L2 Norm of CLS Head Output", fontsize=11)
    ax.set_title(f"Block 11 Per-Head Output Contribution — {model_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"head_output_contribution_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_head_output_contribution_all_tasks(norms_by_task, model_name, output_dir):
    """Grouped bar chart: per-head output contribution for ALL tasks."""
    tasks = list(norms_by_task.keys())
    n_tasks = len(tasks)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(NUM_HEADS)
    width = 0.8 / n_tasks

    for i, task in enumerate(tasks):
        norms = norms_by_task[task]
        offset = (i - n_tasks / 2 + 0.5) * width
        color = TASK_COLORS.get(task, f"C{i}")
        ax.bar(x + offset, norms, width, label=task, color=color,
               alpha=0.85, edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {i}" for i in range(NUM_HEADS)], fontsize=9)
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Mean L2 Norm of CLS Head Output", fontsize=11)
    ax.set_title(f"Block 11 Per-Head Output Contribution (All 8 Tasks) — {model_name}", fontsize=12)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"head_output_contribution_all_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_task_selectivity(selectivity_by_model, output_dir):
    """Bar chart: task selectivity score per head for each model."""
    models = list(selectivity_by_model.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(NUM_HEADS)
    width = 0.8 / n_models

    model_colors = {
        "svhn_ft": "#8B4513",
        "merged_8": "tab:red",
    }
    model_labels = {
        "svhn_ft": "SVHN Fine-tuned",
        "merged_8": "Merged All-8",
    }

    for i, model_name in enumerate(models):
        scores = selectivity_by_model[model_name]
        offset = (i - n_models / 2 + 0.5) * width
        color = model_colors.get(model_name, f"C{i}")
        label = model_labels.get(model_name, model_name)
        bars = ax.bar(x + offset, scores, width, label=label, color=color,
                      alpha=0.85, edgecolor="black", linewidth=0.5)
        # Annotate
        for bar, val in zip(bars, scores):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6,
                        fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Head {i}" for i in range(NUM_HEADS)], fontsize=9)
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Task Selectivity (1 - mean pairwise cos)", fontsize=11)
    ax.set_title("Block 11 Per-Head Task Selectivity", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(s) for s in selectivity_by_model.values()) * 1.2 + 0.01)

    plt.tight_layout()
    path = output_dir / "task_selectivity_per_head.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


def plot_cls_self_attention_heatmap(self_attn_by_task, model_name, output_dir):
    """Heatmap: CLS self-attention weight per head per task."""
    tasks = list(self_attn_by_task.keys())
    n_tasks = len(tasks)

    matrix = np.zeros((NUM_HEADS, n_tasks))
    for j, task in enumerate(tasks):
        matrix[:, j] = self_attn_by_task[task]

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.9), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("CLS Self-Attention Weight", fontsize=10)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(NUM_HEADS))
    ax.set_yticklabels([f"Head {i}" for i in range(NUM_HEADS)], fontsize=9)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Attention Head", fontsize=11)
    ax.set_title(f"Block 11 CLS Self-Attention — {model_name}", fontsize=12)

    for i in range(NUM_HEADS):
        for j in range(n_tasks):
            val = matrix[i, j]
            text_color = "white" if val > (matrix.max() + matrix.min()) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    plt.tight_layout()
    path = output_dir / f"cls_self_attention_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")


# --------------------------------------------------------------------------- #
# JSON serialization helper
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
    parser = argparse.ArgumentParser(
        description="Per-head attention routing analysis in block 11"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 50 samples per task for faster iteration",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of test samples per task (default: 200)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_samples = 50 if args.quick else args.n_samples

    # =================================================================== #
    # Phase 1: Load models
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 1: Loading models")
    logger.info("=" * 70)

    svhn_ft = load_finetuned_encoder("SVHN", MODEL_NAME)
    preprocess = svhn_ft.val_preprocess
    print_memory("after svhn_ft")

    merged_8 = create_merged_model(N8_DATASETS, MODEL_NAME)
    print_memory("after merged_8")

    models = {
        "svhn_ft": svhn_ft,
        "merged_8": merged_8,
    }

    # =================================================================== #
    # Phase 2: Load test data for all 8 N8 tasks
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 2: Loading test data for all 8 tasks")
    logger.info("=" * 70)

    task_data = {}
    for dataset_name in N8_DATASETS:
        images, labels = load_dataset_samples(dataset_name, preprocess, n_samples)
        task_data[dataset_name] = {"images": images, "labels": labels}
        logger.info(f"  {dataset_name}: {len(images)} images, {len(set(labels.tolist()))} classes")
    print_memory("after loading all data")

    # =================================================================== #
    # Phase 3: Extract per-head attention for each model x task
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting per-head attention patterns")
    logger.info("=" * 70)

    # Structure: {model_name: {task_name: {cls_attn_weights, cls_head_outputs}}}
    all_head_data = {}

    for model_name, model in models.items():
        logger.info(f"\n--- Model: {model_name} ---")
        all_head_data[model_name] = {}

        for task_name in N8_DATASETS:
            logger.info(f"  Extracting attention for {task_name}...")
            head_data = extract_per_head_attention(
                model, task_data[task_name]["images"], device, batch_size=BATCH_SIZE
            )
            all_head_data[model_name][task_name] = head_data
            logger.info(
                f"    cls_attn_weights: {head_data['cls_attn_weights'].shape}, "
                f"cls_head_outputs: {head_data['cls_head_outputs'].shape}"
            )

        print_memory(f"after {model_name}")

    # =================================================================== #
    # Phase 4: Compute per-head metrics
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 4: Computing per-head metrics")
    logger.info("=" * 70)

    results = {}

    for model_name in models:
        logger.info(f"\n--- Model: {model_name} ---")
        model_results = {
            "entropy_by_task": {},
            "self_attention_by_task": {},
            "head_norms_by_task": {},
            "mean_attention_patterns": {},
        }

        mean_patterns_for_selectivity = {}

        for task_name in N8_DATASETS:
            hd = all_head_data[model_name][task_name]

            # Attention entropy per head
            entropy = compute_attention_entropy(hd["cls_attn_weights"])
            model_results["entropy_by_task"][task_name] = entropy
            logger.info(f"  {task_name} entropy: min={entropy.min():.3f}, max={entropy.max():.3f}, mean={entropy.mean():.3f}")

            # CLS self-attention per head
            self_attn = compute_cls_self_attention(hd["cls_attn_weights"])
            model_results["self_attention_by_task"][task_name] = self_attn

            # Head output norms
            norms = compute_head_output_norms(hd["cls_head_outputs"])
            model_results["head_norms_by_task"][task_name] = norms
            logger.info(f"  {task_name} head norms: min={norms.min():.3f}, max={norms.max():.3f}")

            # Mean attention pattern (for selectivity)
            mean_pat = compute_mean_attention_pattern(hd["cls_attn_weights"])
            model_results["mean_attention_patterns"][task_name] = mean_pat.numpy()
            mean_patterns_for_selectivity[task_name] = mean_pat

        # Task selectivity per head
        selectivity = compute_task_selectivity(mean_patterns_for_selectivity)
        model_results["task_selectivity"] = selectivity
        logger.info(f"\n  Task selectivity: {selectivity}")

        # Pairwise task cosine per head
        pairwise = compute_pairwise_task_cosine(mean_patterns_for_selectivity)
        model_results["pairwise_task_cosine"] = pairwise

        results[model_name] = model_results

    # =================================================================== #
    # Phase 5: Print summary
    # =================================================================== #
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for model_name in models:
        r = results[model_name]
        logger.info(f"\n{'='*40}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*40}")

        # Entropy summary
        logger.info("\nAttention entropy per head (mean across tasks):")
        mean_entropy = np.stack(list(r["entropy_by_task"].values()), axis=0).mean(axis=0)
        for h in range(NUM_HEADS):
            logger.info(f"  Head {h:2d}: {mean_entropy[h]:.3f}")

        # Head output norms summary
        logger.info("\nHead output norms (mean across tasks):")
        mean_norms = np.stack(list(r["head_norms_by_task"].values()), axis=0).mean(axis=0)
        for h in range(NUM_HEADS):
            logger.info(f"  Head {h:2d}: {mean_norms[h]:.3f}")

        # Task selectivity
        logger.info("\nTask selectivity per head:")
        sel = r["task_selectivity"]
        sorted_heads = np.argsort(sel)[::-1]
        for h in sorted_heads:
            marker = " <-- HIGH" if sel[h] > np.median(sel) * 1.5 else ""
            logger.info(f"  Head {h:2d}: {sel[h]:.4f}{marker}")

        # Most task-selective head: which tasks differ most?
        most_selective = sorted_heads[0]
        logger.info(f"\nMost task-selective head: Head {most_selective} (selectivity={sel[most_selective]:.4f})")
        logger.info(f"  Pairwise cosines for Head {most_selective}:")
        for pair, cos_val in sorted(r["pairwise_task_cosine"][most_selective].items(), key=lambda x: x[1]):
            logger.info(f"    {pair}: {cos_val:.4f}")

    # Comparison: merged vs fine-tuned
    logger.info("\n" + "=" * 70)
    logger.info("MERGED vs FINE-TUNED COMPARISON")
    logger.info("=" * 70)

    logger.info(f"\n{'Head':>6} {'Merged Select':>14} {'FT Select':>12} {'Delta':>8}")
    logger.info("-" * 45)
    for h in range(NUM_HEADS):
        m_sel = results["merged_8"]["task_selectivity"][h]
        ft_sel = results["svhn_ft"]["task_selectivity"][h]
        delta = m_sel - ft_sel
        logger.info(f"  {h:>4d} {m_sel:>14.4f} {ft_sel:>12.4f} {delta:>+8.4f}")

    logger.info("\nHead output norms for SVHN data:")
    logger.info(f"{'Head':>6} {'Merged':>10} {'FT':>10} {'Ratio':>10}")
    logger.info("-" * 40)
    for h in range(NUM_HEADS):
        m_norm = results["merged_8"]["head_norms_by_task"]["SVHN"][h]
        ft_norm = results["svhn_ft"]["head_norms_by_task"]["SVHN"][h]
        ratio = m_norm / (ft_norm + 1e-10)
        logger.info(f"  {h:>4d} {m_norm:>10.3f} {ft_norm:>10.3f} {ratio:>10.3f}")

    # =================================================================== #
    # Phase 6: Generate plots
    # =================================================================== #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Generating plots")
    logger.info("=" * 70)

    for model_name in models:
        r = results[model_name]

        # 1. Entropy heatmap
        plot_entropy_heatmap(r["entropy_by_task"], model_name, OUTPUT_DIR)

        # 2. CLS self-attention heatmap
        plot_cls_self_attention_heatmap(r["self_attention_by_task"], model_name, OUTPUT_DIR)

        # 3. Head output contribution (SVHN vs EuroSAT)
        plot_head_output_contribution(r["head_norms_by_task"], model_name, OUTPUT_DIR,
                                      highlight_tasks=("SVHN", "EuroSAT"))

        # 4. Head output contribution (all 8 tasks)
        plot_head_output_contribution_all_tasks(r["head_norms_by_task"], model_name, OUTPUT_DIR)

    # 5. Task selectivity comparison
    selectivity_by_model = {
        m: results[m]["task_selectivity"] for m in models
    }
    plot_task_selectivity(selectivity_by_model, OUTPUT_DIR)

    # =================================================================== #
    # Phase 7: Save results
    # =================================================================== #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: Saving results")
    logger.info("=" * 70)

    output = {
        "metadata": {
            "model": MODEL_NAME,
            "n_samples_per_task": n_samples,
            "block_analyzed": BLOCK_IDX,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "tasks": N8_DATASETS,
            "models_analyzed": list(models.keys()),
            "merger": "InterferenceAwareMerger(mp_edge=True, isotropic=True)",
        },
        "results": {},
    }

    for model_name in models:
        r = results[model_name]
        output["results"][model_name] = {
            "entropy_by_task": {t: v.tolist() for t, v in r["entropy_by_task"].items()},
            "self_attention_by_task": {t: v.tolist() for t, v in r["self_attention_by_task"].items()},
            "head_norms_by_task": {t: v.tolist() for t, v in r["head_norms_by_task"].items()},
            "task_selectivity": r["task_selectivity"].tolist(),
            "pairwise_task_cosine": make_serializable(r["pairwise_task_cosine"]),
        }

    json_path = OUTPUT_DIR / "attention_heads_analysis.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(output), f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info("Attention head analysis complete.")


if __name__ == "__main__":
    main()
