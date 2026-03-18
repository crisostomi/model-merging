#!/usr/bin/env python3
"""
Fisher-informed merging analysis for ViT-B-32 N8 benchmark.

Uses task vector magnitude as a proxy for diagonal Fisher information to
understand the zero-sum capacity problem in late transformer blocks (9-11).

Analyses:
  1. Per-task "Fisher" magnitude across layers (heatmap)
  2. Pairwise Fisher overlap matrices for early vs late blocks
  3. Comparison of Fisher overlap vs LOO interaction matrix
  4. Fisher-weighted merging prototype (conflict-aware weighting)

Inspired by CAMEx (ICLR 2025) and KFAC-TAK (2025).

Usage:
    uv run python scripts/analyze_fisher_merging.py          # full analysis
    uv run python scripts/analyze_fisher_merging.py --quick   # 3 tasks, subset of layers

Submit via:
    sbatch slurm/launch_analysis.slurm scripts/analyze_fisher_merging.py
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from scipy import stats

# --------------------------------------------------------------------------- #
# Project setup
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.merging.task_vectors import compute_task_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_NAME = "ViT-B-32"
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "fisher_analysis"

# LOO interaction data from our Session 3 experiments (baseline 93.11% norm acc).
# interaction_matrix[i][j] = delta normalized_acc for task i when task j is excluded.
# Positive = j was hurting i. Loaded from file if available, hardcoded as fallback.
LOO_INTERACTION_MATRIX = np.array([
    [np.nan, 0.001996, 0.004688, 0.006033, 0.008052, 0.008724, 0.002669, 0.008724],
    [0.007812, np.nan, -0.00933, -0.004655, 0.006253, 0.003137, 0.001579, 0.001579],
    [-3e-05, 0.001636, np.nan, 0.024954, -0.001695, 0.003302, 0.001636, 0.009964],
    [0.007512, 0.003781, 0.007512, np.nan, -0.003682, 0.011243, 4.9e-05, 0.011243],
    [0.015068, 0.002748, 0.015863, 0.03176, np.nan, -0.008778, -0.011162, 0.01785],
    [0.012778, 0.002374, 0.009577, 0.015179, -0.001627, np.nan, 0.004775, 0.007976],
    [0.004009, 0.001998, 0.008031, 0.000993, -0.007052, 0.012053, np.nan, 0.005015],
    [0.027186, -2.5e-05, 0.006778, 0.013581, -2.5e-05, 0.006778, -2.5e-05, np.nan],
])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_block_index(layer_name: str) -> int:
    """Extract the transformer block index from a layer name, or -1."""
    match = re.search(r"resblocks\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else -1


def classify_block_group(layer_name: str) -> str:
    """Classify layer into 'early' (blocks 0-8), 'late' (9-11), or 'other'."""
    idx = get_block_index(layer_name)
    if idx < 0:
        return "other"
    if idx <= 8:
        return "early"
    return "late"


def classify_layer_type(layer_name: str) -> str:
    """Classify a layer name into a functional category."""
    if "in_proj" in layer_name or "out_proj" in layer_name:
        return "attention"
    if "c_fc" in layer_name:
        return "mlp_fc"
    if "c_proj" in layer_name:
        return "mlp_proj"
    if "ln" in layer_name or "norm" in layer_name:
        return "layernorm"
    if "conv1" in layer_name:
        return "conv_embed"
    if "class_embedding" in layer_name or "positional_embedding" in layer_name:
        return "embedding"
    if "proj" in layer_name:
        return "projection"
    return "other"


def shorten_layer_name(layer_name: str) -> str:
    """Produce a compact label for plots."""
    # e.g. "visual.transformer.resblocks.11.attn.out_proj.weight" -> "blk11.out_proj"
    match = re.search(r"resblocks\.(\d+)\.(.+?)\.weight$", layer_name)
    if match:
        return f"blk{match.group(1)}.{match.group(2)}"
    parts = layer_name.replace(".weight", "").split(".")
    return ".".join(parts[-2:]) if len(parts) > 1 else layer_name


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name: str, dataset_name: str = "base"):
    """Load state dict from HuggingFace cache without creating a full model."""
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def load_task_vectors(datasets: list[str], model_name: str = MODEL_NAME):
    """Load pretrained model and compute task vectors for all datasets."""
    logger.info("Loading pretrained state dict: %s", model_name)
    pretrained = load_state_dict_from_hf_cache(model_name, "base")

    task_vectors = {}
    for ds in datasets:
        logger.info("Loading finetuned state dict for %s", ds)
        finetuned = load_state_dict_from_hf_cache(model_name, ds)
        tv = compute_task_vector(pretrained, finetuned, device="cpu")
        task_vectors[ds] = tv
        del finetuned
        logger.info(
            "  %s: %d layers, total params = %s",
            ds, len(tv), f"{sum(v.numel() for v in tv.values()):,}",
        )

    del pretrained
    return task_vectors


def load_loo_matrix() -> tuple[np.ndarray, list[str]]:
    """Load the LOO interaction matrix. Try JSON file first, fall back to hardcoded."""
    json_path = PROJECT_ROOT / "results" / "analysis" / "leave_one_out_results.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            matrix = np.array(data["interaction_matrix"], dtype=float)
            # Replace None with NaN
            matrix = np.where(matrix == None, np.nan, matrix)  # noqa: E711
            tasks = data["tasks"]
            logger.info("Loaded LOO matrix from %s", json_path)
            return matrix, tasks
        except Exception as e:
            logger.warning("Failed to load LOO JSON (%s), using hardcoded values", e)

    return LOO_INTERACTION_MATRIX.copy(), N8_DATASETS[:]


# --------------------------------------------------------------------------- #
# Part 1: Per-task Fisher magnitude across layers
# --------------------------------------------------------------------------- #

def compute_fisher_magnitudes(task_vectors: dict, layer_names_2d: list[str], datasets: list[str]):
    """
    Compute per-task, per-layer "Fisher magnitude" = mean squared task vector element.

    This is a proxy for the diagonal Fisher information: parameters with large
    task vector magnitude had large gradient updates during fine-tuning, which
    correlates with high curvature (Fisher) in the loss landscape.

    Returns:
        fisher_mag: dict[dataset][layer_name] -> float (mean |tau|^2)
        fisher_per_element: dict[dataset][layer_name] -> np.ndarray (|tau|^2 flattened)
    """
    logger.info("=" * 60)
    logger.info("PART 1: Computing per-task Fisher magnitudes")
    logger.info("=" * 60)

    fisher_mag = {ds: {} for ds in datasets}
    fisher_per_element = {ds: {} for ds in datasets}

    for layer_name in layer_names_2d:
        for ds in datasets:
            tv = task_vectors[ds][layer_name].float()
            sq = (tv ** 2).flatten().numpy()
            fisher_mag[ds][layer_name] = float(sq.mean())
            fisher_per_element[ds][layer_name] = sq

    # Print summary grouped by block
    logger.info("\nPer-block-group average Fisher magnitude:")
    for group in ["early", "late", "other"]:
        group_layers = [l for l in layer_names_2d if classify_block_group(l) == group]
        if not group_layers:
            continue
        for ds in datasets:
            avg = np.mean([fisher_mag[ds][l] for l in group_layers])
            logger.info("  %-8s  %s: %.6f", group, ds, avg)

    return fisher_mag, fisher_per_element


# --------------------------------------------------------------------------- #
# Part 2: Pairwise Fisher overlap
# --------------------------------------------------------------------------- #

def compute_fisher_overlap(
    fisher_per_element: dict,
    layer_names_2d: list[str],
    datasets: list[str],
):
    """
    Compute pairwise Fisher overlap between tasks.

    For each layer, the "Fisher overlap" between task i and task j is the
    Pearson correlation of their per-element squared task vector magnitudes.
    High overlap = both tasks modify the same parameters strongly = potential
    for interference.

    Returns:
        overlap_by_group: dict[group_name] -> 8x8 ndarray (avg over layers in group)
        overlap_per_layer: dict[layer_name] -> 8x8 ndarray
    """
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Pairwise Fisher overlap analysis")
    logger.info("=" * 60)

    n_tasks = len(datasets)
    overlap_per_layer = {}
    overlap_by_group = defaultdict(lambda: np.zeros((n_tasks, n_tasks)))
    group_counts = defaultdict(int)

    for layer_name in layer_names_2d:
        group = classify_block_group(layer_name)
        group_counts[group] += 1

        matrix = np.zeros((n_tasks, n_tasks))
        for i, ds_i in enumerate(datasets):
            for j, ds_j in enumerate(datasets):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                if i > j:
                    matrix[i, j] = matrix[j, i]
                    continue

                fi = fisher_per_element[ds_i][layer_name]
                fj = fisher_per_element[ds_j][layer_name]

                # Pearson correlation of Fisher magnitudes
                if fi.std() > 1e-12 and fj.std() > 1e-12:
                    corr = float(np.corrcoef(fi, fj)[0, 1])
                else:
                    corr = 0.0
                matrix[i, j] = corr
                matrix[j, i] = corr

        overlap_per_layer[layer_name] = matrix
        overlap_by_group[group] += matrix

    # Normalize by count
    for group in overlap_by_group:
        if group_counts[group] > 0:
            overlap_by_group[group] /= group_counts[group]

    # Print summary
    for group in ["early", "late", "other"]:
        if group not in overlap_by_group:
            continue
        mat = overlap_by_group[group]
        # Extract upper triangle (excluding diagonal)
        upper = mat[np.triu_indices(n_tasks, k=1)]
        logger.info(
            "  %s blocks: mean overlap=%.4f, std=%.4f, min=%.4f, max=%.4f",
            group, upper.mean(), upper.std(), upper.min(), upper.max(),
        )

    return dict(overlap_by_group), overlap_per_layer


# --------------------------------------------------------------------------- #
# Part 3: Fisher-weighted merging prototype
# --------------------------------------------------------------------------- #

def fisher_weighted_merge_analysis(
    task_vectors: dict,
    fisher_per_element: dict,
    layer_names_2d: list[str],
    datasets: list[str],
):
    """
    Prototype Fisher-weighted merging and compare with uniform averaging.

    For each parameter, weight each task's contribution inversely to the
    OTHER tasks' Fisher at that parameter. This preserves directions where a
    task has high Fisher but others have low Fisher (task-specific directions).

    We don't actually evaluate the merged model (no GPU), but we compute
    statistics about the weighting:
    - How non-uniform are the weights? (entropy, max/min ratio)
    - How different is the Fisher-weighted merge from uniform?
    - Early vs late block comparison

    Returns:
        stats: dict with per-layer and per-group statistics
    """
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: Fisher-weighted merging prototype")
    logger.info("=" * 60)

    n_tasks = len(datasets)
    stats_per_layer = {}
    group_stats = defaultdict(lambda: {
        "weight_entropy": [],
        "weight_max_ratio": [],
        "cosine_vs_uniform": [],
        "l2_vs_uniform": [],
    })

    for layer_name in layer_names_2d:
        group = classify_block_group(layer_name)
        n_params = fisher_per_element[datasets[0]][layer_name].shape[0]

        # Stack Fisher magnitudes: (n_tasks, n_params)
        fisher_stack = np.stack(
            [fisher_per_element[ds][layer_name] for ds in datasets], axis=0
        )

        # For each parameter, compute weights for each task:
        # w_i[j] = 1 / (eps + sum_{k != i} F_k[j])
        # Then normalize: w_i[j] /= sum_k w_k[j]
        eps = 1e-10
        weights = np.zeros((n_tasks, n_params))
        for i in range(n_tasks):
            # Sum of OTHER tasks' Fisher at each parameter
            others_fisher = fisher_stack.sum(axis=0) - fisher_stack[i]
            weights[i] = 1.0 / (eps + others_fisher)

        # Normalize weights per parameter (so they sum to 1 across tasks)
        weight_sums = weights.sum(axis=0, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        weights /= weight_sums  # (n_tasks, n_params)

        # Compute uniform-weighted merged TV
        tv_stack = np.stack(
            [task_vectors[ds][layer_name].float().flatten().numpy() for ds in datasets],
            axis=0,
        )  # (n_tasks, n_params)

        merged_uniform = tv_stack.mean(axis=0)
        merged_fisher = (weights * tv_stack).sum(axis=0)

        # Statistics about the weights
        # Per-parameter entropy of the weight distribution across tasks
        weight_entropy = -np.sum(
            np.where(weights > 1e-15, weights * np.log(weights + 1e-15), 0.0), axis=0
        )
        avg_entropy = float(weight_entropy.mean())
        max_entropy = float(np.log(n_tasks))  # uniform distribution entropy

        # Max weight ratio: how concentrated is the weight?
        max_weight = weights.max(axis=0)
        min_weight = weights.min(axis=0)
        avg_max_ratio = float(
            np.mean(max_weight / (min_weight + 1e-15))
        )
        # Clip for logging sanity
        avg_max_ratio = min(avg_max_ratio, 1e6)

        # Cosine similarity between uniform and Fisher-weighted merges
        norm_u = np.linalg.norm(merged_uniform)
        norm_f = np.linalg.norm(merged_fisher)
        if norm_u > 1e-12 and norm_f > 1e-12:
            cosine = float(np.dot(merged_uniform, merged_fisher) / (norm_u * norm_f))
        else:
            cosine = 0.0

        # Relative L2 difference
        l2_diff = float(np.linalg.norm(merged_fisher - merged_uniform))
        rel_l2 = l2_diff / (norm_u + 1e-12)

        layer_stats = {
            "avg_weight_entropy": avg_entropy,
            "entropy_ratio": avg_entropy / max_entropy if max_entropy > 0 else 0,
            "avg_max_weight_ratio": avg_max_ratio,
            "cosine_vs_uniform": cosine,
            "relative_l2_vs_uniform": rel_l2,
            "merged_uniform_norm": float(norm_u),
            "merged_fisher_norm": float(norm_f),
        }
        stats_per_layer[layer_name] = layer_stats

        group_stats[group]["weight_entropy"].append(avg_entropy)
        group_stats[group]["weight_max_ratio"].append(avg_max_ratio)
        group_stats[group]["cosine_vs_uniform"].append(cosine)
        group_stats[group]["l2_vs_uniform"].append(rel_l2)

    # Print summary
    logger.info("\nFisher-weighted merge statistics by block group:")
    logger.info(
        "%-8s  %12s  %12s  %12s  %12s",
        "Group", "Entropy Ratio", "Max W Ratio", "Cos(F,U)", "Rel L2",
    )
    logger.info("-" * 62)
    for group in ["early", "late", "other"]:
        gs = group_stats.get(group)
        if not gs or not gs["weight_entropy"]:
            continue
        max_ent = float(np.log(n_tasks))
        logger.info(
            "%-8s  %12.4f  %12.1f  %12.4f  %12.4f",
            group,
            np.mean(gs["weight_entropy"]) / max_ent,
            np.mean(gs["weight_max_ratio"]),
            np.mean(gs["cosine_vs_uniform"]),
            np.mean(gs["l2_vs_uniform"]),
        )

    return stats_per_layer, dict(group_stats)


# --------------------------------------------------------------------------- #
# Part 2b: Fisher overlap vs LOO interaction correlation
# --------------------------------------------------------------------------- #

def compare_fisher_vs_loo(
    overlap_by_group: dict,
    loo_matrix: np.ndarray,
    datasets: list[str],
):
    """
    Compare Fisher overlap with the LOO interaction matrix.

    The LOO matrix tells us which tasks actually interfere: positive values
    mean removing task j helps task i (j hurts i). The Fisher overlap
    predicts interference: tasks that modify the same parameters should
    interfere more.

    We test the hypothesis: high Fisher overlap between tasks i and j
    correlates with the amount of interference between them.

    Returns:
        correlations: dict with Pearson/Spearman correlations per block group
    """
    logger.info("\n" + "=" * 60)
    logger.info("PART 2b: Fisher overlap vs LOO interaction correlation")
    logger.info("=" * 60)

    n_tasks = len(datasets)
    results = {}

    # Build a symmetric "interference" matrix from the LOO data:
    # interference[i,j] = max(loo[i,j], loo[j,i]) -- how much do i and j hurt each other?
    # Alternatively: interference[i,j] = (loo[i,j] + loo[j,i]) / 2
    interference = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue
            val_ij = loo_matrix[i, j] if not np.isnan(loo_matrix[i, j]) else 0
            val_ji = loo_matrix[j, i] if not np.isnan(loo_matrix[j, i]) else 0
            interference[i, j] = (val_ij + val_ji) / 2.0

    # Extract upper triangles for correlation
    triu_idx = np.triu_indices(n_tasks, k=1)
    interference_vec = interference[triu_idx]

    for group_name in ["early", "late", "other"]:
        if group_name not in overlap_by_group:
            continue
        fisher_vec = overlap_by_group[group_name][triu_idx]

        # Pearson and Spearman correlations
        if fisher_vec.std() > 1e-12 and interference_vec.std() > 1e-12:
            pearson_r, pearson_p = stats.pearsonr(fisher_vec, interference_vec)
            spearman_r, spearman_p = stats.spearmanr(fisher_vec, interference_vec)
        else:
            pearson_r = pearson_p = spearman_r = spearman_p = 0.0

        results[group_name] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "fisher_overlap_vec": fisher_vec.tolist(),
            "interference_vec": interference_vec.tolist(),
        }

        logger.info(
            "  %s blocks: Pearson r=%.4f (p=%.4f), Spearman rho=%.4f (p=%.4f)",
            group_name, pearson_r, pearson_p, spearman_r, spearman_p,
        )

    return results, interference


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _task_colors():
    """Consistent color palette for the 8 tasks."""
    cmap = plt.cm.get_cmap("tab10", 10)
    return {ds: cmap(i) for i, ds in enumerate(N8_DATASETS)}


def plot_fisher_heatmap(
    fisher_mag: dict,
    layer_names_2d: list[str],
    datasets: list[str],
    output_dir: Path,
):
    """Heatmap: per-task Fisher magnitude across layers (tasks x layers)."""
    logger.info("Plotting Fisher magnitude heatmap...")

    n_tasks = len(datasets)
    n_layers = len(layer_names_2d)

    # Build matrix: (n_tasks, n_layers)
    data = np.zeros((n_tasks, n_layers))
    for i, ds in enumerate(datasets):
        for j, layer in enumerate(layer_names_2d):
            data[i, j] = fisher_mag[ds][layer]

    # Log-scale for better visibility
    data_log = np.log10(data + 1e-12)

    fig, ax = plt.subplots(figsize=(max(14, n_layers * 0.35), 6))
    im = ax.imshow(data_log, aspect="auto", cmap="viridis", interpolation="nearest")

    # Add block group boundaries
    block_groups = [classify_block_group(l) for l in layer_names_2d]
    prev_group = block_groups[0]
    for j in range(1, n_layers):
        if block_groups[j] != prev_group:
            ax.axvline(x=j - 0.5, color="white", linewidth=2, linestyle="--")
            prev_group = block_groups[j]

    # Mark late block region
    late_start = None
    for j, g in enumerate(block_groups):
        if g == "late" and late_start is None:
            late_start = j
    if late_start is not None:
        ax.axvspan(late_start - 0.5, n_layers - 0.5, alpha=0.08, color="red")
        ax.text(
            (late_start + n_layers - 1) / 2, -0.8, "LATE (blk 9-11)",
            ha="center", va="bottom", fontsize=9, color="red", fontweight="bold",
        )

    short_names = [shorten_layer_name(l) for l in layer_names_2d]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(datasets, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("log10(mean |tau|^2)", fontsize=10)

    ax.set_title("Per-Task Fisher Magnitude (Task Vector Squared) Across Layers", fontsize=12)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Task", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "fisher_magnitude_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fisher_magnitude_heatmap.png")


def plot_fisher_overlap_matrices(
    overlap_by_group: dict,
    datasets: list[str],
    output_dir: Path,
):
    """8x8 Fisher overlap heatmap for each block group (early, late)."""
    logger.info("Plotting Fisher overlap matrices...")

    groups = [g for g in ["early", "late"] if g in overlap_by_group]
    n_groups = len(groups)
    if n_groups == 0:
        logger.warning("No block groups found, skipping overlap plot")
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    n_tasks = len(datasets)

    for ax, group in zip(axes, groups):
        mat = overlap_by_group[group]
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0, interpolation="nearest")

        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(datasets, fontsize=9)

        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_tasks):
                val = mat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        upper = mat[np.triu_indices(n_tasks, k=1)]
        ax.set_title(
            f"{group.upper()} blocks\nmean overlap = {upper.mean():.3f}",
            fontsize=11,
        )

    fig.suptitle(
        "Fisher Overlap Between Tasks\n"
        "(Pearson correlation of per-element |tau|^2)",
        fontsize=13, y=1.02,
    )

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Pearson correlation")

    fig.savefig(output_dir / "fisher_overlap_early_vs_late.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fisher_overlap_early_vs_late.png")


def plot_fisher_vs_loo(
    fisher_loo_results: dict,
    overlap_by_group: dict,
    loo_matrix: np.ndarray,
    interference: np.ndarray,
    datasets: list[str],
    output_dir: Path,
):
    """
    Scatter plots and side-by-side heatmaps comparing Fisher overlap
    with LOO interference.
    """
    logger.info("Plotting Fisher vs LOO comparison...")
    n_tasks = len(datasets)
    triu_idx = np.triu_indices(n_tasks, k=1)

    # --- Figure 1: Scatter plots for each group ---
    groups = [g for g in ["early", "late"] if g in fisher_loo_results]
    n_groups = len(groups)

    if n_groups > 0:
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        if n_groups == 1:
            axes = [axes]

        interference_vec = interference[triu_idx]

        # Build pair labels
        pair_labels = []
        for idx in range(len(triu_idx[0])):
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            pair_labels.append(f"{datasets[i][:4]}-{datasets[j][:4]}")

        for ax, group in zip(axes, groups):
            fisher_vec = overlap_by_group[group][triu_idx]
            res = fisher_loo_results[group]

            ax.scatter(fisher_vec, interference_vec, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)

            # Label each point
            for k, label in enumerate(pair_labels):
                ax.annotate(
                    label, (fisher_vec[k], interference_vec[k]),
                    fontsize=5, alpha=0.8, textcoords="offset points",
                    xytext=(3, 3),
                )

            # Trend line
            if len(fisher_vec) > 2:
                z = np.polyfit(fisher_vec, interference_vec, 1)
                p = np.poly1d(z)
                x_range = np.linspace(fisher_vec.min(), fisher_vec.max(), 50)
                ax.plot(x_range, p(x_range), "r--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Fisher Overlap (Pearson corr of |tau|^2)", fontsize=10)
            ax.set_ylabel("LOO Interference (avg delta when removed)", fontsize=10)
            ax.set_title(
                f"{group.upper()} blocks\n"
                f"Pearson r={res['pearson_r']:.3f} (p={res['pearson_p']:.3f})\n"
                f"Spearman rho={res['spearman_r']:.3f} (p={res['spearman_p']:.3f})",
                fontsize=10,
            )
            ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)
            ax.grid(alpha=0.2)

        fig.suptitle(
            "Fisher Overlap vs LOO Interference\n"
            "Do tasks with high Fisher overlap interfere more?",
            fontsize=12, y=1.03,
        )
        plt.tight_layout()
        fig.savefig(output_dir / "fisher_vs_loo_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved fisher_vs_loo_scatter.png")

    # --- Figure 2: Side-by-side heatmaps (Fisher overlap late vs LOO interference) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LOO interference (symmetrized)
    ax = axes[0]
    masked_int = np.ma.array(interference, mask=(np.eye(n_tasks, dtype=bool)))
    vmax_int = np.nanmax(np.abs(interference))
    im1 = ax.imshow(masked_int, cmap="RdBu_r", vmin=-vmax_int, vmax=vmax_int)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(datasets, fontsize=9)
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                color = "white" if abs(interference[i, j]) > vmax_int * 0.6 else "black"
                ax.text(j, i, f"{interference[i,j]:.3f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im1, ax=ax, shrink=0.7)
    ax.set_title("LOO Interference\n(positive = removing j helps i)", fontsize=11)

    # Fisher overlap (late blocks)
    ax = axes[1]
    late_group = "late" if "late" in overlap_by_group else list(overlap_by_group.keys())[-1]
    mat = overlap_by_group[late_group]
    im2 = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(datasets, fontsize=9)
    for i in range(n_tasks):
        for j in range(n_tasks):
            val = mat[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im2, ax=ax, shrink=0.7)
    ax.set_title(f"Fisher Overlap ({late_group.upper()} blocks)\n(Pearson corr of |tau|^2)", fontsize=11)

    fig.suptitle("Side-by-Side: LOO Interference vs Fisher Overlap", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "fisher_vs_loo_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fisher_vs_loo_heatmaps.png")


def plot_fisher_merge_comparison(
    merge_stats: dict,
    layer_names_2d: list[str],
    output_dir: Path,
):
    """
    Plot how the Fisher-weighted merge differs from uniform across layers.
    Shows cosine similarity and relative L2 difference, early vs late.
    """
    logger.info("Plotting Fisher merge comparison...")

    short_names = [shorten_layer_name(l) for l in layer_names_2d]
    block_groups = [classify_block_group(l) for l in layer_names_2d]
    x = np.arange(len(layer_names_2d))

    # Color by block group
    group_colors = {"early": "#1f77b4", "late": "#d62728", "other": "#7f7f7f"}
    colors = [group_colors.get(g, "#7f7f7f") for g in block_groups]

    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(layer_names_2d) * 0.35), 12), sharex=True)

    # Panel 1: Cosine similarity to uniform merge
    ax = axes[0]
    cosines = [merge_stats[l]["cosine_vs_uniform"] for l in layer_names_2d]
    ax.bar(x, cosines, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_ylabel("Cosine(Fisher, Uniform)")
    ax.set_title("Fisher-Weighted vs Uniform Merge: Cosine Similarity", fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5)
    ax.set_ylim(min(0.9, min(cosines) - 0.02), 1.01)
    ax.grid(axis="y", alpha=0.2)

    # Panel 2: Relative L2 difference
    ax = axes[1]
    rel_l2 = [merge_stats[l]["relative_l2_vs_uniform"] for l in layer_names_2d]
    ax.bar(x, rel_l2, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_ylabel("Relative L2 Difference")
    ax.set_title("Fisher-Weighted vs Uniform Merge: Relative L2 Divergence", fontsize=11)
    ax.grid(axis="y", alpha=0.2)

    # Panel 3: Weight entropy ratio (1.0 = uniform, 0.0 = maximally concentrated)
    ax = axes[2]
    entropy_ratio = [merge_stats[l]["entropy_ratio"] for l in layer_names_2d]
    ax.bar(x, entropy_ratio, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_ylabel("Weight Entropy / max")
    ax.set_title("Fisher Weight Concentration (lower = more task-specific)", fontsize=11)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.5, label="Uniform")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)

    # X labels on bottom
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_xlabel("Layer", fontsize=10)

    # Add legend for block groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group_colors["early"], label="Early (blk 0-8)"),
        Patch(facecolor=group_colors["late"], label="Late (blk 9-11)"),
        Patch(facecolor=group_colors["other"], label="Other"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "fisher_merge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fisher_merge_comparison.png")


def plot_block_group_summary(
    fisher_mag: dict,
    overlap_by_group: dict,
    fisher_loo_results: dict,
    merge_group_stats: dict,
    datasets: list[str],
    layer_names_2d: list[str],
    output_dir: Path,
):
    """
    Summary bar chart comparing early vs late blocks across multiple metrics.
    """
    logger.info("Plotting block group summary...")

    n_tasks = len(datasets)

    # Compute per-group averages
    groups = ["early", "late"]
    metrics = {}

    for group in groups:
        group_layers = [l for l in layer_names_2d if classify_block_group(l) == group]
        if not group_layers:
            metrics[group] = {}
            continue

        # Average Fisher magnitude
        avg_fisher = np.mean([
            fisher_mag[ds][l] for ds in datasets for l in group_layers
        ])

        # Average pairwise Fisher overlap
        if group in overlap_by_group:
            upper = overlap_by_group[group][np.triu_indices(n_tasks, k=1)]
            avg_overlap = upper.mean()
        else:
            avg_overlap = 0.0

        # Fisher-LOO correlation
        if group in fisher_loo_results:
            corr = fisher_loo_results[group]["spearman_r"]
        else:
            corr = 0.0

        # Fisher merge divergence
        if group in merge_group_stats and merge_group_stats[group]["l2_vs_uniform"]:
            avg_divergence = np.mean(merge_group_stats[group]["l2_vs_uniform"])
        else:
            avg_divergence = 0.0

        # Weight concentration (1 - entropy ratio)
        if group in merge_group_stats and merge_group_stats[group]["weight_entropy"]:
            max_ent = float(np.log(n_tasks))
            avg_concentration = 1.0 - np.mean(merge_group_stats[group]["weight_entropy"]) / max_ent
        else:
            avg_concentration = 0.0

        metrics[group] = {
            "Avg Fisher\nMagnitude (x1e3)": avg_fisher * 1e3,
            "Avg Pairwise\nFisher Overlap": avg_overlap,
            "Fisher-LOO\nSpearman rho": corr,
            "Merge\nDivergence": avg_divergence,
            "Weight\nConcentration": avg_concentration,
        }

    # Check we have data for both groups
    if not metrics.get("early") or not metrics.get("late"):
        logger.warning("Not enough data for block group summary plot")
        return

    metric_names = list(metrics["early"].keys())
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_metrics)
    width = 0.35

    early_vals = [metrics["early"][m] for m in metric_names]
    late_vals = [metrics["late"][m] for m in metric_names]

    bars1 = ax.bar(x - width / 2, early_vals, width, label="Early (blk 0-8)", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + width / 2, late_vals, width, label="Late (blk 9-11)", color="#d62728", alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_title(
        "Early vs Late Blocks: Fisher Analysis Summary\n"
        "Late blocks show the zero-sum capacity problem",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(y=0, color="k", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "block_group_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved block_group_summary.png")


def plot_per_block_fisher_overlap_evolution(
    overlap_per_layer: dict,
    layer_names_2d: list[str],
    datasets: list[str],
    output_dir: Path,
):
    """
    Show how Fisher overlap evolves across blocks.
    One line per task pair, x-axis = block index, y-axis = Fisher overlap.
    """
    logger.info("Plotting per-block Fisher overlap evolution...")

    n_tasks = len(datasets)

    # Group layers by block index
    block_layers = defaultdict(list)
    for layer in layer_names_2d:
        idx = get_block_index(layer)
        if idx >= 0:
            block_layers[idx].append(layer)

    if not block_layers:
        logger.warning("No block layers found, skipping evolution plot")
        return

    block_indices = sorted(block_layers.keys())

    # For each block, average Fisher overlap across layers in that block
    # Compute average pairwise overlap (upper triangle mean)
    avg_overlap_per_block = []
    std_overlap_per_block = []
    all_pairs_per_block = {idx: [] for idx in block_indices}

    for idx in block_indices:
        layers = block_layers[idx]
        # Average overlap matrix for this block
        block_overlap = np.zeros((n_tasks, n_tasks))
        for layer in layers:
            block_overlap += overlap_per_layer[layer]
        block_overlap /= len(layers)

        upper = block_overlap[np.triu_indices(n_tasks, k=1)]
        avg_overlap_per_block.append(upper.mean())
        std_overlap_per_block.append(upper.std())

        # Store per-pair values for individual lines
        triu_idx = np.triu_indices(n_tasks, k=1)
        for k in range(len(triu_idx[0])):
            all_pairs_per_block[idx].append(block_overlap[triu_idx[0][k], triu_idx[1][k]])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Average overlap with error band
    ax = axes[0]
    avg = np.array(avg_overlap_per_block)
    std = np.array(std_overlap_per_block)
    ax.plot(block_indices, avg, "o-", color="#2c3e50", linewidth=2, markersize=6, label="Mean")
    ax.fill_between(block_indices, avg - std, avg + std, alpha=0.2, color="#2c3e50")

    # Highlight late blocks
    late_mask = np.array(block_indices) >= 9
    if late_mask.any():
        ax.axvspan(8.5, max(block_indices) + 0.5, alpha=0.1, color="red", label="Late (9-11)")

    ax.set_ylabel("Mean Pairwise Fisher Overlap", fontsize=11)
    ax.set_title("Fisher Overlap Evolution Across Transformer Blocks", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel 2: Individual pair trajectories (highlight most interfering pairs from LOO)
    ax = axes[1]

    # Identify top-3 interfering pairs from LOO
    loo_matrix, _ = load_loo_matrix()
    interference = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                val_ij = loo_matrix[i, j] if not np.isnan(loo_matrix[i, j]) else 0
                val_ji = loo_matrix[j, i] if not np.isnan(loo_matrix[j, i]) else 0
                interference[i, j] = (val_ij + val_ji) / 2.0

    triu_idx = np.triu_indices(n_tasks, k=1)
    pair_interference = []
    for k in range(len(triu_idx[0])):
        i, j = triu_idx[0][k], triu_idx[1][k]
        pair_interference.append((interference[i, j], i, j, k))
    pair_interference.sort(key=lambda x: x[0], reverse=True)

    # Plot all pairs in gray
    for k in range(len(triu_idx[0])):
        i, j = triu_idx[0][k], triu_idx[1][k]
        pair_vals = []
        for idx in block_indices:
            layers = block_layers[idx]
            block_overlap = np.zeros((n_tasks, n_tasks))
            for layer in layers:
                block_overlap += overlap_per_layer[layer]
            block_overlap /= len(layers)
            pair_vals.append(block_overlap[i, j])
        ax.plot(block_indices, pair_vals, "-", color="gray", alpha=0.15, linewidth=0.8)

    # Highlight top-3 interfering pairs
    highlight_colors = ["#e74c3c", "#e67e22", "#8e44ad"]
    for rank, (intf_val, i, j, _) in enumerate(pair_interference[:3]):
        pair_vals = []
        for idx in block_indices:
            layers = block_layers[idx]
            block_overlap = np.zeros((n_tasks, n_tasks))
            for layer in layers:
                block_overlap += overlap_per_layer[layer]
            block_overlap /= len(layers)
            pair_vals.append(block_overlap[i, j])
        label = f"{datasets[i][:4]}-{datasets[j][:4]} (LOO={intf_val:.3f})"
        ax.plot(
            block_indices, pair_vals, "o-",
            color=highlight_colors[rank], linewidth=2, markersize=4,
            label=label,
        )

    # Highlight bottom-3 (most synergistic pairs)
    synergy_colors = ["#27ae60", "#2ecc71", "#1abc9c"]
    for rank, (intf_val, i, j, _) in enumerate(pair_interference[-3:]):
        pair_vals = []
        for idx in block_indices:
            layers = block_layers[idx]
            block_overlap = np.zeros((n_tasks, n_tasks))
            for layer in layers:
                block_overlap += overlap_per_layer[layer]
            block_overlap /= len(layers)
            pair_vals.append(block_overlap[i, j])
        label = f"{datasets[i][:4]}-{datasets[j][:4]} (LOO={intf_val:.3f})"
        ax.plot(
            block_indices, pair_vals, "s--",
            color=synergy_colors[rank], linewidth=1.5, markersize=4,
            label=label,
        )

    if late_mask.any():
        ax.axvspan(8.5, max(block_indices) + 0.5, alpha=0.1, color="red")

    ax.set_xlabel("Transformer Block Index", fontsize=11)
    ax.set_ylabel("Fisher Overlap (Pearson corr)", fontsize=11)
    ax.set_title(
        "Per-Pair Fisher Overlap Trajectories\n"
        "(red = most interfering pairs from LOO, green = most synergistic)",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.2)
    ax.set_xticks(block_indices)

    plt.tight_layout()
    fig.savefig(output_dir / "fisher_overlap_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved fisher_overlap_evolution.png")


# --------------------------------------------------------------------------- #
# JSON serialization
# --------------------------------------------------------------------------- #

def make_json_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
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
        description="Fisher-informed merging analysis for ViT-B-32 N8 benchmark"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 tasks, subset of layers",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        datasets = N8_DATASETS[:3]
        logger.info("QUICK MODE: Using %d datasets: %s", len(datasets), datasets)
    else:
        datasets = N8_DATASETS
        logger.info("Full analysis: %d datasets: %s", len(datasets), datasets)

    # ------------------------------------------------------------------ #
    # Load task vectors
    # ------------------------------------------------------------------ #
    task_vectors = load_task_vectors(datasets, MODEL_NAME)

    first_tv = task_vectors[datasets[0]]
    layer_names_2d = [k for k, v in first_tv.items() if v.dim() == 2]

    if args.quick:
        step = max(1, len(layer_names_2d) // 8)
        layer_names_2d = layer_names_2d[::step][:8]
        logger.info("QUICK MODE: Using %d layers", len(layer_names_2d))

    logger.info("Total 2D layers: %d", len(layer_names_2d))
    n_early = sum(1 for l in layer_names_2d if classify_block_group(l) == "early")
    n_late = sum(1 for l in layer_names_2d if classify_block_group(l) == "late")
    n_other = sum(1 for l in layer_names_2d if classify_block_group(l) == "other")
    logger.info("  early: %d, late: %d, other: %d", n_early, n_late, n_other)

    # ------------------------------------------------------------------ #
    # Part 1: Per-task Fisher magnitude
    # ------------------------------------------------------------------ #
    fisher_mag, fisher_per_element = compute_fisher_magnitudes(
        task_vectors, layer_names_2d, datasets,
    )

    # ------------------------------------------------------------------ #
    # Part 2: Pairwise Fisher overlap
    # ------------------------------------------------------------------ #
    overlap_by_group, overlap_per_layer = compute_fisher_overlap(
        fisher_per_element, layer_names_2d, datasets,
    )

    # ------------------------------------------------------------------ #
    # Part 2b: Compare Fisher overlap vs LOO interaction
    # ------------------------------------------------------------------ #
    loo_matrix, loo_tasks = load_loo_matrix()

    # Only compare if we have the same tasks
    if set(datasets) == set(loo_tasks):
        # Reorder LOO matrix to match our dataset ordering
        reorder_idx = [loo_tasks.index(ds) for ds in datasets]
        loo_reordered = loo_matrix[np.ix_(reorder_idx, reorder_idx)]
        fisher_loo_results, interference = compare_fisher_vs_loo(
            overlap_by_group, loo_reordered, datasets,
        )
    else:
        logger.warning(
            "Dataset mismatch with LOO results (have %s, LOO has %s). "
            "Skipping Fisher-LOO comparison.",
            datasets, loo_tasks,
        )
        fisher_loo_results = {}
        interference = np.zeros((len(datasets), len(datasets)))

    # ------------------------------------------------------------------ #
    # Part 3: Fisher-weighted merge prototype
    # ------------------------------------------------------------------ #
    merge_stats, merge_group_stats = fisher_weighted_merge_analysis(
        task_vectors, fisher_per_element, layer_names_2d, datasets,
    )

    # ------------------------------------------------------------------ #
    # Save JSON results
    # ------------------------------------------------------------------ #
    json_path = OUTPUT_DIR / "fisher_analysis.json"
    all_results = {
        "config": {
            "model": MODEL_NAME,
            "datasets": datasets,
            "n_layers_2d": len(layer_names_2d),
            "quick_mode": args.quick,
        },
        "fisher_magnitude": {
            ds: {l: fisher_mag[ds][l] for l in layer_names_2d}
            for ds in datasets
        },
        "fisher_overlap_by_group": {
            group: mat.tolist()
            for group, mat in overlap_by_group.items()
        },
        "fisher_vs_loo": {
            group: {k: v for k, v in res.items() if k not in ("fisher_overlap_vec", "interference_vec")}
            for group, res in fisher_loo_results.items()
        },
        "merge_stats_per_layer": merge_stats,
        "merge_group_stats": {
            group: {k: float(np.mean(v)) if v else 0.0 for k, v in gs.items()}
            for group, gs in merge_group_stats.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    logger.info("Results saved to %s", json_path)

    # ------------------------------------------------------------------ #
    # Generate plots
    # ------------------------------------------------------------------ #
    logger.info("\nGenerating plots...")

    plot_fisher_heatmap(fisher_mag, layer_names_2d, datasets, OUTPUT_DIR)
    plot_fisher_overlap_matrices(overlap_by_group, datasets, OUTPUT_DIR)

    if fisher_loo_results:
        plot_fisher_vs_loo(
            fisher_loo_results, overlap_by_group, loo_reordered,
            interference, datasets, OUTPUT_DIR,
        )

    plot_fisher_merge_comparison(merge_stats, layer_names_2d, OUTPUT_DIR)

    plot_block_group_summary(
        fisher_mag, overlap_by_group, fisher_loo_results,
        merge_group_stats, datasets, layer_names_2d, OUTPUT_DIR,
    )

    plot_per_block_fisher_overlap_evolution(
        overlap_per_layer, layer_names_2d, datasets, OUTPUT_DIR,
    )

    # ------------------------------------------------------------------ #
    # Print key findings
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    n_tasks = len(datasets)
    for group in ["early", "late"]:
        if group not in overlap_by_group:
            continue
        upper = overlap_by_group[group][np.triu_indices(n_tasks, k=1)]
        logger.info(
            "  %s blocks: mean Fisher overlap = %.4f (std=%.4f)",
            group, upper.mean(), upper.std(),
        )

    if fisher_loo_results:
        for group in ["early", "late"]:
            if group not in fisher_loo_results:
                continue
            res = fisher_loo_results[group]
            logger.info(
                "  %s blocks: Fisher-LOO Spearman rho = %.4f (p=%.4f)",
                group, res["spearman_r"], res["spearman_p"],
            )

    for group in ["early", "late"]:
        gs = merge_group_stats.get(group, {})
        if gs and gs.get("cosine_vs_uniform"):
            logger.info(
                "  %s blocks: Fisher-merge cosine to uniform = %.4f, "
                "rel L2 divergence = %.4f",
                group,
                np.mean(gs["cosine_vs_uniform"]),
                np.mean(gs["l2_vs_uniform"]),
            )

    logger.info("\nAll outputs saved to: %s", OUTPUT_DIR)
    logger.info("Plots: %s", ", ".join(f.name for f in OUTPUT_DIR.glob("*.png")))
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
