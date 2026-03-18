"""
Subspace decomposition validation for Iso-CTS merging.

Validates whether the common/task-specific subspace split used in
IsotropicCommonTaskSpecificMerger is clean and effective.

Analyses performed:
1. Energy distribution: fraction of each task vector's energy in common vs task-specific subspace
2. Interference analysis: pairwise cosine similarity in each subspace
3. Per-task analysis: which tasks are most/least aligned with the common subspace
4. Rank sensitivity: how the split changes with different common subspace sizes

Usage:
    uv run python scripts/analyze_subspace.py           # full analysis (8 tasks, all layers)
    uv run python scripts/analyze_subspace.py --quick   # quick mode (3 tasks, 5 layers)
"""

import argparse
import json
import logging
import os
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.merging.task_vectors import compute_task_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
MODEL_NAME = "ViT-B-32"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

# Default common_space_fraction from iso-cts.yaml
DEFAULT_COMMON_FRACTION = 0.8

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def classify_layer(layer_name: str) -> str:
    """Classify a layer name into a human-readable category."""
    if "in_proj" in layer_name or "out_proj" in layer_name or "q_proj" in layer_name or "k_proj" in layer_name or "v_proj" in layer_name:
        return "attention"
    if "c_fc" in layer_name or "mlp" in layer_name:
        return "mlp_fc"
    if "c_proj" in layer_name:
        return "mlp_proj"
    if "conv1" in layer_name:
        return "conv_embed"
    if "class_embedding" in layer_name or "positional_embedding" in layer_name:
        return "embedding"
    if "ln" in layer_name or "norm" in layer_name:
        return "layernorm"
    if "proj" in layer_name:
        return "projection"
    return "other"


def get_2d_layers(task_vector: OrderedDict):
    """Return list of layer names that are 2D matrices and not text_projection."""
    return [k for k, v in task_vector.items() if v.dim() == 2 and "text_projection" not in k]


def marchenko_pastur_rank(singular_values: np.ndarray, m: int, n: int) -> int:
    """
    Estimate the number of signal singular values using the Marchenko-Pastur edge.

    Returns the number of singular values whose squared value exceeds the MP bulk edge.
    """
    gamma = max(m, n) / min(m, n)
    # Estimate noise variance from the bulk of singular values
    # Use median of squared SVs as a robust estimator
    sv_sq = singular_values ** 2
    sigma2 = float(np.median(sv_sq)) / (np.sqrt(gamma) + 1) ** 2
    if sigma2 <= 0:
        sigma2 = float(np.mean(sv_sq[-len(sv_sq) // 4:])) if len(sv_sq) > 4 else 1.0
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    signal_count = int(np.sum(sv_sq > lambda_plus))
    return max(1, signal_count)


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / (norm_a * norm_b))


def frobenius_norm(t: torch.Tensor) -> float:
    """Frobenius norm of a tensor."""
    return float(torch.norm(t.float(), p="fro"))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    """Load state dict directly from HF cache."""
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def load_task_vectors(datasets, model_name=MODEL_NAME):
    """Load pretrained model and compute task vectors for all datasets."""
    logger.info(f"Loading pretrained state dict: {model_name}")
    pretrained = load_state_dict_from_hf_cache(model_name, "base")

    task_vectors = {}
    for ds in datasets:
        logger.info(f"Loading finetuned state dict for {ds}")
        finetuned = load_state_dict_from_hf_cache(model_name, ds)
        tv = compute_task_vector(pretrained, finetuned, device="cpu")
        task_vectors[ds] = tv
        del finetuned
        logger.info(f"  {ds}: {len(tv)} layers, total params = {sum(v.numel() for v in tv.values()):,}")

    del pretrained
    return task_vectors


# --------------------------------------------------------------------------- #
# Core subspace decomposition
# --------------------------------------------------------------------------- #

def compute_common_subspace_rank(summed_matrix: torch.Tensor, fraction: float, num_tasks: int):
    """
    Compute the common subspace rank following the exact Iso-CTS logic.

    Mirrors IsotropicCommonTaskSpecificMerger:
      common_space_index_s = int(min(shape) * fraction)
      task_specific_total = round((min(shape) - common_space_index_s) / num_tasks) * num_tasks
      common_space_index_s = min(shape) - task_specific_total

    Returns: (common_rank, n_dims_per_task)
    """
    min_dim = min(summed_matrix.shape)
    common_rank = int(min_dim * fraction)
    task_specific_total = round((min_dim - common_rank) / num_tasks) * num_tasks
    common_rank = min_dim - task_specific_total
    n_dims_per_task = int((min_dim - common_rank) / num_tasks) if num_tasks > 0 else 0
    return common_rank, n_dims_per_task


def decompose_layer(task_vectors_layer: dict, datasets: list, common_rank: int):
    """
    Decompose a single layer's task vectors into common and task-specific components.

    Args:
        task_vectors_layer: {dataset_name: tensor} for this layer
        datasets: ordered list of dataset names
        common_rank: number of top singular vectors of summed TV to use as common subspace

    Returns:
        dict with keys: U_common, S_common, V_common,
        common_projections: {ds: tensor}, task_specific_residuals: {ds: tensor}
    """
    # Step 1: Sum all task vectors
    summed_tv = sum(task_vectors_layer[ds].float() for ds in datasets)

    # Step 2: SVD of the summed task vector
    U_sum, S_sum, V_sum = torch.linalg.svd(summed_tv, full_matrices=False)

    # Step 3: Common subspace basis (top-k left singular vectors)
    U_common = U_sum[:, :common_rank]  # (m, k)

    # Step 4: Project each task vector onto common and task-specific subspaces
    common_projections = {}
    task_specific_residuals = {}

    for ds in datasets:
        tau_i = task_vectors_layer[ds].float()

        # Common projection: U_common @ U_common^T @ tau_i
        # This projects the rows of tau_i onto the column space spanned by U_common
        proj_common = U_common @ (U_common.T @ tau_i)

        # Task-specific: residual after removing common projection
        proj_ts = tau_i - proj_common

        common_projections[ds] = proj_common
        task_specific_residuals[ds] = proj_ts

    return {
        "U_common": U_common,
        "S_sum": S_sum,
        "V_sum": V_sum,
        "summed_tv": summed_tv,
        "common_projections": common_projections,
        "task_specific_residuals": task_specific_residuals,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def analyze_subspace_decomposition(task_vectors, layer_names_2d, datasets, common_fraction=DEFAULT_COMMON_FRACTION):
    """
    Main analysis: for each 2D layer, decompose into common/task-specific subspaces
    and measure energy distribution and interference.
    """
    logger.info("=" * 70)
    logger.info("SUBSPACE DECOMPOSITION ANALYSIS")
    logger.info(f"  Common space fraction: {common_fraction}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Layers: {len(layer_names_2d)}")
    logger.info("=" * 70)

    num_tasks = len(datasets)
    results = {}

    for layer_name in layer_names_2d:
        logger.info(f"  Processing {layer_name}...")
        layer_type = classify_layer(layer_name)

        # Collect task vectors for this layer
        tv_layer = {ds: task_vectors[ds][layer_name] for ds in datasets}
        shape = tv_layer[datasets[0]].shape

        # Compute common rank following Iso-CTS logic
        common_rank, n_dims_per_task = compute_common_subspace_rank(
            tv_layer[datasets[0]], common_fraction, num_tasks
        )

        # Decompose
        decomp = decompose_layer(tv_layer, datasets, common_rank)

        # ------ Energy fractions ------
        common_energy_fraction = {}
        task_specific_energy_fraction = {}
        for ds in datasets:
            tau_norm = frobenius_norm(tv_layer[ds])
            if tau_norm < 1e-12:
                common_energy_fraction[ds] = 0.0
                task_specific_energy_fraction[ds] = 0.0
                continue
            common_norm = frobenius_norm(decomp["common_projections"][ds])
            ts_norm = frobenius_norm(decomp["task_specific_residuals"][ds])
            # Energy fractions (squared norms for energy, but Frobenius norm ratio is also informative)
            # Use squared norms for proper energy decomposition (Pythagorean since subspaces are orthogonal)
            common_energy_fraction[ds] = (common_norm / tau_norm) ** 2
            task_specific_energy_fraction[ds] = (ts_norm / tau_norm) ** 2

        # ------ Pairwise cosine similarity (interference) ------
        common_pairwise_cossim = {}
        ts_pairwise_cossim = {}
        original_pairwise_cossim = {}

        for ds_i, ds_j in combinations(datasets, 2):
            pair_key = f"{ds_i}-{ds_j}"

            # Common subspace interference
            common_pairwise_cossim[pair_key] = cosine_similarity_flat(
                decomp["common_projections"][ds_i],
                decomp["common_projections"][ds_j],
            )

            # Task-specific subspace interference
            ts_pairwise_cossim[pair_key] = cosine_similarity_flat(
                decomp["task_specific_residuals"][ds_i],
                decomp["task_specific_residuals"][ds_j],
            )

            # Original (unsplit) interference for comparison
            original_pairwise_cossim[pair_key] = cosine_similarity_flat(
                tv_layer[ds_i], tv_layer[ds_j]
            )

        # ------ Cross-subspace leakage ------
        # For each task, how much does its task-specific component
        # overlap with OTHER tasks' common projections?
        leakage = {}
        for ds_i in datasets:
            # Measure how much ds_i's task-specific residual aligns with the common subspace
            # (should be ~0 by construction, but numerical errors may cause leakage)
            ts_i = decomp["task_specific_residuals"][ds_i]
            # Project ts_i back onto common subspace
            U_common = decomp["U_common"]
            re_projection = U_common @ (U_common.T @ ts_i)
            ts_norm = frobenius_norm(ts_i)
            reproj_norm = frobenius_norm(re_projection)
            leakage[ds_i] = (reproj_norm / ts_norm) ** 2 if ts_norm > 1e-12 else 0.0

        # ------ Singular value spectrum of summed TV ------
        S_sum = decomp["S_sum"].numpy()
        # Truncate to first 50 for storage
        sv_spectrum = S_sum[:min(50, len(S_sum))].tolist()

        # ------ MP rank estimate ------
        mp_rank = marchenko_pastur_rank(S_sum, shape[0], shape[1])

        results[layer_name] = {
            "common_energy_fraction": common_energy_fraction,
            "task_specific_energy_fraction": task_specific_energy_fraction,
            "common_pairwise_cossim": common_pairwise_cossim,
            "task_specific_pairwise_cossim": ts_pairwise_cossim,
            "original_pairwise_cossim": original_pairwise_cossim,
            "leakage": leakage,
            "common_rank_used": common_rank,
            "n_dims_per_task": n_dims_per_task,
            "mp_rank_estimate": mp_rank,
            "summed_sv_spectrum": sv_spectrum,
            "layer_type": layer_type,
            "shape": list(shape),
        }

        # Log summary for this layer
        avg_common_energy = np.mean(list(common_energy_fraction.values()))
        avg_common_cossim = np.mean(list(common_pairwise_cossim.values()))
        avg_ts_cossim = np.mean(list(ts_pairwise_cossim.values()))
        avg_leakage = np.mean(list(leakage.values()))
        logger.info(
            f"    rank={common_rank}/{min(shape)}, "
            f"avg_common_energy={avg_common_energy:.3f}, "
            f"avg_cossim common={avg_common_cossim:.4f} / ts={avg_ts_cossim:.4f}, "
            f"leakage={avg_leakage:.2e}"
        )

    return results


def analyze_rank_sensitivity(task_vectors, layer_names_2d, datasets):
    """
    Analyze how the subspace split changes with different common subspace sizes.

    Tests: fraction=0.8 (default), plus fixed ranks k=num_tasks/2, k=num_tasks, k=num_tasks*2,
    and MP-estimated rank.
    """
    logger.info("\n" + "=" * 70)
    logger.info("RANK SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    num_tasks = len(datasets)
    # We test different rank selection strategies
    rank_configs = {
        "fraction_0.8": {"type": "fraction", "value": 0.8},
        "fraction_0.5": {"type": "fraction", "value": 0.5},
        f"fixed_k={num_tasks // 2}": {"type": "fixed", "value": max(1, num_tasks // 2)},
        f"fixed_k={num_tasks}": {"type": "fixed", "value": num_tasks},
        f"fixed_k={num_tasks * 2}": {"type": "fixed", "value": num_tasks * 2},
        "mp_estimate": {"type": "mp", "value": None},
    }

    results = {}

    for config_name, config in rank_configs.items():
        logger.info(f"\n  Testing rank config: {config_name}")
        config_results = {}

        for layer_name in layer_names_2d:
            tv_layer = {ds: task_vectors[ds][layer_name] for ds in datasets}
            shape = tv_layer[datasets[0]].shape
            min_dim = min(shape)

            # Determine common rank
            if config["type"] == "fraction":
                common_rank, _ = compute_common_subspace_rank(
                    tv_layer[datasets[0]], config["value"], num_tasks
                )
            elif config["type"] == "fixed":
                common_rank = min(config["value"], min_dim - 1)
                common_rank = max(1, common_rank)
            elif config["type"] == "mp":
                summed_tv = sum(tv_layer[ds].float() for ds in datasets)
                S_sum = torch.linalg.svdvals(summed_tv).numpy()
                common_rank = marchenko_pastur_rank(S_sum, shape[0], shape[1])
                common_rank = min(common_rank, min_dim - 1)

            # Decompose
            decomp = decompose_layer(tv_layer, datasets, common_rank)

            # Compute summary metrics
            common_energies = []
            ts_cossims = []
            common_cossims = []

            for ds in datasets:
                tau_norm = frobenius_norm(tv_layer[ds])
                if tau_norm < 1e-12:
                    common_energies.append(0.0)
                    continue
                c_norm = frobenius_norm(decomp["common_projections"][ds])
                common_energies.append((c_norm / tau_norm) ** 2)

            for ds_i, ds_j in combinations(datasets, 2):
                common_cossims.append(cosine_similarity_flat(
                    decomp["common_projections"][ds_i],
                    decomp["common_projections"][ds_j],
                ))
                ts_cossims.append(cosine_similarity_flat(
                    decomp["task_specific_residuals"][ds_i],
                    decomp["task_specific_residuals"][ds_j],
                ))

            config_results[layer_name] = {
                "common_rank": common_rank,
                "min_dim": min_dim,
                "avg_common_energy": float(np.mean(common_energies)),
                "avg_common_cossim": float(np.mean(common_cossims)) if common_cossims else 0.0,
                "avg_ts_cossim": float(np.mean(ts_cossims)) if ts_cossims else 0.0,
            }

        results[config_name] = config_results

        # Print summary
        avg_energy = np.mean([v["avg_common_energy"] for v in config_results.values()])
        avg_c_cos = np.mean([v["avg_common_cossim"] for v in config_results.values()])
        avg_ts_cos = np.mean([v["avg_ts_cossim"] for v in config_results.values()])
        logger.info(
            f"    avg_common_energy={avg_energy:.3f}, "
            f"avg_cossim common={avg_c_cos:.4f} / ts={avg_ts_cos:.4f}"
        )

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_energy_heatmap(results, layer_names_2d, datasets, output_dir):
    """Plot heatmap: tasks (rows) x layers (cols), values = common energy fraction."""
    logger.info("Plotting energy distribution heatmap...")

    # Build matrix
    n_tasks = len(datasets)
    n_layers = len(layer_names_2d)
    energy_matrix = np.zeros((n_tasks, n_layers))

    for j, layer_name in enumerate(layer_names_2d):
        for i, ds in enumerate(datasets):
            energy_matrix[i, j] = results[layer_name]["common_energy_fraction"][ds]

    # Shorten layer names
    short_names = []
    for name in layer_names_2d:
        parts = name.split(".")
        short = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_names.append(short)

    fig, ax = plt.subplots(figsize=(max(14, n_layers * 0.5), max(4, n_tasks * 0.6)))
    im = ax.imshow(energy_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(datasets, fontsize=9)

    # Add text annotations if not too many cells
    if n_layers <= 30:
        for i in range(n_tasks):
            for j in range(n_layers):
                val = energy_matrix[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    plt.colorbar(im, label="Fraction of energy in common subspace", shrink=0.8)
    ax.set_title("Common Subspace Energy Fraction (per task x layer)", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Task")
    plt.tight_layout()
    plt.savefig(output_dir / "subspace_energy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_interference_comparison(results, layer_names_2d, output_dir):
    """Plot average pairwise cosine similarity: common vs task-specific vs original."""
    logger.info("Plotting interference comparison...")

    layers = []
    common_cossims = []
    ts_cossims = []
    orig_cossims = []

    for layer_name in layer_names_2d:
        r = results[layer_name]
        layers.append(layer_name)
        common_cossims.append(np.mean(list(r["common_pairwise_cossim"].values())))
        ts_cossims.append(np.mean(list(r["task_specific_pairwise_cossim"].values())))
        orig_cossims.append(np.mean(list(r["original_pairwise_cossim"].values())))

    short_names = []
    for name in layers:
        parts = name.split(".")
        short = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_names.append(short)

    x = np.arange(len(layers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.5), 5))
    ax.bar(x - width, orig_cossims, width, label="Original (unsplit)", alpha=0.8, color="gray")
    ax.bar(x, common_cossims, width, label="Common subspace", alpha=0.8, color="tab:red")
    ax.bar(x + width, ts_cossims, width, label="Task-specific subspace", alpha=0.8, color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average pairwise cosine similarity")
    ax.set_title("Interference: Common vs Task-Specific Subspace", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "subspace_interference_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_task_energy(results, layer_names_2d, datasets, output_dir):
    """Plot per-task average common energy fraction across layers."""
    logger.info("Plotting per-task energy analysis...")

    # Average over layers for each task
    task_energies = {ds: [] for ds in datasets}
    for layer_name in layer_names_2d:
        for ds in datasets:
            task_energies[ds].append(results[layer_name]["common_energy_fraction"][ds])

    avg_energies = {ds: np.mean(vals) for ds, vals in task_energies.items()}
    std_energies = {ds: np.std(vals) for ds, vals in task_energies.items()}

    # Sort by average energy
    sorted_datasets = sorted(datasets, key=lambda ds: avg_energies[ds], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_datasets))
    means = [avg_energies[ds] for ds in sorted_datasets]
    stds = [std_energies[ds] for ds in sorted_datasets]

    bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, color="tab:orange", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_datasets, fontsize=10)
    ax.set_ylabel("Average common energy fraction")
    ax.set_title("Per-Task Energy in Common Subspace (averaged over layers)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "subspace_per_task_energy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_rank_sensitivity(rank_results, layer_names_2d, output_dir):
    """Plot how metrics change across different rank configurations."""
    logger.info("Plotting rank sensitivity...")

    config_names = list(rank_results.keys())
    n_configs = len(config_names)

    # Aggregate metrics per config
    metrics = {
        "avg_common_energy": [],
        "avg_common_cossim": [],
        "avg_ts_cossim": [],
    }
    avg_ranks = []

    for config_name in config_names:
        config_data = rank_results[config_name]
        for metric_key in metrics:
            metrics[metric_key].append(
                np.mean([v[metric_key] for v in config_data.values()])
            )
        avg_ranks.append(np.mean([v["common_rank"] for v in config_data.values()]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Common energy vs rank config
    axes[0].bar(range(n_configs), metrics["avg_common_energy"], alpha=0.8, color="tab:orange")
    axes[0].set_xticks(range(n_configs))
    axes[0].set_xticklabels(config_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Avg common energy fraction")
    axes[0].set_title("Common Energy vs Rank Config")
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Interference comparison
    x = np.arange(n_configs)
    width = 0.35
    axes[1].bar(x - width / 2, metrics["avg_common_cossim"], width, label="Common", color="tab:red", alpha=0.8)
    axes[1].bar(x + width / 2, metrics["avg_ts_cossim"], width, label="Task-specific", color="tab:blue", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Avg pairwise cosine sim")
    axes[1].set_title("Interference vs Rank Config")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axhline(y=0, color="k", linewidth=0.5)

    # 3. Average rank used
    axes[2].bar(range(n_configs), avg_ranks, alpha=0.8, color="tab:green")
    axes[2].set_xticks(range(n_configs))
    axes[2].set_xticklabels(config_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Avg common rank")
    axes[2].set_title("Common Rank Used")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Rank Sensitivity Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "subspace_rank_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sv_spectrum_summed(results, layer_names_2d, output_dir):
    """Plot singular value spectra of the summed task vectors for representative layers."""
    logger.info("Plotting summed TV singular value spectra...")

    # Pick representative layers (one per type)
    representative_layers = []
    seen_types = set()
    for layer_name in layer_names_2d:
        lt = classify_layer(layer_name)
        if lt not in seen_types and lt in ("attention", "mlp_fc", "mlp_proj", "projection"):
            representative_layers.append(layer_name)
            seen_types.add(lt)
        if len(representative_layers) >= 4:
            break
    if not representative_layers:
        representative_layers = layer_names_2d[:min(4, len(layer_names_2d))]

    n_layers = len(representative_layers)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, representative_layers):
        r = results[layer_name]
        sv = np.array(r["summed_sv_spectrum"])
        common_rank = r["common_rank_used"]
        mp_rank = r["mp_rank_estimate"]

        ax.semilogy(sv, "b-", alpha=0.8, label="Summed TV SVs")
        ax.axvline(x=common_rank, color="r", linestyle="--", alpha=0.7,
                    label=f"Iso-CTS rank ({common_rank})")
        ax.axvline(x=mp_rank, color="g", linestyle=":", alpha=0.7,
                    label=f"MP rank ({mp_rank})")

        layer_short = ".".join(layer_name.split(".")[-2:])
        layer_type = classify_layer(layer_name)
        ax.set_title(f"{layer_type}\n{layer_short}", fontsize=9)
        ax.set_xlabel("SV index")
        ax.set_ylabel("Singular value")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Singular Values of Summed Task Vectors (with rank cutoffs)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "subspace_summed_sv_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_summary_table(results, layer_names_2d, datasets):
    """Print a summary table to stdout."""
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 100)

    header = (
        f"{'Layer':<45s}  {'Type':<12s}  {'Rank':>5s}  {'MP':>4s}  "
        f"{'Avg E_com':>9s}  {'CosSim_com':>10s}  {'CosSim_ts':>9s}  {'CosSim_orig':>11s}  {'Leakage':>8s}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    # Accumulators for global summary
    all_common_energies = []
    all_common_cossim = []
    all_ts_cossim = []
    all_orig_cossim = []
    all_leakage = []

    for layer_name in layer_names_2d:
        r = results[layer_name]

        avg_common_e = np.mean(list(r["common_energy_fraction"].values()))
        avg_common_cos = np.mean(list(r["common_pairwise_cossim"].values()))
        avg_ts_cos = np.mean(list(r["task_specific_pairwise_cossim"].values()))
        avg_orig_cos = np.mean(list(r["original_pairwise_cossim"].values()))
        avg_leak = np.mean(list(r["leakage"].values()))

        all_common_energies.append(avg_common_e)
        all_common_cossim.append(avg_common_cos)
        all_ts_cossim.append(avg_ts_cos)
        all_orig_cossim.append(avg_orig_cos)
        all_leakage.append(avg_leak)

        # Truncate layer name for display
        display_name = layer_name if len(layer_name) <= 44 else "..." + layer_name[-41:]

        logger.info(
            f"{display_name:<45s}  {r['layer_type']:<12s}  {r['common_rank_used']:>5d}  {r['mp_rank_estimate']:>4d}  "
            f"{avg_common_e:>9.3f}  {avg_common_cos:>10.4f}  {avg_ts_cos:>9.4f}  {avg_orig_cos:>11.4f}  {avg_leak:>8.2e}"
        )

    logger.info("-" * len(header))
    logger.info(
        f"{'GLOBAL AVERAGE':<45s}  {'':<12s}  {'':>5s}  {'':>4s}  "
        f"{np.mean(all_common_energies):>9.3f}  {np.mean(all_common_cossim):>10.4f}  "
        f"{np.mean(all_ts_cossim):>9.4f}  {np.mean(all_orig_cossim):>11.4f}  {np.mean(all_leakage):>8.2e}"
    )

    # Per-task summary
    logger.info("\n--- Per-Task Average Common Energy Fraction ---")
    task_avg = {}
    for ds in datasets:
        vals = [results[ln]["common_energy_fraction"][ds] for ln in layer_names_2d]
        task_avg[ds] = np.mean(vals)
    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1], reverse=True)
    for ds, avg_e in sorted_tasks:
        logger.info(f"  {ds:<12s}: {avg_e:.4f}")

    # Interference reduction summary
    logger.info("\n--- Interference Reduction (common vs task-specific) ---")
    logger.info(f"  Avg original pairwise cossim:      {np.mean(all_orig_cossim):.4f}")
    logger.info(f"  Avg common subspace cossim:         {np.mean(all_common_cossim):.4f}")
    logger.info(f"  Avg task-specific subspace cossim:  {np.mean(all_ts_cossim):.4f}")
    reduction = np.mean(all_orig_cossim) - np.mean(all_ts_cossim)
    logger.info(f"  Interference reduction (orig - ts): {reduction:.4f}")

    logger.info(f"\n  Avg leakage (should be ~0):         {np.mean(all_leakage):.2e}")


# --------------------------------------------------------------------------- #
# JSON serialization
# --------------------------------------------------------------------------- #

def make_json_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
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
        description="Validate common/task-specific subspace decomposition for Iso-CTS merging"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks, 5 layers")
    parser.add_argument(
        "--common-fraction", type=float, default=DEFAULT_COMMON_FRACTION,
        help=f"Common space fraction (default: {DEFAULT_COMMON_FRACTION}, matching iso-cts.yaml)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        datasets = N8_DATASETS[:3]
        logger.info(f"QUICK MODE: Using {len(datasets)} datasets: {datasets}")
    else:
        datasets = N8_DATASETS
        logger.info(f"Full analysis: {len(datasets)} datasets: {datasets}")

    # Load task vectors
    task_vectors = load_task_vectors(datasets)

    # Get 2D layer names (consistent across tasks), excluding text_projection
    first_tv = task_vectors[datasets[0]]
    layer_names_2d = get_2d_layers(first_tv)

    if args.quick:
        step = max(1, len(layer_names_2d) // 5)
        layer_names_2d = layer_names_2d[::step][:5]
        logger.info(f"QUICK MODE: Using {len(layer_names_2d)} layers")

    logger.info(f"Total 2D layers to analyze: {len(layer_names_2d)}")
    logger.info(f"Datasets: {datasets}")

    all_results = {}
    json_path = OUTPUT_DIR / "subspace_analysis.json"

    def save_intermediate():
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        logger.info(f"Intermediate results saved to: {json_path}")

    # 1. Main subspace decomposition analysis
    decomp_results = analyze_subspace_decomposition(
        task_vectors, layer_names_2d, datasets, args.common_fraction
    )
    all_results["subspace_decomposition"] = decomp_results
    all_results["metadata"] = {
        "model": MODEL_NAME,
        "datasets": datasets,
        "common_fraction": args.common_fraction,
        "num_layers_analyzed": len(layer_names_2d),
        "layer_names": layer_names_2d,
        "quick_mode": args.quick,
    }
    save_intermediate()

    # 2. Rank sensitivity analysis
    rank_results = analyze_rank_sensitivity(task_vectors, layer_names_2d, datasets)
    all_results["rank_sensitivity"] = rank_results
    save_intermediate()

    # 3. Print summary
    print_summary_table(decomp_results, layer_names_2d, datasets)

    # 4. Generate plots
    logger.info("\nGenerating plots...")
    plot_energy_heatmap(decomp_results, layer_names_2d, datasets, OUTPUT_DIR)
    plot_interference_comparison(decomp_results, layer_names_2d, OUTPUT_DIR)
    plot_per_task_energy(decomp_results, layer_names_2d, datasets, OUTPUT_DIR)
    plot_rank_sensitivity(rank_results, layer_names_2d, OUTPUT_DIR)
    plot_sv_spectrum_summed(decomp_results, layer_names_2d, OUTPUT_DIR)

    logger.info(f"\nAll results saved to: {json_path}")
    logger.info(f"All plots saved to: {OUTPUT_DIR}")
    logger.info("Subspace analysis complete.")


if __name__ == "__main__":
    main()
