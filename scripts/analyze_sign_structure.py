"""
Analyze element-wise sign patterns of task vectors from N8 ViT-B-32 models.

Investigates whether individual parameter-level sign agreement/conflict between
tasks correlates with interference during merging.

Analyses:
1. Sign agreement score distribution across all parameters
2. Sign agreement by layer type and transformer block depth
3. Sign-magnitude interaction (scatter of agreement vs mean |tau|)
4. Sign consensus aggregation vs standard mean
5. Pairwise sign agreement matrix between all task pairs
6. Layer-specific sign conflict profiles

Usage:
    uv run python scripts/analyze_sign_structure.py          # full analysis (8 tasks)
    uv run python scripts/analyze_sign_structure.py --quick   # quick mode (3 tasks)
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path so we can import model_merging
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "sign_analysis"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def classify_layer(layer_name: str) -> str:
    """Classify a layer name into a human-readable category."""
    if (
        "in_proj" in layer_name
        or "out_proj" in layer_name
        or "q_proj" in layer_name
        or "k_proj" in layer_name
        or "v_proj" in layer_name
    ):
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


def get_block_depth(layer_name: str) -> int:
    """Extract transformer block index from layer name. Returns -1 for non-block layers."""
    match = re.search(r"resblocks\.(\d+)", layer_name)
    return int(match.group(1)) if match else -1


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_state_dict(model_name, dataset_name="base"):
    """Load state dict directly from HF cache without creating ImageEncoder."""
    from huggingface_hub import hf_hub_download

    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_task_vectors(model_name, datasets):
    """Load pretrained model and compute task vectors for all datasets."""
    logger.info(f"Loading pretrained state dict: {model_name}")
    base_sd = load_state_dict(model_name, "base")

    task_vectors = {}
    for ds in datasets:
        logger.info(f"Loading finetuned state dict for {ds}")
        ft_sd = load_state_dict(model_name, ds)
        tv = compute_task_vector(base_sd, ft_sd, device="cpu")
        task_vectors[ds] = tv
        del ft_sd
        logger.info(
            f"  {ds}: {len(tv)} layers, total params = {sum(v.numel() for v in tv.values()):,}"
        )

    del base_sd
    return task_vectors


# --------------------------------------------------------------------------- #
# JSON serialization helper
# --------------------------------------------------------------------------- #


def make_json_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
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
# Analysis 1: Sign Agreement Score
# --------------------------------------------------------------------------- #


def compute_sign_agreement_all_params(task_vectors, datasets):
    """
    For each parameter position, compute sign agreement = max(n_pos, n_neg) / n_tasks.

    Returns:
        all_agreements: 1D numpy array of sign agreement scores for every parameter
        per_layer_agreements: dict mapping layer_name -> 1D numpy array of scores
        per_layer_magnitudes: dict mapping layer_name -> 1D numpy array of mean |tau|
    """
    logger.info("Computing per-parameter sign agreement scores...")
    n_tasks = len(datasets)
    all_layers = list(task_vectors[datasets[0]].keys())

    all_agreements = []
    all_magnitudes = []
    per_layer_agreements = {}
    per_layer_magnitudes = {}

    for layer_name in all_layers:
        tensors = [task_vectors[ds][layer_name].float().flatten() for ds in datasets]
        stacked = torch.stack(tensors, dim=0)  # (n_tasks, n_params)

        # Count positive signs per parameter position
        signs = torch.sign(stacked)  # -1, 0, or 1
        n_positive = (signs > 0).sum(dim=0).numpy().astype(np.float64)  # (n_params,)
        n_negative = (signs < 0).sum(dim=0).numpy().astype(np.float64)
        n_nonzero = n_positive + n_negative

        # Sign agreement: max(n_pos, n_neg) / n_tasks
        # For parameters where all tasks have value 0, agreement is 1.0
        agreement = np.where(
            n_nonzero > 0,
            np.maximum(n_positive, n_negative) / n_tasks,
            1.0,
        )

        # Mean absolute magnitude across tasks
        mean_abs_magnitude = stacked.abs().mean(dim=0).numpy()

        per_layer_agreements[layer_name] = agreement
        per_layer_magnitudes[layer_name] = mean_abs_magnitude

        all_agreements.append(agreement)
        all_magnitudes.append(mean_abs_magnitude)

    all_agreements = np.concatenate(all_agreements)
    all_magnitudes = np.concatenate(all_magnitudes)

    logger.info(f"  Total parameters analyzed: {len(all_agreements):,}")
    logger.info(f"  Mean sign agreement: {all_agreements.mean():.4f}")
    logger.info(f"  Median sign agreement: {np.median(all_agreements):.4f}")
    logger.info(f"  Fraction with perfect agreement (1.0): {(all_agreements == 1.0).mean():.4f}")
    logger.info(
        f"  Fraction with high conflict (<= 0.625): {(all_agreements <= 0.625).mean():.4f}"
    )

    return all_agreements, all_magnitudes, per_layer_agreements, per_layer_magnitudes


# --------------------------------------------------------------------------- #
# Analysis 2: Sign Agreement by Layer Type and Depth
# --------------------------------------------------------------------------- #


def analyze_sign_by_layer_and_depth(per_layer_agreements, datasets):
    """Break down sign agreement by layer category and block depth."""
    logger.info("Analyzing sign agreement by layer type and depth...")

    all_layers = list(per_layer_agreements.keys())

    # By layer type
    layer_type_stats = defaultdict(list)
    for layer_name in all_layers:
        lt = classify_layer(layer_name)
        mean_agreement = float(per_layer_agreements[layer_name].mean())
        layer_type_stats[lt].append(mean_agreement)

    logger.info("\n--- Sign Agreement by Layer Type ---")
    logger.info(f"{'Layer Type':15s}  {'Mean':>8s}  {'Std':>8s}  {'Count':>6s}")
    logger.info("-" * 45)
    layer_type_results = {}
    for lt in sorted(layer_type_stats.keys()):
        vals = layer_type_stats[lt]
        layer_type_results[lt] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "count": len(vals),
        }
        logger.info(f"{lt:15s}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  {len(vals):6d}")

    # By block depth
    depth_stats = defaultdict(list)
    for layer_name in all_layers:
        depth = get_block_depth(layer_name)
        mean_agreement = float(per_layer_agreements[layer_name].mean())
        depth_stats[depth].append(mean_agreement)

    logger.info("\n--- Sign Agreement by Block Depth ---")
    logger.info(f"{'Depth':>6s}  {'Mean':>8s}  {'Std':>8s}  {'Count':>6s}")
    logger.info("-" * 35)
    depth_results = {}
    for depth in sorted(depth_stats.keys()):
        vals = depth_stats[depth]
        label = str(depth) if depth >= 0 else "non-block"
        depth_results[label] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "count": len(vals),
        }
        logger.info(f"{label:>6s}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  {len(vals):6d}")

    # Detailed: by layer type AND depth (for heatmap)
    # Build a matrix: rows = depths (0..11), cols = layer types
    all_layer_types = sorted(set(classify_layer(ln) for ln in all_layers))
    all_depths = sorted(set(get_block_depth(ln) for ln in all_layers))

    depth_type_matrix = {}
    for layer_name in all_layers:
        lt = classify_layer(layer_name)
        depth = get_block_depth(layer_name)
        key = (depth, lt)
        if key not in depth_type_matrix:
            depth_type_matrix[key] = []
        depth_type_matrix[key].append(float(per_layer_agreements[layer_name].mean()))

    # Average within each cell
    heatmap_data = {}
    for (depth, lt), vals in depth_type_matrix.items():
        depth_label = str(depth) if depth >= 0 else "non-block"
        if depth_label not in heatmap_data:
            heatmap_data[depth_label] = {}
        heatmap_data[depth_label][lt] = float(np.mean(vals))

    return {
        "by_layer_type": layer_type_results,
        "by_depth": depth_results,
        "depth_type_heatmap": heatmap_data,
        "all_layer_types": all_layer_types,
        "all_depths": [str(d) if d >= 0 else "non-block" for d in all_depths],
    }


# --------------------------------------------------------------------------- #
# Analysis 3: Sign-Magnitude Interaction
# --------------------------------------------------------------------------- #


def analyze_sign_magnitude_interaction(per_layer_agreements, per_layer_magnitudes):
    """Compute correlation between sign agreement and parameter magnitude."""
    logger.info("Analyzing sign-magnitude interaction...")

    all_layers = list(per_layer_agreements.keys())

    # Per-layer summary: mean agreement vs mean magnitude
    layer_agreement_means = []
    layer_magnitude_means = []
    layer_names_out = []
    layer_types_out = []

    for layer_name in all_layers:
        agr = per_layer_agreements[layer_name]
        mag = per_layer_magnitudes[layer_name]
        layer_agreement_means.append(float(agr.mean()))
        layer_magnitude_means.append(float(mag.mean()))
        layer_names_out.append(layer_name)
        layer_types_out.append(classify_layer(layer_name))

    # Correlation at the parameter level (subsample for tractability)
    all_agr = np.concatenate([per_layer_agreements[ln] for ln in all_layers])
    all_mag = np.concatenate([per_layer_magnitudes[ln] for ln in all_layers])

    # Subsample if too many parameters
    n_total = len(all_agr)
    max_sample = 500_000
    if n_total > max_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, max_sample, replace=False)
        sample_agr = all_agr[idx]
        sample_mag = all_mag[idx]
    else:
        sample_agr = all_agr
        sample_mag = all_mag

    # Spearman correlation
    from scipy.stats import spearmanr

    corr, pval = spearmanr(sample_agr, sample_mag)
    logger.info(f"  Spearman correlation (agreement vs magnitude): {corr:.4f} (p={pval:.2e})")

    # Binned analysis: average magnitude at each agreement level
    n_tasks_possible = 8  # max tasks
    possible_agreements = sorted(set(all_agr))
    binned_magnitudes = {}
    for agr_val in possible_agreements:
        mask = all_agr == agr_val
        if mask.sum() > 0:
            binned_magnitudes[float(agr_val)] = {
                "mean_magnitude": float(all_mag[mask].mean()),
                "std_magnitude": float(all_mag[mask].std()),
                "count": int(mask.sum()),
            }

    return {
        "spearman_corr": float(corr),
        "spearman_pval": float(pval),
        "binned_magnitudes": binned_magnitudes,
        "layer_level": {
            "names": layer_names_out,
            "types": layer_types_out,
            "mean_agreements": layer_agreement_means,
            "mean_magnitudes": layer_magnitude_means,
        },
        "scatter_sample_agreement": sample_agr.tolist(),
        "scatter_sample_magnitude": sample_mag.tolist(),
    }


# --------------------------------------------------------------------------- #
# Analysis 4: Sign Consensus Aggregation
# --------------------------------------------------------------------------- #


def analyze_sign_consensus_aggregation(task_vectors, datasets):
    """
    Create a merged task vector using sign-consensus: for each parameter, take the
    mean only across tasks with the majority sign, zero out minority-sign contributions.
    Compare to the standard mean.
    """
    logger.info("Analyzing sign consensus aggregation...")

    n_tasks = len(datasets)
    all_layers = list(task_vectors[datasets[0]].keys())

    consensus_tv = OrderedDict()
    standard_mean_tv = OrderedDict()

    per_layer_comparison = {}

    for layer_name in all_layers:
        tensors = [task_vectors[ds][layer_name].float() for ds in datasets]
        stacked = torch.stack(tensors, dim=0)  # (n_tasks, *shape)

        # Standard mean
        standard_mean = stacked.mean(dim=0)

        # Sign consensus mean
        signs = torch.sign(stacked)
        sign_sum = signs.sum(dim=0)
        majority_sign = torch.sign(sign_sum)  # +1 or -1 where there's a majority, 0 for ties

        # For each parameter, keep only tasks agreeing with the majority sign
        # majority_sign: (*shape), stacked: (n_tasks, *shape)
        majority_mask = signs == majority_sign.unsqueeze(0)  # (n_tasks, *shape)

        # Mask out minority contributions
        masked_values = stacked * majority_mask.float()  # zero out minority
        n_majority = majority_mask.float().sum(dim=0).clamp(min=1)  # avoid div by 0
        consensus_mean = masked_values.sum(dim=0) / n_majority

        consensus_tv[layer_name] = consensus_mean
        standard_mean_tv[layer_name] = standard_mean

        # Compare norms
        consensus_norm = float(consensus_mean.norm())
        standard_norm = float(standard_mean.norm())
        cosine_sim = float(
            torch.nn.functional.cosine_similarity(
                consensus_mean.flatten().unsqueeze(0),
                standard_mean.flatten().unsqueeze(0),
            )
        )
        # Fraction of parameters zeroed out in consensus (those at tied positions)
        n_zeroed = int((majority_sign == 0).sum())
        n_total = int(majority_sign.numel())

        per_layer_comparison[layer_name] = {
            "consensus_norm": consensus_norm,
            "standard_mean_norm": standard_norm,
            "norm_ratio": consensus_norm / standard_norm if standard_norm > 0 else 0.0,
            "cosine_similarity": cosine_sim,
            "frac_tied_zeroed": n_zeroed / n_total if n_total > 0 else 0.0,
            "layer_type": classify_layer(layer_name),
        }

    # Global summary
    all_consensus_norms = [v["consensus_norm"] for v in per_layer_comparison.values()]
    all_standard_norms = [v["standard_mean_norm"] for v in per_layer_comparison.values()]
    all_cosines = [v["cosine_similarity"] for v in per_layer_comparison.values()]
    all_norm_ratios = [v["norm_ratio"] for v in per_layer_comparison.values()]

    summary = {
        "avg_norm_ratio": float(np.mean(all_norm_ratios)),
        "avg_cosine_similarity": float(np.mean(all_cosines)),
        "total_consensus_norm": float(
            np.sqrt(sum(v["consensus_norm"] ** 2 for v in per_layer_comparison.values()))
        ),
        "total_standard_norm": float(
            np.sqrt(sum(v["standard_mean_norm"] ** 2 for v in per_layer_comparison.values()))
        ),
    }

    logger.info(f"  Avg norm ratio (consensus / standard): {summary['avg_norm_ratio']:.4f}")
    logger.info(f"  Avg cosine similarity: {summary['avg_cosine_similarity']:.4f}")
    logger.info(f"  Total consensus norm: {summary['total_consensus_norm']:.4f}")
    logger.info(f"  Total standard norm: {summary['total_standard_norm']:.4f}")

    return {
        "summary": summary,
        "per_layer": per_layer_comparison,
    }


# --------------------------------------------------------------------------- #
# Analysis 5: Pairwise Sign Agreement Matrix
# --------------------------------------------------------------------------- #


def compute_pairwise_sign_agreement(task_vectors, datasets):
    """For each pair of tasks, compute fraction of parameters with same sign."""
    logger.info("Computing pairwise sign agreement matrix...")
    n_tasks = len(datasets)

    # Flatten all task vectors into single vectors
    flat_tvs = {}
    for ds in datasets:
        flat_tvs[ds] = torch.cat(
            [task_vectors[ds][k].float().flatten() for k in task_vectors[ds].keys()]
        )

    # Pairwise sign agreement
    agreement_matrix = np.eye(n_tasks)
    cosine_matrix = np.eye(n_tasks)

    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            if i >= j:
                continue
            v1 = flat_tvs[ds1]
            v2 = flat_tvs[ds2]

            # Sign agreement (excluding zeros)
            s1 = torch.sign(v1)
            s2 = torch.sign(v2)
            nonzero_mask = (s1 != 0) & (s2 != 0)
            if nonzero_mask.sum() > 0:
                agreement = float((s1[nonzero_mask] == s2[nonzero_mask]).float().mean())
            else:
                agreement = 0.5

            # Cosine similarity
            cos_sim = float(
                torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            )

            agreement_matrix[i, j] = agreement
            agreement_matrix[j, i] = agreement
            cosine_matrix[i, j] = cos_sim
            cosine_matrix[j, i] = cos_sim

    # Correlation between sign agreement and cosine similarity (off-diagonal)
    off_diag_agr = []
    off_diag_cos = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            off_diag_agr.append(agreement_matrix[i, j])
            off_diag_cos.append(cosine_matrix[i, j])

    from scipy.stats import pearsonr

    if len(off_diag_agr) >= 3:
        corr, pval = pearsonr(off_diag_agr, off_diag_cos)
    else:
        corr, pval = 0.0, 1.0

    logger.info(f"  Pairwise sign agreement range: [{min(off_diag_agr):.4f}, {max(off_diag_agr):.4f}]")
    logger.info(f"  Pairwise cosine sim range: [{min(off_diag_cos):.4f}, {max(off_diag_cos):.4f}]")
    logger.info(f"  Pearson correlation (sign agr vs cosine): {corr:.4f} (p={pval:.4f})")

    return {
        "sign_agreement_matrix": agreement_matrix.tolist(),
        "cosine_similarity_matrix": cosine_matrix.tolist(),
        "pearson_corr_sign_vs_cosine": float(corr),
        "pearson_pval": float(pval),
    }


# --------------------------------------------------------------------------- #
# Analysis 6: Layer-Specific Sign Conflict Profiles
# --------------------------------------------------------------------------- #


def compute_sign_conflict_profiles(per_layer_agreements):
    """For each layer, compute fraction of sign-conflicted parameters (agreement < 0.75)."""
    logger.info("Computing layer-specific sign conflict profiles...")

    all_layers = list(per_layer_agreements.keys())

    # Per-layer conflict fraction
    per_layer_conflict = {}
    for layer_name in all_layers:
        agr = per_layer_agreements[layer_name]
        frac_conflicted = float((agr < 0.75).mean())
        per_layer_conflict[layer_name] = {
            "frac_conflicted": frac_conflicted,
            "mean_agreement": float(agr.mean()),
            "layer_type": classify_layer(layer_name),
            "block_depth": get_block_depth(layer_name),
            "n_params": len(agr),
        }

    # Aggregate by block depth
    depth_conflict = defaultdict(list)
    for layer_name, info in per_layer_conflict.items():
        depth = info["block_depth"]
        depth_conflict[depth].append(info["frac_conflicted"])

    depth_conflict_summary = {}
    logger.info("\n--- Sign Conflict by Block Depth ---")
    logger.info(f"{'Depth':>6s}  {'Mean frac conflicted':>22s}  {'Std':>8s}")
    logger.info("-" * 42)
    for depth in sorted(depth_conflict.keys()):
        vals = depth_conflict[depth]
        label = str(depth) if depth >= 0 else "non-block"
        depth_conflict_summary[label] = {
            "mean_frac_conflicted": float(np.mean(vals)),
            "std_frac_conflicted": float(np.std(vals)),
            "n_layers": len(vals),
        }
        logger.info(f"{label:>6s}  {np.mean(vals):22.4f}  {np.std(vals):8.4f}")

    # Find the most conflicted layers
    sorted_layers = sorted(per_layer_conflict.items(), key=lambda x: -x[1]["frac_conflicted"])
    logger.info("\n--- Top 10 Most Sign-Conflicted Layers ---")
    for layer_name, info in sorted_layers[:10]:
        logger.info(
            f"  {layer_name:<55s}  conflict={info['frac_conflicted']:.4f}  "
            f"type={info['layer_type']}  depth={info['block_depth']}"
        )

    return {
        "per_layer": per_layer_conflict,
        "by_depth": depth_conflict_summary,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_sign_agreement_histogram(all_agreements, output_dir, n_tasks):
    """Histogram of sign agreement scores across all parameters."""
    logger.info("Plotting sign agreement histogram...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Possible discrete values: for n_tasks tasks, agreement = max(k, n-k)/n for k=0..n
    # E.g. for 8 tasks: 0.5, 0.625, 0.75, 0.875, 1.0
    possible_values = sorted(set(np.round(all_agreements, decimals=6)))

    counts = []
    labels = []
    for val in possible_values:
        count = (np.abs(all_agreements - val) < 1e-5).sum()
        counts.append(count)
        labels.append(f"{val:.3f}")

    bars = ax.bar(range(len(possible_values)), counts, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(range(len(possible_values)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Sign Agreement Score (max(n_pos, n_neg) / n_tasks)", fontsize=12)
    ax.set_ylabel("Number of Parameters", fontsize=12)
    ax.set_title(f"Distribution of Sign Agreement Across All Parameters\n(n_tasks={n_tasks})", fontsize=13)

    # Add percentage annotations on bars
    total = len(all_agreements)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sign_agreement_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sign_agreement_by_layer(layer_depth_results, per_layer_agreements, output_dir):
    """Heatmap: layers x sign agreement level (fraction of params at each level)."""
    logger.info("Plotting sign agreement by layer heatmap...")

    all_layers = list(per_layer_agreements.keys())
    # Only include layers within transformer blocks for readability
    block_layers = [ln for ln in all_layers if get_block_depth(ln) >= 0]

    # Get all possible agreement levels
    all_agr = np.concatenate([per_layer_agreements[ln] for ln in block_layers])
    possible_values = sorted(set(np.round(all_agr, decimals=6)))

    # Build matrix: rows = layers, cols = agreement levels
    matrix = np.zeros((len(block_layers), len(possible_values)))
    for i, layer_name in enumerate(block_layers):
        agr = per_layer_agreements[layer_name]
        n_total = len(agr)
        for j, val in enumerate(possible_values):
            matrix[i, j] = (np.abs(agr - val) < 1e-5).sum() / n_total

    # Shorten layer names
    short_names = []
    for name in block_layers:
        parts = name.split(".")
        # e.g., visual.transformer.resblocks.0.attn.in_proj_weight -> blk0.attn.in_proj_weight
        depth = get_block_depth(name)
        suffix = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_names.append(f"b{depth}.{suffix}")

    fig, ax = plt.subplots(figsize=(12, max(10, len(block_layers) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(possible_values)))
    ax.set_xticklabels([f"{v:.3f}" for v in possible_values], fontsize=9)
    ax.set_yticks(range(len(block_layers)))
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel("Sign Agreement Score", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Fraction of Parameters at Each Sign Agreement Level", fontsize=13)

    plt.colorbar(im, ax=ax, label="Fraction of params", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "sign_agreement_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sign_magnitude_scatter(sign_mag_results, output_dir):
    """Scatter plot: sign agreement vs mean |tau|."""
    logger.info("Plotting sign-magnitude scatter...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Parameter-level scatter (subsampled)
    ax = axes[0]
    sample_agr = np.array(sign_mag_results["scatter_sample_agreement"])
    sample_mag = np.array(sign_mag_results["scatter_sample_magnitude"])

    ax.scatter(sample_agr, sample_mag, alpha=0.02, s=1, color="steelblue", rasterized=True)

    # Add binned means
    binned = sign_mag_results["binned_magnitudes"]
    bin_agr = sorted(binned.keys())
    bin_mag_mean = [binned[k]["mean_magnitude"] for k in bin_agr]
    bin_mag_std = [binned[k]["std_magnitude"] for k in bin_agr]
    ax.errorbar(
        bin_agr,
        bin_mag_mean,
        yerr=bin_mag_std,
        fmt="ro-",
        markersize=8,
        linewidth=2,
        label="Binned mean +/- std",
        zorder=5,
    )

    corr = sign_mag_results["spearman_corr"]
    ax.set_xlabel("Sign Agreement", fontsize=11)
    ax.set_ylabel("Mean |task vector|", fontsize=11)
    ax.set_title(f"Parameter-level (Spearman r={corr:.4f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Layer-level scatter, colored by layer type
    ax = axes[1]
    layer_info = sign_mag_results["layer_level"]
    types = layer_info["types"]
    agr = layer_info["mean_agreements"]
    mag = layer_info["mean_magnitudes"]

    unique_types = sorted(set(types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    type_to_color = dict(zip(unique_types, colors))

    for lt in unique_types:
        mask = [t == lt for t in types]
        lt_agr = [a for a, m in zip(agr, mask) if m]
        lt_mag = [mg for mg, m in zip(mag, mask) if m]
        ax.scatter(lt_agr, lt_mag, c=[type_to_color[lt]], label=lt, alpha=0.7, s=40, edgecolors="k", linewidth=0.3)

    ax.set_xlabel("Mean Sign Agreement", fontsize=11)
    ax.set_ylabel("Mean |task vector|", fontsize=11)
    ax.set_title("Layer-level", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Sign Agreement vs Parameter Magnitude", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "sign_magnitude_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pairwise_sign_agreement(pairwise_results, datasets, output_dir):
    """8x8 heatmap of pairwise sign agreement between tasks."""
    logger.info("Plotting pairwise sign agreement heatmap...")

    n_tasks = len(datasets)
    agreement_matrix = np.array(pairwise_results["sign_agreement_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Sign agreement
    ax = axes[0]
    im = ax.imshow(agreement_matrix, cmap="RdYlGn", vmin=0.4, vmax=0.7)
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(datasets, fontsize=9)

    for i in range(n_tasks):
        for j in range(n_tasks):
            text_color = "white" if agreement_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{agreement_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Fraction of same-sign parameters", shrink=0.8)
    ax.set_title("Pairwise Sign Agreement", fontsize=12)

    # Panel 2: Cosine similarity for comparison
    ax = axes[1]
    cosine_matrix = np.array(pairwise_results["cosine_similarity_matrix"])
    im2 = ax.imshow(cosine_matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.3)
    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(datasets, fontsize=9)

    for i in range(n_tasks):
        for j in range(n_tasks):
            text_color = "white" if cosine_matrix[i, j] > 0.15 else "black"
            ax.text(j, i, f"{cosine_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    plt.colorbar(im2, ax=ax, label="Cosine similarity", shrink=0.8)
    ax.set_title("Pairwise Cosine Similarity (for comparison)", fontsize=12)

    corr = pairwise_results["pearson_corr_sign_vs_cosine"]
    fig.suptitle(
        f"Pairwise Task Vector Comparisons (Pearson r(sign,cosine) = {corr:.4f})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pairwise_sign_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sign_conflict_by_depth(conflict_results, output_dir):
    """Bar chart: fraction of sign-conflicted params by transformer block depth."""
    logger.info("Plotting sign conflict by depth...")

    depth_data = conflict_results["by_depth"]

    # Separate block depths from non-block
    block_depths = []
    non_block_val = None
    for label, info in depth_data.items():
        if label == "non-block":
            non_block_val = info
        else:
            block_depths.append((int(label), info))

    block_depths.sort(key=lambda x: x[0])

    labels = [str(d) for d, _ in block_depths]
    means = [info["mean_frac_conflicted"] for _, info in block_depths]
    stds = [info["std_frac_conflicted"] for _, info in block_depths]

    if non_block_val is not None:
        labels.append("non-block")
        means.append(non_block_val["mean_frac_conflicted"])
        stds.append(non_block_val["std_frac_conflicted"])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="coral", edgecolor="darkred", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Transformer Block Depth", fontsize=12)
    ax.set_ylabel("Fraction of Sign-Conflicted Parameters\n(agreement < 0.75)", fontsize=11)
    ax.set_title("Sign Conflict by Transformer Block Depth", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "sign_conflict_by_depth.png", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sign patterns of task vectors from N8 ViT-B-32 models"
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = N8_DATASETS[:3] if args.quick else N8_DATASETS
    if args.quick:
        logger.info(f"QUICK MODE: Using {len(datasets)} datasets: {datasets}")
    else:
        logger.info(f"Full analysis: {len(datasets)} datasets: {datasets}")

    # Load task vectors
    task_vectors = load_task_vectors(MODEL_NAME, datasets)

    all_results = {}
    json_path = OUTPUT_DIR / "sign_analysis.json"

    def save_intermediate():
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        logger.info(f"Intermediate results saved to: {json_path}")

    # ---- Analysis 1: Sign agreement scores ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 1: Sign Agreement Score Distribution")
    logger.info("=" * 60)
    (
        all_agreements,
        all_magnitudes,
        per_layer_agreements,
        per_layer_magnitudes,
    ) = compute_sign_agreement_all_params(task_vectors, datasets)

    all_results["sign_agreement_summary"] = {
        "mean": float(all_agreements.mean()),
        "median": float(np.median(all_agreements)),
        "std": float(all_agreements.std()),
        "frac_perfect_agreement": float((all_agreements == 1.0).mean()),
        "frac_high_conflict_leq_0.625": float((all_agreements <= 0.625).mean()),
        "frac_conflict_lt_0.75": float((all_agreements < 0.75).mean()),
        "n_params": int(len(all_agreements)),
    }
    save_intermediate()

    # ---- Analysis 2: Sign agreement by layer type and depth ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Sign Agreement by Layer Type and Depth")
    logger.info("=" * 60)
    layer_depth_results = analyze_sign_by_layer_and_depth(per_layer_agreements, datasets)
    all_results["sign_by_layer_and_depth"] = layer_depth_results
    save_intermediate()

    # ---- Analysis 3: Sign-magnitude interaction ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Sign-Magnitude Interaction")
    logger.info("=" * 60)
    sign_mag_results = analyze_sign_magnitude_interaction(per_layer_agreements, per_layer_magnitudes)
    # Don't save the large scatter sample arrays in JSON
    sign_mag_results_for_json = {
        k: v
        for k, v in sign_mag_results.items()
        if k not in ("scatter_sample_agreement", "scatter_sample_magnitude")
    }
    all_results["sign_magnitude_interaction"] = sign_mag_results_for_json
    save_intermediate()

    # ---- Analysis 4: Sign consensus aggregation ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: Sign Consensus Aggregation")
    logger.info("=" * 60)
    consensus_results = analyze_sign_consensus_aggregation(task_vectors, datasets)
    all_results["sign_consensus_aggregation"] = consensus_results
    save_intermediate()

    # ---- Analysis 5: Pairwise sign agreement matrix ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 5: Pairwise Sign Agreement Matrix")
    logger.info("=" * 60)
    pairwise_results = compute_pairwise_sign_agreement(task_vectors, datasets)
    all_results["pairwise_sign_agreement"] = pairwise_results
    save_intermediate()

    # ---- Analysis 6: Sign conflict profiles ----
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 6: Layer-Specific Sign Conflict Profiles")
    logger.info("=" * 60)
    conflict_results = compute_sign_conflict_profiles(per_layer_agreements)
    all_results["sign_conflict_profiles"] = conflict_results
    save_intermediate()

    # ---- Generate plots ----
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 60)

    plot_sign_agreement_histogram(all_agreements, OUTPUT_DIR, len(datasets))
    plot_sign_agreement_by_layer(layer_depth_results, per_layer_agreements, OUTPUT_DIR)
    plot_sign_magnitude_scatter(sign_mag_results, OUTPUT_DIR)
    plot_pairwise_sign_agreement(pairwise_results, datasets, OUTPUT_DIR)
    plot_sign_conflict_by_depth(conflict_results, OUTPUT_DIR)

    logger.info(f"\nAll results saved to: {json_path}")
    logger.info(f"All plots saved to: {OUTPUT_DIR}")
    logger.info("Sign structure analysis complete.")


if __name__ == "__main__":
    main()
