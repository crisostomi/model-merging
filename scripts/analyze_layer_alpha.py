"""
Layer-wise alpha analysis for model merging.

Investigates whether different layers should have different scaling factors
(alpha) when merging task vectors. Currently, merging uses a single scalar:

    theta_merged = theta_pretrained + alpha * merged_task_vector

This script analyzes whether a layer-dependent alpha would be beneficial.

Analyses performed:
1. Task vector magnitude profile: ||tau_layer|| / ||theta_pretrained_layer||
2. Task vector agreement profile: pairwise cosine similarity per layer
3. Interference estimation: how much of each task survives summation
4. Fisher-weighted analysis (approximation via spectral properties)
5. Cross-layer consistency: systematic patterns across model depth

Usage:
    uv run python scripts/analyze_layer_alpha.py          # full analysis (8 tasks)
    uv run python scripts/analyze_layer_alpha.py --quick   # quick mode (3 tasks)
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

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.merging.task_vectors import compute_task_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
MODEL_NAME = "ViT-B-32"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_state_dict(model_name, dataset_name="base"):
    """Load state dict directly from HuggingFace without creating ImageEncoder."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(f"crisostomi/{model_name}-{dataset_name}", "pytorch_model.bin")
    return torch.load(path, map_location="cpu")


def extract_block_index(layer_name: str):
    """Extract the transformer block index from a layer name, or None if not block-specific."""
    match = re.search(r"resblocks\.(\d+)\.", layer_name)
    if match:
        return int(match.group(1))
    return None


def classify_layer(layer_name: str) -> str:
    """Classify a layer name into a human-readable category."""
    if "in_proj" in layer_name or "out_proj" in layer_name:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pretrained_and_task_vectors(datasets, model_name=MODEL_NAME):
    """Load pretrained state dict and compute task vectors for all datasets.

    Returns:
        pretrained: the pretrained state dict (kept for norm-ratio analysis)
        task_vectors: dict of {dataset_name: OrderedDict of task vectors}
    """
    logger.info(f"Loading pretrained state dict: {model_name}-base")
    pretrained = load_state_dict(model_name, "base")

    task_vectors = {}
    for ds in datasets:
        logger.info(f"Loading finetuned state dict for {ds}")
        finetuned = load_state_dict(model_name, ds)
        tv = compute_task_vector(pretrained, finetuned, device="cpu")
        task_vectors[ds] = tv
        del finetuned
        logger.info(
            f"  {ds}: {len(tv)} layers, "
            f"total params = {sum(v.numel() for v in tv.values()):,}"
        )

    return pretrained, task_vectors


# ---------------------------------------------------------------------------
# Analysis 1: Task Vector Magnitude Profile
# ---------------------------------------------------------------------------

def analyze_magnitude_profile(pretrained, task_vectors, datasets, layer_names):
    """For each layer, compute ||tau|| / ||theta_pretrained|| across tasks.

    Layers with large ratios were heavily modified during fine-tuning;
    layers with small ratios barely changed.
    """
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: Task Vector Magnitude Profile")
    logger.info("=" * 60)

    results = {}

    for layer_name in layer_names:
        pretrained_norm = float(torch.norm(pretrained[layer_name].float(), p="fro"))
        ratios = {}
        for ds in datasets:
            tv_norm = float(torch.norm(task_vectors[ds][layer_name].float(), p="fro"))
            ratio = tv_norm / pretrained_norm if pretrained_norm > 1e-12 else 0.0
            ratios[ds] = ratio

        avg_ratio = float(np.mean(list(ratios.values())))
        std_ratio = float(np.std(list(ratios.values())))

        results[layer_name] = {
            "per_task_ratios": ratios,
            "pretrained_norm": pretrained_norm,
            "avg_ratio": avg_ratio,
            "std_ratio": std_ratio,
            "block_index": extract_block_index(layer_name),
            "layer_type": classify_layer(layer_name),
        }

    # Print summary table
    logger.info(f"\n{'Layer':<55s}  {'Avg Ratio':>10s}  {'Std':>8s}  {'Block':>5s}")
    logger.info("-" * 85)
    for layer_name in layer_names:
        r = results[layer_name]
        block_str = str(r["block_index"]) if r["block_index"] is not None else "-"
        logger.info(
            f"{layer_name:<55s}  {r['avg_ratio']:10.5f}  {r['std_ratio']:8.5f}  {block_str:>5s}"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Task Vector Agreement Profile
# ---------------------------------------------------------------------------

def analyze_agreement_profile(task_vectors, datasets, layer_names):
    """For each layer, compute pairwise cosine similarity between task vectors.

    High agreement = safe to use large alpha.
    Low agreement = should use smaller alpha to avoid interference.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Task Vector Agreement Profile")
    logger.info("=" * 60)

    results = {}

    for layer_name in layer_names:
        # Flatten each task vector for this layer
        flat_tvs = {
            ds: task_vectors[ds][layer_name].float().flatten()
            for ds in datasets
        }

        pairwise_sims = {}
        sim_values = []
        for ds_a, ds_b in combinations(datasets, 2):
            va = flat_tvs[ds_a]
            vb = flat_tvs[ds_b]
            norm_a = torch.norm(va)
            norm_b = torch.norm(vb)
            if norm_a < 1e-12 or norm_b < 1e-12:
                sim = 0.0
            else:
                sim = float(torch.dot(va, vb) / (norm_a * norm_b))
            pair_key = f"{ds_a}-{ds_b}"
            pairwise_sims[pair_key] = sim
            sim_values.append(sim)

        avg_sim = float(np.mean(sim_values)) if sim_values else 0.0
        std_sim = float(np.std(sim_values)) if sim_values else 0.0
        min_sim = float(np.min(sim_values)) if sim_values else 0.0
        max_sim = float(np.max(sim_values)) if sim_values else 0.0

        results[layer_name] = {
            "pairwise_cosine_sims": pairwise_sims,
            "avg_cosine_sim": avg_sim,
            "std_cosine_sim": std_sim,
            "min_cosine_sim": min_sim,
            "max_cosine_sim": max_sim,
            "block_index": extract_block_index(layer_name),
            "layer_type": classify_layer(layer_name),
        }

    # Print summary
    logger.info(f"\n{'Layer':<55s}  {'Avg CosSim':>10s}  {'Min':>8s}  {'Max':>8s}")
    logger.info("-" * 85)
    for layer_name in layer_names:
        r = results[layer_name]
        logger.info(
            f"{layer_name:<55s}  {r['avg_cosine_sim']:10.5f}  "
            f"{r['min_cosine_sim']:8.5f}  {r['max_cosine_sim']:8.5f}"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Interference Estimation
# ---------------------------------------------------------------------------

def analyze_interference(task_vectors, datasets, layer_names):
    """For each layer, estimate interference when summing task vectors.

    Procedure:
    - Sum all task vectors: sum_tv = sum_i(tau_i)
    - For each task i, compute cos(tau_i, sum_tv) = "retained fraction"
    - Average retained fraction = layer interference score
    - Low score => high interference (task vectors cancel out)
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Interference Estimation")
    logger.info("=" * 60)

    results = {}

    for layer_name in layer_names:
        flat_tvs = [task_vectors[ds][layer_name].float().flatten() for ds in datasets]

        # Sum of all task vectors
        sum_tv = torch.stack(flat_tvs, dim=0).sum(dim=0)
        sum_norm = torch.norm(sum_tv)

        retained_fractions = {}
        for i, ds in enumerate(datasets):
            tv_norm = torch.norm(flat_tvs[i])
            if tv_norm < 1e-12 or sum_norm < 1e-12:
                retained = 0.0
            else:
                retained = float(torch.dot(flat_tvs[i], sum_tv) / (tv_norm * sum_norm))
            retained_fractions[ds] = retained

        avg_retained = float(np.mean(list(retained_fractions.values())))

        # Also compute the "magnitude loss": ||sum|| / sum(||tau_i||)
        # A value of 1.0 means no cancellation; lower values mean more interference
        individual_norms_sum = sum(float(torch.norm(tv)) for tv in flat_tvs)
        magnitude_ratio = float(sum_norm / individual_norms_sum) if individual_norms_sum > 1e-12 else 0.0

        results[layer_name] = {
            "retained_fractions": retained_fractions,
            "avg_retained_fraction": avg_retained,
            "magnitude_ratio": magnitude_ratio,
            "block_index": extract_block_index(layer_name),
            "layer_type": classify_layer(layer_name),
        }

    # Print summary
    logger.info(f"\n{'Layer':<55s}  {'Avg Retained':>12s}  {'Mag Ratio':>10s}")
    logger.info("-" * 80)
    for layer_name in layer_names:
        r = results[layer_name]
        logger.info(
            f"{layer_name:<55s}  {r['avg_retained_fraction']:12.5f}  "
            f"{r['magnitude_ratio']:10.5f}"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Fisher-Weighted Analysis (Approximation)
# ---------------------------------------------------------------------------

def analyze_fisher_proxy(task_vectors, datasets, layer_names):
    """Approximate layer importance via spectral properties (no gradients needed).

    Metrics:
    - spectral_norm: top singular value (magnitude of dominant change direction)
    - frobenius_norm: total magnitude of change
    - concentration_ratio: spectral_norm / frobenius_norm
      High = change is concentrated in few directions (signal-like)
      Low = change is diffuse (noise-like)
    - top_sv_fraction: fraction of energy in top-k SVs (k = min(10, rank//4))

    Together these suggest which layers carry the most "signal" vs "noise",
    helping inform per-layer alpha choices.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: Fisher-Weighted Analysis (Proxy)")
    logger.info("=" * 60)

    results = {}

    for layer_name in layer_names:
        layer_results = {}
        for ds in datasets:
            tensor = task_vectors[ds][layer_name].float()

            if tensor.dim() == 2:
                svs = torch.linalg.svdvals(tensor)
                spectral_norm = float(svs[0])
                frob_norm = float(torch.norm(tensor, p="fro"))
                concentration = spectral_norm / frob_norm if frob_norm > 1e-12 else 0.0

                # Top-k SV energy fraction
                k = min(10, max(1, svs.shape[0] // 4))
                total_energy = float((svs ** 2).sum())
                topk_energy = float((svs[:k] ** 2).sum())
                topk_fraction = topk_energy / total_energy if total_energy > 1e-12 else 0.0

                layer_results[ds] = {
                    "spectral_norm": spectral_norm,
                    "frobenius_norm": frob_norm,
                    "concentration_ratio": concentration,
                    "topk_sv_energy_fraction": topk_fraction,
                    "k_used": k,
                    "total_rank": int(svs.shape[0]),
                }
            else:
                # 1D tensor (bias, layernorm params)
                frob_norm = float(torch.norm(tensor))
                layer_results[ds] = {
                    "spectral_norm": frob_norm,
                    "frobenius_norm": frob_norm,
                    "concentration_ratio": 1.0,
                    "topk_sv_energy_fraction": 1.0,
                    "k_used": 1,
                    "total_rank": int(tensor.shape[0]) if tensor.dim() >= 1 else 1,
                }

        # Averages across tasks
        avg_concentration = float(np.mean([
            layer_results[ds]["concentration_ratio"] for ds in datasets
        ]))
        avg_topk_frac = float(np.mean([
            layer_results[ds]["topk_sv_energy_fraction"] for ds in datasets
        ]))
        avg_spectral = float(np.mean([
            layer_results[ds]["spectral_norm"] for ds in datasets
        ]))

        results[layer_name] = {
            "per_task": layer_results,
            "avg_concentration_ratio": avg_concentration,
            "avg_topk_sv_energy_fraction": avg_topk_frac,
            "avg_spectral_norm": avg_spectral,
            "block_index": extract_block_index(layer_name),
            "layer_type": classify_layer(layer_name),
        }

    # Print summary
    logger.info(
        f"\n{'Layer':<55s}  {'Avg Conc':>9s}  {'Avg TopK%':>10s}  {'Avg Spec':>9s}"
    )
    logger.info("-" * 88)
    for layer_name in layer_names:
        r = results[layer_name]
        logger.info(
            f"{layer_name:<55s}  {r['avg_concentration_ratio']:9.5f}  "
            f"{r['avg_topk_sv_energy_fraction']:10.5f}  {r['avg_spectral_norm']:9.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Cross-Layer Consistency (Block-level aggregation)
# ---------------------------------------------------------------------------

def analyze_cross_layer_consistency(
    magnitude_results, agreement_results, interference_results, fisher_results, layer_names
):
    """Aggregate metrics by transformer block to reveal depth-dependent patterns.

    Groups layers by their block index and computes per-block statistics for
    all four preceding analyses.
    """
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 5: Cross-Layer Consistency (Block-Level)")
    logger.info("=" * 60)

    # Group layers by block
    block_layers = defaultdict(list)
    non_block_layers = []
    for layer_name in layer_names:
        block_idx = extract_block_index(layer_name)
        if block_idx is not None:
            block_layers[block_idx].append(layer_name)
        else:
            non_block_layers.append(layer_name)

    results = {"per_block": {}, "non_block": {}}

    # Compute per-block averages
    sorted_blocks = sorted(block_layers.keys())
    logger.info(
        f"\n{'Block':>5s}  {'Avg MagRatio':>12s}  {'Avg CosSim':>10s}  "
        f"{'Avg Retained':>12s}  {'Avg Conc':>9s}  {'Layers':>6s}"
    )
    logger.info("-" * 62)

    for block_idx in sorted_blocks:
        layers = block_layers[block_idx]
        mag_ratios = [magnitude_results[ln]["avg_ratio"] for ln in layers]
        cos_sims = [agreement_results[ln]["avg_cosine_sim"] for ln in layers]
        retained = [interference_results[ln]["avg_retained_fraction"] for ln in layers]
        conc = [fisher_results[ln]["avg_concentration_ratio"] for ln in layers]

        block_data = {
            "avg_magnitude_ratio": float(np.mean(mag_ratios)),
            "avg_cosine_sim": float(np.mean(cos_sims)),
            "avg_retained_fraction": float(np.mean(retained)),
            "avg_concentration_ratio": float(np.mean(conc)),
            "num_layers": len(layers),
            "layer_names": layers,
        }
        results["per_block"][int(block_idx)] = block_data

        logger.info(
            f"{block_idx:5d}  {block_data['avg_magnitude_ratio']:12.5f}  "
            f"{block_data['avg_cosine_sim']:10.5f}  "
            f"{block_data['avg_retained_fraction']:12.5f}  "
            f"{block_data['avg_concentration_ratio']:9.5f}  "
            f"{len(layers):6d}"
        )

    # Non-block layers summary
    if non_block_layers:
        mag_ratios = [magnitude_results[ln]["avg_ratio"] for ln in non_block_layers]
        cos_sims = [agreement_results[ln]["avg_cosine_sim"] for ln in non_block_layers]
        retained = [interference_results[ln]["avg_retained_fraction"] for ln in non_block_layers]
        conc = [fisher_results[ln]["avg_concentration_ratio"] for ln in non_block_layers]

        results["non_block"] = {
            "avg_magnitude_ratio": float(np.mean(mag_ratios)),
            "avg_cosine_sim": float(np.mean(cos_sims)),
            "avg_retained_fraction": float(np.mean(retained)),
            "avg_concentration_ratio": float(np.mean(conc)),
            "num_layers": len(non_block_layers),
            "layer_names": non_block_layers,
        }
        logger.info(
            f"{'non-blk':>5s}  {results['non_block']['avg_magnitude_ratio']:12.5f}  "
            f"{results['non_block']['avg_cosine_sim']:10.5f}  "
            f"{results['non_block']['avg_retained_fraction']:12.5f}  "
            f"{results['non_block']['avg_concentration_ratio']:9.5f}  "
            f"{len(non_block_layers):6d}"
        )

    # Log pattern observations
    if len(sorted_blocks) >= 3:
        early = sorted_blocks[:len(sorted_blocks) // 3]
        mid = sorted_blocks[len(sorted_blocks) // 3 : 2 * len(sorted_blocks) // 3]
        late = sorted_blocks[2 * len(sorted_blocks) // 3:]

        for group_name, group_blocks in [("early", early), ("mid", mid), ("late", late)]:
            group_mag = np.mean([
                results["per_block"][b]["avg_magnitude_ratio"] for b in group_blocks
            ])
            group_sim = np.mean([
                results["per_block"][b]["avg_cosine_sim"] for b in group_blocks
            ])
            group_ret = np.mean([
                results["per_block"][b]["avg_retained_fraction"] for b in group_blocks
            ])
            logger.info(
                f"  {group_name:5s} blocks ({group_blocks}): "
                f"mag_ratio={group_mag:.5f}, cos_sim={group_sim:.5f}, retained={group_ret:.5f}"
            )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_block_and_type_labels(layer_names):
    """Create short labels for x-axis: 'B{block}.{type}' or just the short name."""
    labels = []
    for name in layer_names:
        block = extract_block_index(name)
        ltype = classify_layer(name)
        if block is not None:
            # Abbreviate layer type
            type_abbr = {
                "attention": "attn",
                "mlp_fc": "fc",
                "mlp_proj": "proj",
                "layernorm": "ln",
            }.get(ltype, ltype[:4])
            # Include weight/bias distinction if present
            suffix = ""
            if ".weight" in name:
                suffix = ".w"
            elif ".bias" in name:
                suffix = ".b"
            labels.append(f"B{block}.{type_abbr}{suffix}")
        else:
            parts = name.split(".")
            labels.append(".".join(parts[-2:]) if len(parts) > 1 else name)
    return labels


def plot_magnitude_profile(magnitude_results, layer_names, datasets, output_dir):
    """Plot norm ratios ||tau|| / ||theta_pretrained|| across layers."""
    logger.info("Plotting magnitude profile...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})

    labels = _get_block_and_type_labels(layer_names)
    x = np.arange(len(layer_names))

    # Top: per-task ratios
    ax = axes[0]
    for ds in datasets:
        values = [magnitude_results[ln]["per_task_ratios"][ds] for ln in layer_names]
        ax.plot(x, values, "o-", markersize=3, alpha=0.6, label=ds)

    # Overlay average
    avg_values = [magnitude_results[ln]["avg_ratio"] for ln in layer_names]
    ax.plot(x, avg_values, "k-", linewidth=2.0, alpha=0.9, label="Average")

    ax.set_ylabel("||task_vector|| / ||pretrained||")
    ax.set_title("Task Vector Magnitude Relative to Pretrained Weights")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Bottom: avg +/- std
    ax2 = axes[1]
    stds = [magnitude_results[ln]["std_ratio"] for ln in layer_names]
    ax2.bar(x, avg_values, yerr=stds, color="steelblue", alpha=0.7, capsize=2)
    ax2.set_ylabel("Avg ratio")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_alpha_magnitude_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir / 'layer_alpha_magnitude_profile.png'}")


def plot_agreement_heatmap(agreement_results, layer_names, datasets, output_dir):
    """Plot pairwise cosine similarity heatmap per block."""
    logger.info("Plotting agreement heatmap...")

    # Group layers by block
    block_layers = defaultdict(list)
    non_block = []
    for ln in layer_names:
        bi = extract_block_index(ln)
        if bi is not None:
            block_layers[bi].append(ln)
        else:
            non_block.append(ln)

    # Determine grid layout: one heatmap per block (+ 1 for non-block if any)
    sorted_blocks = sorted(block_layers.keys())
    n_panels = len(sorted_blocks)
    if non_block:
        n_panels += 1

    # Limit to a reasonable grid
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    pairs = list(combinations(datasets, 2))
    n_tasks = len(datasets)

    panel_idx = 0
    for block_idx in sorted_blocks:
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]

        block_ln = block_layers[block_idx]

        # Build average pairwise sim matrix across layers in this block
        sim_matrix = np.zeros((n_tasks, n_tasks))
        for ln in block_ln:
            for (ds_a, ds_b) in pairs:
                pair_key = f"{ds_a}-{ds_b}"
                sim_val = agreement_results[ln]["pairwise_cosine_sims"].get(pair_key, 0.0)
                ia = datasets.index(ds_a)
                ib = datasets.index(ds_b)
                sim_matrix[ia, ib] += sim_val
                sim_matrix[ib, ia] += sim_val

        # Average over layers in this block
        if len(block_ln) > 0:
            sim_matrix /= len(block_ln)

        # Diagonal = 1
        np.fill_diagonal(sim_matrix, 1.0)

        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels([ds[:5] for ds in datasets], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([ds[:5] for ds in datasets], fontsize=7)
        ax.set_title(f"Block {block_idx}", fontsize=10)

        # Annotate
        for i in range(n_tasks):
            for j in range(n_tasks):
                color = "white" if abs(sim_matrix[i, j]) > 0.15 else "black"
                ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

        panel_idx += 1

    # Handle non-block layers if any
    if non_block and panel_idx < nrows * ncols:
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        sim_matrix = np.zeros((n_tasks, n_tasks))
        for ln in non_block:
            for (ds_a, ds_b) in pairs:
                pair_key = f"{ds_a}-{ds_b}"
                sim_val = agreement_results[ln]["pairwise_cosine_sims"].get(pair_key, 0.0)
                ia = datasets.index(ds_a)
                ib = datasets.index(ds_b)
                sim_matrix[ia, ib] += sim_val
                sim_matrix[ib, ia] += sim_val
        if len(non_block) > 0:
            sim_matrix /= len(non_block)
        np.fill_diagonal(sim_matrix, 1.0)
        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(n_tasks))
        ax.set_yticks(range(n_tasks))
        ax.set_xticklabels([ds[:5] for ds in datasets], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([ds[:5] for ds in datasets], fontsize=7)
        ax.set_title("Non-block layers", fontsize=10)
        panel_idx += 1

    # Hide unused panels
    for idx in range(panel_idx, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    fig.suptitle("Pairwise Task Vector Cosine Similarity by Block", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine similarity")
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_dir / "layer_alpha_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir / 'layer_alpha_agreement.png'}")


def plot_interference(interference_results, layer_names, datasets, output_dir):
    """Plot interference scores across layers."""
    logger.info("Plotting interference scores...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})
    labels = _get_block_and_type_labels(layer_names)
    x = np.arange(len(layer_names))

    # Top: retained fraction per task
    ax = axes[0]
    for ds in datasets:
        values = [interference_results[ln]["retained_fractions"][ds] for ln in layer_names]
        ax.plot(x, values, "o-", markersize=3, alpha=0.6, label=ds)

    avg_retained = [interference_results[ln]["avg_retained_fraction"] for ln in layer_names]
    ax.plot(x, avg_retained, "k-", linewidth=2.0, alpha=0.9, label="Average")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("cos(tau_i, sum(tau_j))")
    ax.set_title("Retained Fraction per Task (higher = less interference)")
    ax.legend(fontsize=7, ncol=5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Bottom: magnitude ratio
    ax2 = axes[1]
    mag_ratios = [interference_results[ln]["magnitude_ratio"] for ln in layer_names]
    colors = ["green" if r > 0.5 else "orange" if r > 0.3 else "red" for r in mag_ratios]
    ax2.bar(x, mag_ratios, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No cancellation")
    ax2.set_ylabel("||sum(tau)|| / sum(||tau_i||)")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=5)
    ax2.set_title("Magnitude Ratio (1.0 = no cancellation)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_alpha_interference.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {output_dir / 'layer_alpha_interference.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise alpha analysis for model merging"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: use only 3 tasks instead of 8"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        datasets = N8_DATASETS[:3]
        logger.info(f"QUICK MODE: Using {len(datasets)} datasets: {datasets}")
    else:
        datasets = N8_DATASETS
        logger.info(f"Full analysis: {len(datasets)} datasets: {datasets}")

    # Load data
    pretrained, task_vectors = load_pretrained_and_task_vectors(datasets)

    # Use all layers (not just 2D) for analyses that support any dimension,
    # but track which are 2D for SVD-based analyses.
    all_layer_names = list(task_vectors[datasets[0]].keys())
    logger.info(f"Total layers: {len(all_layer_names)}")

    # Store all results for JSON output
    all_results = {}
    json_path = OUTPUT_DIR / "layer_alpha_analysis.json"

    def save_intermediate():
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        logger.info(f"Intermediate results saved to: {json_path}")

    # --- Analysis 1: Magnitude Profile ---
    magnitude_results = analyze_magnitude_profile(
        pretrained, task_vectors, datasets, all_layer_names
    )
    all_results["magnitude_profile"] = magnitude_results
    save_intermediate()

    # --- Analysis 2: Agreement Profile ---
    agreement_results = analyze_agreement_profile(
        task_vectors, datasets, all_layer_names
    )
    all_results["agreement_profile"] = agreement_results
    save_intermediate()

    # --- Analysis 3: Interference Estimation ---
    interference_results = analyze_interference(
        task_vectors, datasets, all_layer_names
    )
    all_results["interference"] = interference_results
    save_intermediate()

    # --- Analysis 4: Fisher Proxy ---
    fisher_results = analyze_fisher_proxy(
        task_vectors, datasets, all_layer_names
    )
    all_results["fisher_proxy"] = fisher_results
    save_intermediate()

    # --- Analysis 5: Cross-Layer Consistency ---
    consistency_results = analyze_cross_layer_consistency(
        magnitude_results, agreement_results, interference_results,
        fisher_results, all_layer_names
    )
    all_results["cross_layer_consistency"] = consistency_results
    save_intermediate()

    # Free pretrained to save memory before plotting
    del pretrained

    # --- Plots ---
    logger.info("\nGenerating plots...")
    plot_magnitude_profile(magnitude_results, all_layer_names, datasets, OUTPUT_DIR)
    plot_agreement_heatmap(agreement_results, all_layer_names, datasets, OUTPUT_DIR)
    plot_interference(interference_results, all_layer_names, datasets, OUTPUT_DIR)

    logger.info(f"\nAll results saved to: {json_path}")
    logger.info(f"All plots saved to: {OUTPUT_DIR}")
    logger.info("Layer alpha analysis complete.")


if __name__ == "__main__":
    main()
