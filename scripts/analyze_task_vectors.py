"""
Comprehensive analysis of task vectors from N8 ViT-B-32 models.

Analyses performed:
1. Singular value distribution analysis (power law, log-normal, exponential fits + Marchenko-Pastur)
2. Cross-task subspace overlap via principal angles
3. Sign structure analysis (pairwise agreement + consensus)
4. Layer-wise norm analysis (Frobenius, spectral, concentration ratio)

Usage:
    uv run python scripts/analyze_task_vectors.py          # full analysis
    uv run python scripts/analyze_task_vectors.py --quick   # 3 tasks, 5 layers
"""

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


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
    """Return dict of {name: tensor} for all 2D (matrix) layers."""
    return {k: v for k, v in task_vector.items() if v.dim() == 2}


def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
    """Marchenko-Pastur density for ratio gamma = rows/cols."""
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    pdf = np.zeros_like(x)
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (
        2 * np.pi * gamma * sigma2 * x[mask]
    )
    return pdf


def effective_rank(singular_values: np.ndarray) -> float:
    """Compute effective rank = exp(entropy of normalized SVs)."""
    sv = singular_values[singular_values > 1e-12]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def fit_distributions(sv_normalized: np.ndarray):
    """
    Fit normalized SVs to power law, log-normal, and exponential.
    Returns dict with fit parameters and KS statistics.
    """
    results = {}
    n = len(sv_normalized)
    ranks = np.arange(1, n + 1)

    # --- Power law fit: sv ~ C * rank^(-alpha) ---
    try:
        log_ranks = np.log(ranks)
        log_sv = np.log(sv_normalized + 1e-30)
        slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_sv)
        sv_pred = np.exp(intercept) * ranks ** slope
        sv_pred_norm = sv_pred / sv_pred.sum()
        ks_stat = np.max(np.abs(np.cumsum(sv_normalized) - np.cumsum(sv_pred_norm)))
        results["power_law"] = {
            "alpha": float(-slope),
            "r_squared": float(r_value ** 2),
            "ks_stat": float(ks_stat),
        }
    except Exception:
        results["power_law"] = {"alpha": float("nan"), "r_squared": 0.0, "ks_stat": 1.0}

    # --- Log-normal fit to the SV magnitudes ---
    try:
        log_sv = np.log(sv_normalized[sv_normalized > 1e-30])
        if len(log_sv) > 2:
            mu, sigma = np.mean(log_sv), np.std(log_sv)
            # Generate predicted distribution
            sv_pred = np.exp(mu + sigma ** 2 / 2) * np.exp(
                -((np.log(sv_normalized + 1e-30) - mu) ** 2) / (2 * sigma ** 2)
            )
            sv_pred_norm = sv_pred / sv_pred.sum() if sv_pred.sum() > 0 else sv_pred
            ks_stat = np.max(np.abs(np.cumsum(sv_normalized) - np.cumsum(sv_pred_norm)))
            results["lognormal"] = {
                "mu": float(mu),
                "sigma": float(sigma),
                "ks_stat": float(ks_stat),
            }
        else:
            results["lognormal"] = {"mu": float("nan"), "sigma": float("nan"), "ks_stat": 1.0}
    except Exception:
        results["lognormal"] = {"mu": float("nan"), "sigma": float("nan"), "ks_stat": 1.0}

    # --- Exponential fit: sv ~ C * exp(-beta * rank) ---
    try:

        def exp_func(x, C, beta):
            return C * np.exp(-beta * x)

        popt, _ = curve_fit(exp_func, ranks, sv_normalized, p0=[sv_normalized[0], 0.1], maxfev=5000)
        sv_pred = exp_func(ranks, *popt)
        sv_pred_norm = sv_pred / sv_pred.sum() if sv_pred.sum() > 0 else sv_pred
        ks_stat = np.max(np.abs(np.cumsum(sv_normalized) - np.cumsum(sv_pred_norm)))
        results["exponential"] = {
            "C": float(popt[0]),
            "beta": float(popt[1]),
            "ks_stat": float(ks_stat),
        }
    except Exception:
        results["exponential"] = {"C": float("nan"), "beta": float("nan"), "ks_stat": 1.0}

    # Best fit by KS statistic (lower is better)
    best = min(results, key=lambda k: results[k]["ks_stat"])
    results["best_fit"] = best

    return results


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    """Load state dict directly from HF cache without creating ImageEncoder."""
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
        del finetuned  # free memory
        logger.info(f"  {ds}: {len(tv)} layers, total params = {sum(v.numel() for v in tv.values()):,}")

    del pretrained  # free memory
    return task_vectors


# --------------------------------------------------------------------------- #
# Analysis 1: Singular Value Distribution
# --------------------------------------------------------------------------- #

def analyze_sv_distributions(task_vectors, layer_names_2d, datasets):
    """Analyze singular value distributions for each task and 2D layer."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: Singular Value Distribution")
    logger.info("=" * 60)

    results = {}
    # Collect effective ranks and fit results per layer type
    layer_type_fits = defaultdict(lambda: defaultdict(list))
    all_effective_ranks = {}  # layer_name -> {dataset -> eff_rank}

    for layer_name in layer_names_2d:
        results[layer_name] = {}
        all_effective_ranks[layer_name] = {}
        layer_type = classify_layer(layer_name)

        for ds in datasets:
            matrix = task_vectors[ds][layer_name].float()
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            sv = S.numpy()
            sv_norm = sv / sv.sum() if sv.sum() > 0 else sv

            # Effective rank
            eff_rank = effective_rank(sv)
            all_effective_ranks[layer_name][ds] = eff_rank

            # Fit distributions
            fits = fit_distributions(sv_norm)

            # Marchenko-Pastur comparison
            m, n = matrix.shape
            gamma = m / n if m > n else n / m
            # Compute variance of task vector entries to scale MP
            sigma2 = float(matrix.var())
            mp_lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
            mp_lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
            # Fraction of SVs above MP bulk edge (signal SVs)
            sv_squared = sv ** 2 / n  # eigenvalues of sample cov
            signal_fraction = float(np.mean(sv_squared > mp_lambda_plus))

            fits["mp_signal_fraction"] = signal_fraction
            fits["mp_lambda_plus"] = float(mp_lambda_plus)
            fits["effective_rank"] = eff_rank
            fits["num_svs"] = len(sv)
            fits["shape"] = list(matrix.shape)

            results[layer_name][ds] = fits
            layer_type_fits[layer_type][fits["best_fit"]].append(1)

        # Print summary for this layer
        avg_eff_rank = np.mean(list(all_effective_ranks[layer_name].values()))
        best_fits = [results[layer_name][ds]["best_fit"] for ds in datasets]
        most_common_fit = max(set(best_fits), key=best_fits.count)
        logger.info(
            f"  {layer_name}: avg_eff_rank={avg_eff_rank:.1f}/{results[layer_name][datasets[0]]['num_svs']}, "
            f"best_fit={most_common_fit}, signal_frac={np.mean([results[layer_name][ds]['mp_signal_fraction'] for ds in datasets]):.2f}"
        )

    # Summary by layer type
    logger.info("\n--- Best Fit Distribution by Layer Type ---")
    for ltype, fit_counts in sorted(layer_type_fits.items()):
        total = sum(len(v) for v in fit_counts.values())
        summary = ", ".join(f"{k}: {len(v)}/{total}" for k, v in sorted(fit_counts.items()))
        logger.info(f"  {ltype:15s}: {summary}")

    return results, all_effective_ranks


# --------------------------------------------------------------------------- #
# Analysis 2: Cross-Task Subspace Overlap
# --------------------------------------------------------------------------- #

def analyze_cross_task_overlap(task_vectors, layer_names_2d, datasets):
    """Compute principal angles between top-k left singular subspaces for each pair of tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Cross-Task Subspace Overlap")
    logger.info("=" * 60)

    # Pre-compute SVDs
    logger.info("Pre-computing SVDs for all task/layer combinations...")
    svd_cache = {}
    for ds in datasets:
        svd_cache[ds] = {}
        for layer_name in layer_names_2d:
            matrix = task_vectors[ds][layer_name].float()
            U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
            svd_cache[ds][layer_name] = {"U": U, "S": S, "Vh": Vh}

    k_values_labels = [1, 5, 10, "rank//4"]

    # Results: for each k, a matrix of average overlaps across layers
    n_tasks = len(datasets)
    results = {}

    for k_label in k_values_labels:
        overlap_matrix = np.zeros((n_tasks, n_tasks))
        per_layer_type_overlaps = defaultdict(list)

        for layer_name in layer_names_2d:
            min_rank = svd_cache[datasets[0]][layer_name]["S"].shape[0]
            if k_label == "rank//4":
                k = max(1, min_rank // 4)
            else:
                k = min(k_label, min_rank)

            layer_type = classify_layer(layer_name)

            for i, ds1 in enumerate(datasets):
                for j, ds2 in enumerate(datasets):
                    if i == j:
                        overlap_matrix[i, j] += 1.0
                        continue
                    if i > j:
                        continue  # fill symmetric later

                    U1 = svd_cache[ds1][layer_name]["U"][:, :k]
                    U2 = svd_cache[ds2][layer_name]["U"][:, :k]

                    # Principal angles: SVs of U1^T @ U2 are cos(angles)
                    cos_angles = torch.linalg.svdvals(U1.T @ U2)
                    avg_overlap = float(cos_angles.mean())

                    overlap_matrix[i, j] += avg_overlap
                    overlap_matrix[j, i] += avg_overlap
                    per_layer_type_overlaps[layer_type].append(avg_overlap)

        # Average across layers
        n_layers = len(layer_names_2d)
        overlap_matrix /= n_layers

        results[str(k_label)] = {
            "overlap_matrix": overlap_matrix.tolist(),
            "per_layer_type": {
                lt: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for lt, v in per_layer_type_overlaps.items()
            },
        }

        logger.info(f"\n--- k={k_label} ---")
        # Print overlap matrix
        header = "".ljust(12) + "".join(ds[:7].ljust(8) for ds in datasets)
        logger.info(header)
        for i, ds in enumerate(datasets):
            row = ds[:11].ljust(12) + "".join(f"{overlap_matrix[i, j]:.3f}   " for j in range(n_tasks))
            logger.info(row)

        # Print per-layer-type summary
        logger.info("  Per layer type:")
        for lt, vals in sorted(per_layer_type_overlaps.items()):
            logger.info(f"    {lt:15s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    return results


# --------------------------------------------------------------------------- #
# Analysis 3: Sign Structure
# --------------------------------------------------------------------------- #

def analyze_sign_structure(task_vectors, datasets):
    """Analyze sign agreement between task vectors."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Sign Structure Analysis")
    logger.info("=" * 60)

    all_layers = list(task_vectors[datasets[0]].keys())
    results = {}
    layer_type_sign_stats = defaultdict(lambda: {"pairwise": [], "consensus": []})

    for layer_name in all_layers:
        # Stack all task vectors for this layer
        tensors = [task_vectors[ds][layer_name].float().flatten() for ds in datasets]
        stacked = torch.stack(tensors, dim=0)  # (n_tasks, n_params)
        n_tasks = stacked.shape[0]
        n_params = stacked.shape[1]

        # Pairwise sign agreement
        signs = torch.sign(stacked)  # (n_tasks, n_params)
        pairwise_agreements = {}
        pairwise_values = []
        for i, j in combinations(range(n_tasks), 2):
            # Fraction of parameters with same sign (excluding zeros)
            nonzero_mask = (signs[i] != 0) & (signs[j] != 0)
            if nonzero_mask.sum() > 0:
                agreement = float((signs[i][nonzero_mask] == signs[j][nonzero_mask]).float().mean())
            else:
                agreement = 0.5
            pair_key = f"{datasets[i]}-{datasets[j]}"
            pairwise_agreements[pair_key] = agreement
            pairwise_values.append(agreement)

        # Consensus sign (majority vote)
        consensus_sign = torch.sign(signs.sum(dim=0))  # majority vote
        consensus_fractions = {}
        consensus_values = []
        for i, ds in enumerate(datasets):
            nonzero_mask = (signs[i] != 0) & (consensus_sign != 0)
            if nonzero_mask.sum() > 0:
                frac = float((signs[i][nonzero_mask] == consensus_sign[nonzero_mask]).float().mean())
            else:
                frac = 0.5
            consensus_fractions[ds] = frac
            consensus_values.append(frac)

        layer_type = classify_layer(layer_name)
        layer_type_sign_stats[layer_type]["pairwise"].extend(pairwise_values)
        layer_type_sign_stats[layer_type]["consensus"].extend(consensus_values)

        results[layer_name] = {
            "pairwise_mean": float(np.mean(pairwise_values)) if pairwise_values else 0.5,
            "pairwise_std": float(np.std(pairwise_values)) if pairwise_values else 0.0,
            "consensus_mean": float(np.mean(consensus_values)),
            "consensus_std": float(np.std(consensus_values)),
            "layer_type": layer_type,
        }

    # Summary by layer type
    logger.info("\n--- Sign Statistics by Layer Type ---")
    logger.info(f"{'Layer Type':15s}  {'Pairwise Agree':>15s}  {'Consensus Agree':>16s}")
    logger.info("-" * 50)
    for ltype, stats_dict in sorted(layer_type_sign_stats.items()):
        pw_mean = np.mean(stats_dict["pairwise"]) if stats_dict["pairwise"] else 0.5
        cs_mean = np.mean(stats_dict["consensus"]) if stats_dict["consensus"] else 0.5
        logger.info(f"{ltype:15s}  {pw_mean:15.4f}  {cs_mean:16.4f}")

    return results


# --------------------------------------------------------------------------- #
# Analysis 4: Layer-wise Norm Analysis
# --------------------------------------------------------------------------- #

def analyze_layer_norms(task_vectors, layer_names_2d, datasets):
    """Compute Frobenius norm, spectral norm, and concentration ratio for each layer."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: Layer-wise Norm Analysis")
    logger.info("=" * 60)

    results = {}
    for layer_name in layer_names_2d:
        results[layer_name] = {}
        for ds in datasets:
            matrix = task_vectors[ds][layer_name].float()
            frob_norm = float(torch.norm(matrix, p="fro"))
            spectral_norm = float(torch.linalg.svdvals(matrix)[0])
            concentration = spectral_norm / frob_norm if frob_norm > 0 else 0.0

            results[layer_name][ds] = {
                "frobenius_norm": frob_norm,
                "spectral_norm": spectral_norm,
                "concentration_ratio": concentration,
                "shape": list(matrix.shape),
            }

    # Print summary table
    logger.info(f"\n{'Layer':<50s}  {'Avg Frob':>10s}  {'Avg Spec':>10s}  {'Avg Conc':>10s}")
    logger.info("-" * 85)
    for layer_name in layer_names_2d:
        frobs = [results[layer_name][ds]["frobenius_norm"] for ds in datasets]
        specs = [results[layer_name][ds]["spectral_norm"] for ds in datasets]
        concs = [results[layer_name][ds]["concentration_ratio"] for ds in datasets]
        logger.info(
            f"{layer_name:<50s}  {np.mean(frobs):10.4f}  {np.mean(specs):10.4f}  {np.mean(concs):10.4f}"
        )

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_sv_distributions(task_vectors, layer_names_2d, datasets, output_dir):
    """Plot SV spectra for representative layers across tasks."""
    logger.info("Plotting SV distributions...")

    # Pick representative layers: one attention, one MLP, one projection
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
        representative_layers = layer_names_2d[:4]

    n_layers = len(representative_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for ax, layer_name in zip(axes, representative_layers):
        for ds in datasets:
            matrix = task_vectors[ds][layer_name].float()
            _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
            sv = S.numpy()
            sv_norm = sv / sv.sum() if sv.sum() > 0 else sv
            ax.semilogy(sv_norm, label=ds, alpha=0.7)

        # Add MP prediction for reference
        matrix = task_vectors[datasets[0]][layer_name].float()
        m, n = matrix.shape
        gamma = m / n if m >= n else n / m
        sigma2 = float(matrix.var())
        mp_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
        # Mark the MP edge
        ax.axhline(y=mp_plus / (min(m, n) * sigma2) if sigma2 > 0 else 0, color="k", linestyle="--", alpha=0.4, label="MP edge (approx)")

        layer_short = layer_name.split(".")[-1] if "." in layer_name else layer_name
        layer_type = classify_layer(layer_name)
        ax.set_title(f"{layer_type}\n{layer_short}", fontsize=9)
        ax.set_xlabel("SV index")
        ax.set_ylabel("Normalized SV")
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Singular Value Distributions (Normalized)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "sv_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_task_overlap(overlap_results, datasets, output_dir):
    """Plot heatmap of average subspace overlap between tasks."""
    logger.info("Plotting cross-task overlap heatmap...")

    # Use rank//4 results (most informative)
    k_key = "rank//4"
    if k_key not in overlap_results:
        k_key = list(overlap_results.keys())[-1]

    overlap_matrix = np.array(overlap_results[k_key]["overlap_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(datasets)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(datasets, fontsize=9)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(datasets)):
            text_color = "white" if overlap_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{overlap_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    plt.colorbar(im, label="Average cosine of principal angles")
    ax.set_title(f"Cross-Task Subspace Overlap (k={k_key})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "cross_task_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_norms(norm_results, layer_names_2d, datasets, output_dir):
    """Plot task vector norms across layers."""
    logger.info("Plotting layer norms...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Shorten layer names for x-axis
    short_names = []
    for name in layer_names_2d:
        parts = name.split(".")
        # Keep block number and final component
        short = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_names.append(short)

    x = np.arange(len(layer_names_2d))
    width = 0.8 / len(datasets)

    metrics = [
        ("frobenius_norm", "Frobenius Norm", axes[0]),
        ("spectral_norm", "Spectral Norm", axes[1]),
        ("concentration_ratio", "Spectral / Frobenius", axes[2]),
    ]

    for metric_key, metric_label, ax in metrics:
        for i, ds in enumerate(datasets):
            values = [norm_results[layer_name][ds][metric_key] for layer_name in layer_names_2d]
            ax.bar(x + i * width, values, width, label=ds, alpha=0.8)
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x + width * len(datasets) / 2)
    axes[-1].set_xticklabels(short_names, rotation=90, fontsize=6)
    axes[-1].set_xlabel("Layer")

    fig.suptitle("Task Vector Norms Across Layers", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_norms.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_effective_rank(eff_rank_results, layer_names_2d, datasets, output_dir):
    """Plot effective rank across layers."""
    logger.info("Plotting effective ranks...")

    fig, ax = plt.subplots(figsize=(14, 5))

    short_names = []
    for name in layer_names_2d:
        parts = name.split(".")
        short = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_names.append(short)

    x = np.arange(len(layer_names_2d))

    for ds in datasets:
        values = [eff_rank_results[layer_name][ds] for layer_name in layer_names_2d]
        ax.plot(x, values, "o-", label=ds, markersize=3, alpha=0.7)

    # Also plot actual rank for reference
    actual_ranks = []
    for layer_name in layer_names_2d:
        # All tasks have the same shape, take from first
        shape = list(datasets)[0]
        matrix = list(eff_rank_results[layer_name].values())
        # We need the actual min(m,n) - we'll get it from the key count
        # Actually, let's just store actual rank separately
        actual_ranks.append(None)  # placeholder

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank Across Layers (exp(entropy of normalized SVs))")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
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


def main():
    parser = argparse.ArgumentParser(description="Analyze task vectors from N8 ViT-B-32 models")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks, 5 layers")
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

    # Get 2D layer names (consistent across tasks)
    first_tv = task_vectors[datasets[0]]
    layer_names_2d = [k for k, v in first_tv.items() if v.dim() == 2]

    if args.quick:
        # Subsample layers evenly
        step = max(1, len(layer_names_2d) // 5)
        layer_names_2d = layer_names_2d[::step][:5]
        logger.info(f"QUICK MODE: Using {len(layer_names_2d)} layers")

    logger.info(f"Total 2D layers to analyze: {len(layer_names_2d)}")
    logger.info(f"Datasets: {datasets}")

    # Run analyses — save intermediate results after each phase
    all_results = {}
    json_path = OUTPUT_DIR / "task_vector_analysis.json"

    def save_intermediate():
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        logger.info(f"Intermediate results saved to: {json_path}")

    # 1. SV Distribution Analysis
    sv_results, eff_rank_results = analyze_sv_distributions(task_vectors, layer_names_2d, datasets)
    all_results["sv_distributions"] = sv_results
    all_results["effective_ranks"] = {
        layer: {ds: float(v) for ds, v in ranks.items()}
        for layer, ranks in eff_rank_results.items()
    }
    save_intermediate()

    # 2. Cross-Task Subspace Overlap
    overlap_results = analyze_cross_task_overlap(task_vectors, layer_names_2d, datasets)
    all_results["cross_task_overlap"] = overlap_results
    save_intermediate()

    # 3. Sign Structure
    sign_results = analyze_sign_structure(task_vectors, datasets)
    all_results["sign_structure"] = sign_results
    save_intermediate()

    # 4. Layer-wise Norms
    norm_results = analyze_layer_norms(task_vectors, layer_names_2d, datasets)
    all_results["layer_norms"] = norm_results
    save_intermediate()

    logger.info(f"\nFull results saved to: {json_path}")

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_sv_distributions(task_vectors, layer_names_2d, datasets, OUTPUT_DIR)
    plot_cross_task_overlap(overlap_results, datasets, OUTPUT_DIR)
    plot_layer_norms(norm_results, layer_names_2d, datasets, OUTPUT_DIR)
    plot_effective_rank(eff_rank_results, layer_names_2d, datasets, OUTPUT_DIR)

    logger.info(f"\nAll plots saved to: {OUTPUT_DIR}")
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
