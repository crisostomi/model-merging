"""
Analyze the singular value spectrum AFTER TSV concatenation + Procrustes orthogonalization
but BEFORE isotropic replacement.

This investigates WHY replacing all SVs with their arithmetic mean (isotropic) is uniquely
effective compared to other replacement strategies (geometric mean, median, top-k, etc.).

Pipeline replicated per 2D layer:
  1. SVD each task vector: U_i, S_i, Vh_i = SVD(tau_i)
  2. Keep SVs above Marchenko-Pastur noise edge (clamped to [mp_min_rank, mp_max_rank])
  3. Concatenate retained components into sum_u, sum_s, sum_v
  4. Procrustes orthogonalization: SVD(sum_u) and SVD(sum_v)
  5. Record sum_s (the post-Procrustes spectrum that isotropic replaces with its mean)

Analyses:
  - Pre-Procrustes per-task spectrum shape
  - Post-Procrustes spectrum statistics (skewness, kurtosis, Gini, effective rank, etc.)
  - Layer-type breakdown of spectrum shape
  - Comparison of replacement strategies (mean, geometric_mean, median, ones, pretrained)
  - Distribution fitting (log-normal, exponential, power-law)

Usage:
    uv run python scripts/analyze_spectrum_shape.py            # full N8 analysis (GPU recommended)
    uv run python scripts/analyze_spectrum_shape.py --quick    # 3 tasks, 5 layers (login node OK)
    uv run python scripts/analyze_spectrum_shape.py --device cpu  # force CPU
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

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
N8_DATASETS = [
    "SUN397",
    "Cars",
    "RESISC45",
    "EuroSAT",
    "SVHN",
    "GTSRB",
    "MNIST",
    "DTD",
]
MODEL_NAME = "ViT-B-32"
OUTPUT_DIR = PROJECT_ROOT / "results" / "spectrum_analysis"

# MP-edge defaults matching InterferenceAwareMerger
MP_MIN_RANK = 4
MP_MAX_RANK = 128


# ---------------------------------------------------------------------------
# Helpers — model loading
# ---------------------------------------------------------------------------
def load_state_dict(model_name: str, dataset_name: str = "base") -> OrderedDict:
    """Download and load a state dict from HuggingFace hub."""
    from huggingface_hub import hf_hub_download

    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    path = hf_hub_download(repo_id, "pytorch_model.bin")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Helpers — layer classification (mirrors interference_aware_merger.py)
# ---------------------------------------------------------------------------
def classify_layer_type(layer_name: str) -> str:
    """Classify layer for analysis grouping."""
    if "ln" in layer_name or "norm" in layer_name:
        return "layernorm"
    if "bias" in layer_name:
        return "bias"
    if "c_fc" in layer_name:
        return "mlp_fc"
    if "c_proj" in layer_name:
        return "mlp_proj"
    if "in_proj" in layer_name or "out_proj" in layer_name or "attn" in layer_name:
        return "attention"
    if "conv1" in layer_name:
        return "conv"
    if "proj" in layer_name:
        return "projection"
    if "embedding" in layer_name:
        return "embedding"
    return "other"


def get_block_index(layer_name: str) -> int:
    """Extract transformer block index from a layer name. Returns -1 if not in a resblock."""
    match = re.search(r"resblocks\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else -1


# ---------------------------------------------------------------------------
# Helpers — MP edge (copied from interference_aware_merger.py)
# ---------------------------------------------------------------------------
def compute_mp_edge(matrix: torch.Tensor) -> float:
    """Marchenko-Pastur upper edge singular-value threshold."""
    m, n = matrix.shape
    gamma = m / n if m >= n else n / m
    sigma2 = float(matrix.var())
    lambda_plus = sigma2 * (1 + gamma**0.5) ** 2
    sv_threshold = (lambda_plus * min(m, n)) ** 0.5
    return sv_threshold


# ---------------------------------------------------------------------------
# Helpers — spectrum statistics
# ---------------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of an array of values (0 = perfect equality, 1 = max inequality)."""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() < 1e-30:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def effective_rank(singular_values: np.ndarray) -> float:
    """exp(entropy of normalised SVs)."""
    sv = singular_values[singular_values > 1e-12]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def top_k_fraction(sv: np.ndarray, k: int) -> float:
    """Fraction of total energy (sum of SVs) in the top-k singular values."""
    if len(sv) == 0:
        return 0.0
    total = sv.sum()
    if total < 1e-30:
        return 0.0
    sorted_sv = np.sort(sv)[::-1]
    return float(sorted_sv[:k].sum() / total)


def compute_spectrum_stats(sv: np.ndarray) -> dict:
    """Full spectrum statistics for an array of singular values."""
    if len(sv) == 0:
        return {}
    return {
        "n": int(len(sv)),
        "mean": float(np.mean(sv)),
        "median": float(np.median(sv)),
        "std": float(np.std(sv)),
        "max": float(np.max(sv)),
        "min": float(np.min(sv)),
        "skewness": float(stats.skew(sv)),
        "kurtosis": float(stats.kurtosis(sv)),
        "gini": gini_coefficient(sv),
        "effective_rank": effective_rank(sv),
        "top1_frac": top_k_fraction(sv, 1),
        "top5_frac": top_k_fraction(sv, 5),
        "top10_frac": top_k_fraction(sv, 10),
        "cv": float(np.std(sv) / np.mean(sv)) if np.mean(sv) > 1e-30 else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers — distribution fitting
# ---------------------------------------------------------------------------
def fit_distributions(sv: np.ndarray) -> dict:
    """
    Fit the SV spectrum to several candidate distributions.
    Returns fit parameters and KS goodness-of-fit for each.
    """
    results = {}
    if len(sv) < 5:
        return results

    # Normalise SVs to unit sum for distribution fitting
    sv_sorted = np.sort(sv)[::-1]
    sv_pos = sv_sorted[sv_sorted > 1e-30]
    if len(sv_pos) < 5:
        return results

    # 1) Log-normal fit
    try:
        log_sv = np.log(sv_pos)
        mu, sigma = np.mean(log_sv), np.std(log_sv)
        ks_stat, ks_p = stats.kstest(sv_pos, "lognorm", args=(sigma, 0, np.exp(mu)))
        results["lognormal"] = {
            "mu": float(mu),
            "sigma": float(sigma),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    # 2) Exponential fit
    try:
        loc, scale = stats.expon.fit(sv_pos)
        ks_stat, ks_p = stats.kstest(sv_pos, "expon", args=(loc, scale))
        results["exponential"] = {
            "loc": float(loc),
            "scale": float(scale),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    # 3) Power-law fit (via rank regression on log-log)
    try:
        ranks = np.arange(1, len(sv_pos) + 1)
        log_ranks = np.log(ranks)
        log_vals = np.log(sv_pos)
        slope, intercept, r_value, _, _ = stats.linregress(log_ranks, log_vals)
        results["power_law"] = {
            "alpha": float(-slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
        }
    except Exception:
        pass

    # 4) Gamma fit
    try:
        a, loc, scale = stats.gamma.fit(sv_pos, floc=0)
        ks_stat, ks_p = stats.kstest(sv_pos, "gamma", args=(a, loc, scale))
        results["gamma"] = {
            "shape": float(a),
            "scale": float(scale),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# Core analysis: replicate MP-edge + Procrustes pipeline per layer
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze_layer(
    layer_name: str,
    task_matrices: dict,
    pretrained_matrix: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """
    For a single 2D layer, replicate the MP-edge + TSV concatenation + Procrustes pipeline
    and capture spectra at each stage.

    Args:
        layer_name: name of the layer
        task_matrices: {dataset_name: 2D task-vector tensor for this layer}
        pretrained_matrix: the pretrained model's weight for this layer
        device: compute device

    Returns:
        dict with all analysis results for this layer.
    """
    layer_type = classify_layer_type(layer_name)
    block_idx = get_block_index(layer_name)

    # --- Step 1: SVD each task vector + MP-edge filtering ---
    per_task_spectra = {}
    retained_components = []

    for ds_name, tv in task_matrices.items():
        tv = tv.to(device).float()
        U, S, Vh = torch.linalg.svd(tv, full_matrices=False)

        # MP edge
        threshold = compute_mp_edge(tv)
        k = int((S > threshold).sum().item())
        k = max(MP_MIN_RANK, min(MP_MAX_RANK, k))

        per_task_spectra[ds_name] = {
            "full_sv": S.cpu().numpy().tolist(),
            "mp_edge": float(threshold),
            "retained_rank": k,
            "retained_sv": S[:k].cpu().numpy().tolist(),
        }

        retained_components.append(
            (U[:, :k].to(device), S[:k].to(device), Vh[:k, :].to(device))
        )

    # --- Step 2: Concatenate retained components (TSV style) ---
    total_rank = sum(s.shape[0] for _, s, _ in retained_components)
    m = retained_components[0][0].shape[0]
    n = retained_components[0][2].shape[1]

    sum_u = torch.zeros(m, total_rank, device=device)
    sum_s = torch.zeros(total_rank, device=device)
    sum_v = torch.zeros(total_rank, n, device=device)

    offset = 0
    for u_i, s_i, v_i in retained_components:
        rank_i = s_i.shape[0]
        sum_u[:, offset : offset + rank_i] = u_i
        sum_s[offset : offset + rank_i] = s_i
        sum_v[offset : offset + rank_i, :] = v_i
        offset += rank_i

    # --- Step 3: Procrustes orthogonalization (same as aggregate_interference_aware) ---
    # The actual code does:
    #   u_u, s_u, v_u = SVD(sum_u)
    #   u_v, s_v, v_v = SVD(sum_v)
    #   merged = u_u @ v_u @ diag(sum_s) @ u_v @ v_v
    # Note: the SVs used are sum_s (the concatenated per-task SVs), NOT the SVs from
    # the Procrustes SVDs of sum_u and sum_v. The Procrustes step only orthogonalises
    # the left and right bases; sum_s is the spectrum that isotropic replaces.
    u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

    post_procrustes_sv = sum_s.cpu().numpy()

    # Also compute the SVs of the fully reconstructed merged matrix for comparison
    # merged_matrix = u_u @ v_u @ diag(sum_s) @ u_v @ v_v
    merged_matrix = torch.linalg.multi_dot(
        (u_u, v_u, torch.diag(sum_s), u_v, v_v)
    )
    _, actual_merged_sv, _ = torch.linalg.svd(merged_matrix, full_matrices=False)
    actual_merged_sv_np = actual_merged_sv.cpu().numpy()

    # Pretrained SVs for comparison
    pretrained_matrix = pretrained_matrix.to(device).float()
    _, pretrained_sv, _ = torch.linalg.svd(pretrained_matrix, full_matrices=False)
    pretrained_sv_np = pretrained_sv.cpu().numpy()

    # --- Step 4: Spectrum statistics ---
    # Sort post-Procrustes SVs descending for analysis
    sorted_sv = np.sort(post_procrustes_sv)[::-1]
    spec_stats = compute_spectrum_stats(sorted_sv)

    # Distribution fits on the sorted post-Procrustes spectrum
    dist_fits = fit_distributions(sorted_sv)

    # --- Step 5: Replacement strategy comparison ---
    # For each strategy, compute the Frobenius norm of the resulting merged task vector
    iso_mean = float(np.mean(post_procrustes_sv))
    iso_median = float(np.median(post_procrustes_sv))
    iso_geomean = float(np.exp(np.mean(np.log(post_procrustes_sv + 1e-30))))

    strategies = {}

    # Original (TSV): keep sum_s as-is
    # Frobenius norm of merged = ||u_u @ v_u @ diag(sum_s) @ u_v @ v_v||_F
    # Since u_u @ v_u and u_v @ v_v are products of orthogonal-ish matrices,
    # the Frobenius norm is approximately ||sum_s||_2 but not exactly.
    # Compute exactly from merged_matrix.
    strategies["original"] = {
        "frobenius_norm": float(merged_matrix.norm().item()),
        "replacement_value": None,
    }

    # Mean (isotropic)
    merged_iso_mean = iso_mean * torch.linalg.multi_dot((u_u, v_u, u_v, v_v))
    strategies["mean"] = {
        "frobenius_norm": float(merged_iso_mean.norm().item()),
        "replacement_value": iso_mean,
    }

    # Geometric mean
    merged_iso_geomean = iso_geomean * torch.linalg.multi_dot((u_u, v_u, u_v, v_v))
    strategies["geometric_mean"] = {
        "frobenius_norm": float(merged_iso_geomean.norm().item()),
        "replacement_value": iso_geomean,
    }

    # Median
    merged_iso_median = iso_median * torch.linalg.multi_dot((u_u, v_u, u_v, v_v))
    strategies["median"] = {
        "frobenius_norm": float(merged_iso_median.norm().item()),
        "replacement_value": iso_median,
    }

    # Ones (pure direction, all SVs = 1.0)
    merged_ones = 1.0 * torch.linalg.multi_dot((u_u, v_u, u_v, v_v))
    strategies["ones"] = {
        "frobenius_norm": float(merged_ones.norm().item()),
        "replacement_value": 1.0,
    }

    # Pretrained: use truncated pretrained SVs (match rank of post-Procrustes)
    rank_used = len(post_procrustes_sv)
    pretrained_truncated_sv = pretrained_sv_np[:rank_used]
    pretrained_s_tensor = torch.from_numpy(pretrained_truncated_sv).to(device).float()
    # Pad if pretrained has fewer SVs
    if len(pretrained_truncated_sv) < rank_used:
        pad = torch.zeros(rank_used - len(pretrained_truncated_sv), device=device)
        pretrained_s_tensor = torch.cat([pretrained_s_tensor, pad])
    merged_pretrained = torch.linalg.multi_dot(
        (u_u, v_u, torch.diag(pretrained_s_tensor), u_v, v_v)
    )
    strategies["pretrained"] = {
        "frobenius_norm": float(merged_pretrained.norm().item()),
        "replacement_value": float(np.mean(pretrained_truncated_sv)),
    }

    # Clean up GPU memory
    del merged_matrix, merged_iso_mean, merged_iso_geomean, merged_iso_median
    del merged_ones, merged_pretrained
    del sum_u, sum_s, sum_v, u_u, v_u, u_v, v_v
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "layer_name": layer_name,
        "layer_type": layer_type,
        "block_index": block_idx,
        "shape": [m, n],
        "total_concatenated_rank": total_rank,
        "per_task_spectra": per_task_spectra,
        "post_procrustes_sv": sorted_sv.tolist(),
        "actual_merged_sv": actual_merged_sv_np.tolist(),
        "pretrained_sv_top20": pretrained_sv_np[:20].tolist(),
        "spectrum_stats": spec_stats,
        "distribution_fits": dist_fits,
        "replacement_strategies": strategies,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_post_procrustes_spectra(results: list, output_dir: Path):
    """Plot post-Procrustes spectra for representative layers (log scale)."""
    # Pick representative layers: one attention, one MLP, one projection, one early, one late
    type_examples = {}
    for r in results:
        lt = r["layer_type"]
        bi = r["block_index"]
        key = f"{lt}_block{bi}"
        if lt not in type_examples or (lt in type_examples and bi > type_examples[lt]["block_index"]):
            type_examples[lt] = r

    # Also grab one early layer
    early = [r for r in results if 0 <= r["block_index"] <= 2]
    late = [r for r in results if r["block_index"] >= 9]

    examples = []
    seen_names = set()
    for lt in ["attention", "mlp_fc", "mlp_proj", "projection"]:
        if lt in type_examples and type_examples[lt]["layer_name"] not in seen_names:
            examples.append(type_examples[lt])
            seen_names.add(type_examples[lt]["layer_name"])
    if early:
        r = early[0]
        if r["layer_name"] not in seen_names:
            examples.append(r)
            seen_names.add(r["layer_name"])
    if late:
        r = late[-1]
        if r["layer_name"] not in seen_names:
            examples.append(r)
            seen_names.add(r["layer_name"])

    examples = examples[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, r in enumerate(examples):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sv = np.array(r["post_procrustes_sv"])
        actual_sv = np.array(r["actual_merged_sv"])

        ax.semilogy(sv, "b-", linewidth=2, alpha=0.8, label="Concatenated SVs (sum_s)")
        ax.semilogy(actual_sv[:len(sv)], "r--", linewidth=1.5, alpha=0.7, label="Actual merged SVs")

        # Mean line
        mean_val = np.mean(sv)
        ax.axhline(mean_val, color="green", linestyle=":", linewidth=2, label=f"Mean = {mean_val:.4f}")
        # Median line
        median_val = np.median(sv)
        ax.axhline(median_val, color="orange", linestyle=":", linewidth=1.5, label=f"Median = {median_val:.4f}")
        # Geometric mean
        geomean_val = np.exp(np.mean(np.log(sv + 1e-30)))
        ax.axhline(geomean_val, color="purple", linestyle=":", linewidth=1.5, label=f"GeoMean = {geomean_val:.4f}")

        short_name = r["layer_name"].replace("model.visual.transformer.", "").replace("model.visual.", "")
        ax.set_title(f"{short_name}\n[{r['layer_type']}, block {r['block_index']}]", fontsize=10)
        ax.set_xlabel("Index (sorted descending)")
        ax.set_ylabel("Singular Value")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(examples), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Post-Procrustes Singular Value Spectra (log scale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "post_procrustes_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved post_procrustes_spectra.png")


def plot_spectrum_statistics_by_layer_type(results: list, output_dir: Path):
    """Bar charts of skewness, kurtosis, Gini, effective_rank by layer type."""
    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r["spectrum_stats"])

    layer_types = sorted(by_type.keys())
    metrics = ["skewness", "kurtosis", "gini", "effective_rank", "cv"]
    metric_labels = ["Skewness", "Kurtosis", "Gini Coefficient", "Effective Rank", "Coeff. of Variation"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        means = []
        stds = []
        for lt in layer_types:
            vals = [s[metric] for s in by_type[lt] if metric in s]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)

        x = np.arange(len(layer_types))
        bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.7, color="steelblue", edgecolor="navy")
        ax.set_xticks(x)
        ax.set_xticklabels(layer_types, rotation=45, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Post-Procrustes Spectrum Statistics by Layer Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "spectrum_statistics_by_layer_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved spectrum_statistics_by_layer_type.png")


def plot_replacement_strategy_norms(results: list, output_dir: Path):
    """Total merged vector norm under each SV replacement strategy."""
    strategies_names = ["original", "mean", "geometric_mean", "median", "ones", "pretrained"]
    strategy_labels = ["Original\n(TSV)", "Mean\n(Isotropic)", "Geo. Mean", "Median", "Ones\n(direction)", "Pretrained\nSVs"]

    # Aggregate norms across all layers for each strategy
    total_norms = {s: 0.0 for s in strategies_names}
    per_layer_norms = {s: [] for s in strategies_names}

    for r in results:
        for s in strategies_names:
            fn = r["replacement_strategies"][s]["frobenius_norm"]
            total_norms[s] += fn**2  # Sum of squared Frobenius norms
            per_layer_norms[s].append(fn)

    # Convert to total Frobenius norm (sqrt of sum of squares)
    total_norms = {s: np.sqrt(v) for s, v in total_norms.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: total norm
    x = np.arange(len(strategies_names))
    vals = [total_norms[s] for s in strategies_names]
    bars = ax1.bar(x, vals, alpha=0.7, color=["gray", "green", "purple", "orange", "red", "blue"], edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_labels, fontsize=10)
    ax1.set_ylabel("Total Frobenius Norm (all layers)")
    ax1.set_title("Total Merged Task Vector Norm by Strategy")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # Right: per-layer norm distribution (box plot)
    bp_data = [per_layer_norms[s] for s in strategies_names]
    bp = ax2.boxplot(bp_data, labels=strategy_labels, patch_artist=True)
    colors = ["gray", "green", "purple", "orange", "red", "blue"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax2.set_ylabel("Per-Layer Frobenius Norm")
    ax2.set_title("Per-Layer Norm Distribution by Strategy")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Replacement Strategy Comparison: Energy (Frobenius Norm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "replacement_strategy_norms.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved replacement_strategy_norms.png")


def plot_distribution_fits(results: list, output_dir: Path):
    """Histogram overlays and QQ plots showing distribution shape of post-Procrustes SVs."""
    # Pick 4 representative layers
    examples = []
    seen_types = set()
    for r in results:
        lt = r["layer_type"]
        if lt not in seen_types and lt in ("attention", "mlp_fc", "mlp_proj", "projection"):
            examples.append(r)
            seen_types.add(lt)
        if len(examples) >= 4:
            break
    # Fill up with whatever is available
    for r in results:
        if len(examples) >= 4:
            break
        if r["layer_name"] not in {e["layer_name"] for e in examples}:
            examples.append(r)

    fig, axes = plt.subplots(2, min(4, len(examples)), figsize=(5 * min(4, len(examples)), 10))
    if len(examples) == 1:
        axes = axes.reshape(2, 1)

    for col, r in enumerate(examples[:4]):
        if col >= axes.shape[1]:
            break
        sv = np.array(r["post_procrustes_sv"])
        sv_pos = sv[sv > 1e-30]
        fits = r.get("distribution_fits", {})

        # Top row: histogram + fitted PDFs
        ax_top = axes[0, col]
        ax_top.hist(sv_pos, bins=30, density=True, alpha=0.5, color="steelblue", edgecolor="navy", label="Data")

        x_range = np.linspace(sv_pos.min() * 0.8, sv_pos.max() * 1.1, 200)

        if "lognormal" in fits:
            f = fits["lognormal"]
            pdf = stats.lognorm.pdf(x_range, f["sigma"], 0, np.exp(f["mu"]))
            ax_top.plot(x_range, pdf, "r-", linewidth=2, label=f"LogNorm (KS={f['ks_stat']:.3f})")

        if "exponential" in fits:
            f = fits["exponential"]
            pdf = stats.expon.pdf(x_range, f["loc"], f["scale"])
            ax_top.plot(x_range, pdf, "g--", linewidth=2, label=f"Expon (KS={f['ks_stat']:.3f})")

        if "gamma" in fits:
            f = fits["gamma"]
            pdf = stats.gamma.pdf(x_range, f["shape"], 0, f["scale"])
            ax_top.plot(x_range, pdf, "m:", linewidth=2, label=f"Gamma (KS={f['ks_stat']:.3f})")

        short_name = r["layer_name"].replace("model.visual.transformer.", "").replace("model.visual.", "")
        ax_top.set_title(f"{short_name}\n[{r['layer_type']}]", fontsize=9)
        ax_top.legend(fontsize=7)
        ax_top.set_xlabel("Singular Value")
        ax_top.set_ylabel("Density")
        ax_top.grid(True, alpha=0.3)

        # Bottom row: QQ plot against log-normal
        ax_bot = axes[1, col]
        if "lognormal" in fits:
            f = fits["lognormal"]
            theoretical_quantiles = stats.lognorm.ppf(
                np.linspace(0.01, 0.99, len(sv_pos)), f["sigma"], 0, np.exp(f["mu"])
            )
            sorted_data = np.sort(sv_pos)
            ax_bot.scatter(theoretical_quantiles, sorted_data, s=10, alpha=0.6, color="steelblue")
            lims = [
                min(theoretical_quantiles.min(), sorted_data.min()),
                max(theoretical_quantiles.max(), sorted_data.max()),
            ]
            ax_bot.plot(lims, lims, "r--", linewidth=1, label="y=x")
            ax_bot.set_xlabel("Log-Normal Theoretical Quantiles")
            ax_bot.set_ylabel("Observed Quantiles")
            ax_bot.set_title("QQ Plot (Log-Normal)")
            ax_bot.legend(fontsize=8)
        else:
            ax_bot.text(0.5, 0.5, "No log-normal fit", transform=ax_bot.transAxes, ha="center")
        ax_bot.grid(True, alpha=0.3)

    fig.suptitle("Distribution Fits for Post-Procrustes Spectra", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_fits.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution_fits.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze post-Procrustes SV spectrum shape")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 3 tasks, 5 layers")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model

    datasets = N8_DATASETS
    if args.quick:
        datasets = datasets[:3]
        logger.info(f"Quick mode: using {len(datasets)} datasets: {datasets}")
    else:
        logger.info(f"Full mode: using {len(datasets)} datasets: {datasets}")

    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")

    # --- Load pretrained model ---
    logger.info("Loading pretrained (base) state dict...")
    pretrained_sd = load_state_dict(model_name, "base")
    logger.info(f"  Loaded {len(pretrained_sd)} keys from pretrained model")

    # --- Identify 2D layers ---
    matrix_layers = [k for k, v in pretrained_sd.items() if v.dim() == 2 and "text_projection" not in k]
    logger.info(f"  Found {len(matrix_layers)} 2D (matrix) layers")

    if args.quick:
        # In quick mode, sample 5 representative layers
        sampled = []
        # Try to get one of each type
        seen_types = set()
        for k in matrix_layers:
            lt = classify_layer_type(k)
            if lt not in seen_types:
                sampled.append(k)
                seen_types.add(lt)
            if len(sampled) >= 5:
                break
        # Fill up if needed
        for k in matrix_layers:
            if len(sampled) >= 5:
                break
            if k not in sampled:
                sampled.append(k)
        matrix_layers = sampled
        logger.info(f"  Quick mode: analyzing {len(matrix_layers)} layers")

    # --- Load fine-tuned models and compute task vectors ---
    logger.info("Loading fine-tuned models and computing task vectors...")
    task_vectors = {}  # {dataset: {layer: tensor}}
    for ds in datasets:
        logger.info(f"  Loading {ds}...")
        ft_sd = load_state_dict(model_name, ds)
        tv = OrderedDict()
        for key in pretrained_sd:
            if pretrained_sd[key].dtype in [torch.int64, torch.uint8]:
                continue
            tv[key] = ft_sd[key] - pretrained_sd[key]
        task_vectors[ds] = tv
        del ft_sd

    logger.info(f"Computed task vectors for {len(task_vectors)} datasets")

    # --- Run analysis for each 2D layer ---
    all_results = []
    for layer_idx, layer_name in enumerate(matrix_layers):
        logger.info(
            f"[{layer_idx + 1}/{len(matrix_layers)}] Analyzing {layer_name} "
            f"({classify_layer_type(layer_name)}, shape={list(pretrained_sd[layer_name].shape)})"
        )

        task_mats = {ds: task_vectors[ds][layer_name] for ds in datasets}
        pretrained_mat = pretrained_sd[layer_name]

        result = analyze_layer(layer_name, task_mats, pretrained_mat, device=device)
        all_results.append(result)

        if (layer_idx + 1) % 10 == 0:
            logger.info(f"  Progress: {layer_idx + 1}/{len(matrix_layers)} layers done")

    logger.info(f"Analysis complete for {len(all_results)} layers")

    # --- Compute aggregate statistics ---
    logger.info("Computing aggregate statistics...")
    agg = {
        "n_layers": len(all_results),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "model": model_name,
    }

    # Overall spectrum stats aggregated across all layers
    all_skewness = [r["spectrum_stats"]["skewness"] for r in all_results]
    all_kurtosis = [r["spectrum_stats"]["kurtosis"] for r in all_results]
    all_gini = [r["spectrum_stats"]["gini"] for r in all_results]
    all_effrank = [r["spectrum_stats"]["effective_rank"] for r in all_results]
    all_cv = [r["spectrum_stats"]["cv"] for r in all_results]

    agg["overall_spectrum_stats"] = {
        "skewness": {"mean": float(np.mean(all_skewness)), "std": float(np.std(all_skewness))},
        "kurtosis": {"mean": float(np.mean(all_kurtosis)), "std": float(np.std(all_kurtosis))},
        "gini": {"mean": float(np.mean(all_gini)), "std": float(np.std(all_gini))},
        "effective_rank": {"mean": float(np.mean(all_effrank)), "std": float(np.std(all_effrank))},
        "cv": {"mean": float(np.mean(all_cv)), "std": float(np.std(all_cv))},
    }

    # By layer type
    by_type = defaultdict(list)
    for r in all_results:
        by_type[r["layer_type"]].append(r)

    agg["by_layer_type"] = {}
    for lt, layer_results in sorted(by_type.items()):
        s_list = [r["spectrum_stats"] for r in layer_results]
        agg["by_layer_type"][lt] = {
            "n_layers": len(layer_results),
            "avg_skewness": float(np.mean([s["skewness"] for s in s_list])),
            "avg_kurtosis": float(np.mean([s["kurtosis"] for s in s_list])),
            "avg_gini": float(np.mean([s["gini"] for s in s_list])),
            "avg_effective_rank": float(np.mean([s["effective_rank"] for s in s_list])),
            "avg_cv": float(np.mean([s["cv"] for s in s_list])),
            "avg_top1_frac": float(np.mean([s["top1_frac"] for s in s_list])),
            "avg_top5_frac": float(np.mean([s["top5_frac"] for s in s_list])),
            "avg_top10_frac": float(np.mean([s["top10_frac"] for s in s_list])),
        }

    # Best-fitting distribution across layers
    fit_wins = defaultdict(int)
    for r in all_results:
        fits = r.get("distribution_fits", {})
        ks_scores = {}
        for dist_name, fit_data in fits.items():
            if "ks_stat" in fit_data:
                ks_scores[dist_name] = fit_data["ks_stat"]
        if ks_scores:
            best = min(ks_scores, key=ks_scores.get)
            fit_wins[best] += 1

    agg["best_distribution_fit_counts"] = dict(fit_wins)

    # Mean/median/geomean comparison across all layers
    mean_vs_median = []
    mean_vs_geomean = []
    for r in all_results:
        sv = np.array(r["post_procrustes_sv"])
        mean_val = np.mean(sv)
        median_val = np.median(sv)
        geomean_val = np.exp(np.mean(np.log(sv + 1e-30)))
        if mean_val > 1e-30:
            mean_vs_median.append(median_val / mean_val)
            mean_vs_geomean.append(geomean_val / mean_val)

    agg["mean_vs_alternatives"] = {
        "median_over_mean": {
            "avg": float(np.mean(mean_vs_median)),
            "std": float(np.std(mean_vs_median)),
        },
        "geomean_over_mean": {
            "avg": float(np.mean(mean_vs_geomean)),
            "std": float(np.std(mean_vs_geomean)),
        },
    }

    # --- Save JSON results ---
    # The per-layer results can be large; store the full spectra separately
    # and keep the main JSON with stats only
    json_results = {
        "aggregate": agg,
        "per_layer": all_results,
    }

    json_path = output_dir / "spectrum_analysis.json"
    logger.info(f"Saving results to {json_path}...")

    # Custom serializer for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  Saved {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")

    # --- Generate plots ---
    logger.info("Generating plots...")
    plot_post_procrustes_spectra(all_results, output_dir)
    plot_spectrum_statistics_by_layer_type(all_results, output_dir)
    plot_replacement_strategy_norms(all_results, output_dir)
    plot_distribution_fits(all_results, output_dir)

    # --- Print summary ---
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Analyzed {len(all_results)} 2D layers across {len(datasets)} tasks")
    logger.info(f"")
    logger.info(f"Overall post-Procrustes spectrum shape:")
    logger.info(f"  Skewness:       {agg['overall_spectrum_stats']['skewness']['mean']:.3f} +/- {agg['overall_spectrum_stats']['skewness']['std']:.3f}")
    logger.info(f"  Kurtosis:       {agg['overall_spectrum_stats']['kurtosis']['mean']:.3f} +/- {agg['overall_spectrum_stats']['kurtosis']['std']:.3f}")
    logger.info(f"  Gini coeff:     {agg['overall_spectrum_stats']['gini']['mean']:.3f} +/- {agg['overall_spectrum_stats']['gini']['std']:.3f}")
    logger.info(f"  Effective rank: {agg['overall_spectrum_stats']['effective_rank']['mean']:.1f} +/- {agg['overall_spectrum_stats']['effective_rank']['std']:.1f}")
    logger.info(f"  CV:             {agg['overall_spectrum_stats']['cv']['mean']:.3f} +/- {agg['overall_spectrum_stats']['cv']['std']:.3f}")
    logger.info(f"")
    logger.info(f"Mean vs alternatives (ratio across layers):")
    logger.info(f"  median/mean:    {agg['mean_vs_alternatives']['median_over_mean']['avg']:.4f}")
    logger.info(f"  geomean/mean:   {agg['mean_vs_alternatives']['geomean_over_mean']['avg']:.4f}")
    logger.info(f"")
    logger.info(f"Best-fitting distributions: {dict(fit_wins)}")
    logger.info(f"")

    # Strategy norms
    logger.info(f"Total Frobenius norm by replacement strategy:")
    strategies = ["original", "mean", "geometric_mean", "median", "ones", "pretrained"]
    for s in strategies:
        total = np.sqrt(sum(r["replacement_strategies"][s]["frobenius_norm"] ** 2 for r in all_results))
        logger.info(f"  {s:20s}: {total:.2f}")

    logger.info(f"")
    logger.info(f"By layer type:")
    for lt, info in sorted(agg["by_layer_type"].items()):
        logger.info(
            f"  {lt:15s}: n={info['n_layers']:2d}, "
            f"skew={info['avg_skewness']:.2f}, "
            f"gini={info['avg_gini']:.3f}, "
            f"eff_rank={info['avg_effective_rank']:.1f}, "
            f"top5={info['avg_top5_frac']:.3f}"
        )

    logger.info(f"")
    logger.info(f"Results saved to {output_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
