"""
Cross-layer structure analysis of task vectors from N8 ViT-B-32 models.

Research question: Current merging methods treat each layer independently.
Is there exploitable structure ACROSS layers? If a task changes block 5 in
a particular way, does that predict how it changes block 11?

Analyses performed:
1. Layer magnitude profile per task (relative Frobenius norms across depth)
2. Inter-layer correlation matrix (cosine similarity between layers, averaged across tasks)
3. Cross-task layer profile similarity (do tasks with similar profiles interfere more?)
4. Task vector as a matrix -- SVD across layers (effective rank of task x block matrix)
5. Layer-pair co-variation (correlation of TV values across tasks for same-type layer pairs)
6. Principal "task modes" across layers (PCA on full concatenated task vectors)

Usage:
    uv run python scripts/analyze_cross_layer.py          # full analysis
    uv run python scripts/analyze_cross_layer.py --quick   # 3 tasks, subset of layers
"""

import argparse
import json
import logging
import os
import sys
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from itertools import combinations

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_layer"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

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


def get_block_depth(layer_name: str) -> int:
    """Extract transformer block index from layer name, or -1 if not a block layer."""
    match = re.search(r'resblocks\.(\d+)', layer_name)
    return int(match.group(1)) if match else -1


def get_layer_type_key(layer_name: str) -> str:
    """Get a canonical layer-type key by replacing the block index with a wildcard.

    E.g. 'visual.transformer.resblocks.5.attn.in_proj_weight'
      -> 'visual.transformer.resblocks.*.attn.in_proj_weight'
    """
    return re.sub(r'resblocks\.\d+', 'resblocks.*', layer_name)


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
# Loading
# --------------------------------------------------------------------------- #

def load_state_dict(model_name, dataset_name="base"):
    """Load state dict from HuggingFace cache."""
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
        logger.info(f"  {ds}: {len(tv)} layers, total params = {sum(v.numel() for v in tv.values()):,}")

    # Keep base_sd for analysis 1 (normalizing by pretrained weight norm)
    return task_vectors, base_sd


def load_loo_interference():
    """Load leave-one-out interference matrix if available."""
    loo_path = PROJECT_ROOT / "results" / "analysis" / "leave_one_out_results.json"
    if loo_path.exists():
        with open(loo_path) as f:
            data = json.load(f)
        return data
    return None


# --------------------------------------------------------------------------- #
# Analysis 1: Layer Magnitude Profile Per Task
# --------------------------------------------------------------------------- #

def analyze_layer_magnitude_profiles(task_vectors, base_sd, datasets):
    """Compute relative Frobenius norm of task vector at each layer, normalized by pretrained weight norm."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: Layer Magnitude Profile Per Task")
    logger.info("=" * 60)

    all_layer_names = list(task_vectors[datasets[0]].keys())

    # Build ordered list of layers that belong to transformer blocks
    block_layers = [(name, get_block_depth(name)) for name in all_layer_names if get_block_depth(name) >= 0]
    # Also include non-block layers at the end
    non_block_layers = [name for name in all_layer_names if get_block_depth(name) < 0]

    # Sort block layers by depth, then by name for consistent ordering
    block_layers.sort(key=lambda x: (x[1], x[0]))
    ordered_layers = [name for name, _ in block_layers] + non_block_layers

    # For the profile plot, we aggregate per block: sum of Frobenius norms across all
    # sublayers within each block, normalized by pretrained norms
    max_block = max(get_block_depth(name) for name in all_layer_names)
    block_indices = list(range(max_block + 1))

    profiles = {}  # dataset -> list of relative magnitudes per block
    raw_layer_profiles = {}  # dataset -> {layer_name: relative_norm}

    for ds in datasets:
        block_tv_norms = defaultdict(float)
        block_base_norms = defaultdict(float)
        layer_profile = {}

        for name in all_layer_names:
            tv_tensor = task_vectors[ds][name].float()
            tv_norm = float(torch.norm(tv_tensor, p="fro"))
            base_norm = float(torch.norm(base_sd[name].float(), p="fro")) if name in base_sd else 1.0
            rel_norm = tv_norm / base_norm if base_norm > 1e-12 else 0.0
            layer_profile[name] = rel_norm

            depth = get_block_depth(name)
            if depth >= 0:
                block_tv_norms[depth] += tv_norm ** 2  # sum of squares for Frobenius
                block_base_norms[depth] += base_norm ** 2

        # Per-block relative magnitude: sqrt(sum_sq_tv) / sqrt(sum_sq_base)
        block_profile = []
        for b in block_indices:
            tv_total = np.sqrt(block_tv_norms[b]) if block_tv_norms[b] > 0 else 0.0
            base_total = np.sqrt(block_base_norms[b]) if block_base_norms[b] > 0 else 1.0
            block_profile.append(tv_total / base_total if base_total > 1e-12 else 0.0)

        profiles[ds] = block_profile
        raw_layer_profiles[ds] = layer_profile

    # Log summary
    for ds in datasets:
        p = profiles[ds]
        front_half = np.mean(p[:len(p)//2])
        back_half = np.mean(p[len(p)//2:])
        logger.info(f"  {ds:12s}: front_avg={front_half:.4f}, back_avg={back_half:.4f}, "
                     f"ratio(back/front)={back_half/front_half:.2f}, "
                     f"max_block={np.argmax(p)}, max_val={np.max(p):.4f}")

    results = {
        "block_indices": block_indices,
        "profiles": {ds: profiles[ds] for ds in datasets},
        "raw_layer_profiles": {ds: raw_layer_profiles[ds] for ds in datasets},
    }
    return results


# --------------------------------------------------------------------------- #
# Analysis 2: Inter-layer Correlation Matrix
# --------------------------------------------------------------------------- #

def analyze_inter_layer_correlation(task_vectors, datasets, quick=False):
    """Compute cosine similarity between every pair of layers' flattened task vectors,
    averaged across tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: Inter-layer Correlation Matrix")
    logger.info("=" * 60)

    all_layers = list(task_vectors[datasets[0]].keys())
    # Focus on block layers for a meaningful depth-ordered heatmap
    block_layer_names = [name for name in all_layers if get_block_depth(name) >= 0]
    block_layer_names.sort(key=lambda n: (get_block_depth(n), n))

    if quick:
        # Subsample: take every other layer
        block_layer_names = block_layer_names[::2]

    n_layers = len(block_layer_names)
    logger.info(f"  Computing {n_layers}x{n_layers} inter-layer cosine similarity matrix...")

    # Accumulate across tasks
    sim_matrix_sum = np.zeros((n_layers, n_layers))
    n_tasks = len(datasets)

    for ds in datasets:
        # Flatten each layer's task vector
        flat_vectors = []
        for name in block_layer_names:
            flat_vectors.append(task_vectors[ds][name].float().flatten())

        # Compute pairwise cosine similarity (only for same-sized layers)
        for i in range(n_layers):
            for j in range(i, n_layers):
                if flat_vectors[i].shape[0] != flat_vectors[j].shape[0]:
                    # Cannot compute cosine similarity for different-sized layers
                    continue
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    flat_vectors[i].unsqueeze(0), flat_vectors[j].unsqueeze(0)
                ))
                sim_matrix_sum[i, j] += cos_sim
                sim_matrix_sum[j, i] += cos_sim

    sim_matrix = sim_matrix_sum / n_tasks

    # Analyze block structure: average within-block vs between-block similarity
    depths = [get_block_depth(n) for n in block_layer_names]
    within_block_sims = []
    between_adjacent_sims = []
    between_distant_sims = []

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            if depths[i] == depths[j]:
                within_block_sims.append(sim_matrix[i, j])
            elif abs(depths[i] - depths[j]) == 1:
                between_adjacent_sims.append(sim_matrix[i, j])
            else:
                between_distant_sims.append(sim_matrix[i, j])

    logger.info(f"  Within-block avg cosine sim:    {np.mean(within_block_sims):.4f} (n={len(within_block_sims)})")
    logger.info(f"  Adjacent-block avg cosine sim:  {np.mean(between_adjacent_sims):.4f} (n={len(between_adjacent_sims)})")
    logger.info(f"  Distant-block avg cosine sim:   {np.mean(between_distant_sims):.4f} (n={len(between_distant_sims)})")

    # Create short labels for plotting
    short_labels = []
    for name in block_layer_names:
        depth = get_block_depth(name)
        parts = name.split(".")
        sublayer = ".".join(parts[-2:]) if len(parts) > 1 else name
        short_labels.append(f"B{depth}.{sublayer}")

    results = {
        "layer_names": block_layer_names,
        "short_labels": short_labels,
        "sim_matrix": sim_matrix.tolist(),
        "within_block_avg": float(np.mean(within_block_sims)) if within_block_sims else 0.0,
        "adjacent_block_avg": float(np.mean(between_adjacent_sims)) if between_adjacent_sims else 0.0,
        "distant_block_avg": float(np.mean(between_distant_sims)) if between_distant_sims else 0.0,
    }
    return results


# --------------------------------------------------------------------------- #
# Analysis 3: Cross-task Layer Profile Similarity
# --------------------------------------------------------------------------- #

def analyze_cross_task_profile_similarity(profile_results, datasets):
    """Compute task similarity based on their layer magnitude profiles.
    Compare to cosine similarity and LOO interference matrices."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Cross-task Layer Profile Similarity")
    logger.info("=" * 60)

    profiles = profile_results["profiles"]
    n_tasks = len(datasets)

    # Profile similarity: Pearson correlation of magnitude vectors
    profile_sim = np.zeros((n_tasks, n_tasks))
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            p1 = np.array(profiles[ds1])
            p2 = np.array(profiles[ds2])
            if np.std(p1) > 1e-12 and np.std(p2) > 1e-12:
                profile_sim[i, j] = float(np.corrcoef(p1, p2)[0, 1])
            else:
                profile_sim[i, j] = 0.0

    logger.info("  Profile similarity matrix (Pearson correlation of layer magnitude profiles):")
    header = "".ljust(12) + "".join(ds[:7].ljust(8) for ds in datasets)
    logger.info(header)
    for i, ds in enumerate(datasets):
        row = ds[:11].ljust(12) + "".join(f"{profile_sim[i, j]:.3f}   " for j in range(n_tasks))
        logger.info(row)

    # Compare with LOO interference
    loo_data = load_loo_interference()
    comparison = {}
    if loo_data is not None:
        loo_matrix = np.array(loo_data["interaction_matrix"])
        loo_tasks = loo_data["tasks"]

        # Align task ordering
        task_idx = {t: i for i, t in enumerate(loo_tasks)}
        if all(ds in task_idx for ds in datasets):
            # Extract LOO interference for our task ordering
            loo_aligned = np.zeros((n_tasks, n_tasks))
            for i, ds1 in enumerate(datasets):
                for j, ds2 in enumerate(datasets):
                    val = loo_matrix[task_idx[ds1]][task_idx[ds2]]
                    loo_aligned[i, j] = val if val is not None else 0.0

            # Compute correlation between profile similarity and LOO interference
            # Use off-diagonal elements only
            off_diag_mask = ~np.eye(n_tasks, dtype=bool)
            profile_off = profile_sim[off_diag_mask]
            loo_off = loo_aligned[off_diag_mask]

            # Replace NaN in LOO (diagonal) with 0
            valid_mask = ~np.isnan(loo_off)
            if valid_mask.sum() > 2:
                corr = float(np.corrcoef(profile_off[valid_mask], loo_off[valid_mask])[0, 1])
                logger.info(f"\n  Correlation(profile_similarity, LOO_interference) = {corr:.4f}")
                comparison["profile_vs_loo_correlation"] = corr
                comparison["loo_interference_matrix"] = loo_aligned.tolist()
            else:
                logger.info("  Not enough valid LOO entries for correlation.")
    else:
        logger.info("  LOO interference data not found, skipping comparison.")

    results = {
        "profile_similarity_matrix": profile_sim.tolist(),
        "comparison_with_loo": comparison,
    }
    return results


# --------------------------------------------------------------------------- #
# Analysis 4: Task Vector as a Matrix -- SVD Across Layers
# --------------------------------------------------------------------------- #

def analyze_layer_type_svd(task_vectors, datasets, quick=False):
    """For each layer type (e.g. all attn.in_proj_weight across blocks 0-11),
    stack the 8 tasks' flattened vectors into a matrix (8 x n_params) and SVD it.
    Reports effective rank."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 4: SVD Across Layers (Task x Block Matrix)")
    logger.info("=" * 60)

    all_layers = list(task_vectors[datasets[0]].keys())

    # Group layers by type (canonical key with wildcard block index)
    layer_type_groups = defaultdict(list)  # type_key -> list of (layer_name, block_depth)
    for name in all_layers:
        depth = get_block_depth(name)
        if depth >= 0:
            type_key = get_layer_type_key(name)
            layer_type_groups[type_key].append((name, depth))

    # Sort each group by block depth
    for key in layer_type_groups:
        layer_type_groups[key].sort(key=lambda x: x[1])

    if quick:
        # Keep only a few representative types
        keys = list(layer_type_groups.keys())
        layer_type_groups = {k: layer_type_groups[k] for k in keys[:4]}

    n_tasks = len(datasets)
    results = {}

    logger.info(f"  {'Layer Type':<60s}  {'Shape':>12s}  {'Eff Rank':>10s}  {'SV Spectrum':>30s}")
    logger.info("-" * 120)

    for type_key, layer_list in sorted(layer_type_groups.items()):
        n_blocks = len(layer_list)
        if n_blocks < 2:
            continue  # need multiple blocks to analyze cross-layer structure

        # For each task, concatenate this layer type's values across all blocks
        # Result: matrix of shape (n_tasks, n_blocks * n_params_per_layer)
        task_block_vectors = []
        for ds in datasets:
            block_cat = []
            for layer_name, _ in layer_list:
                block_cat.append(task_vectors[ds][layer_name].float().flatten())
            task_block_vectors.append(torch.cat(block_cat))

        task_matrix = torch.stack(task_block_vectors)  # (n_tasks, total_params)

        # SVD: since n_tasks << total_params, use the Gram matrix approach
        # Gram = task_matrix @ task_matrix.T, shape (n_tasks, n_tasks)
        gram = task_matrix @ task_matrix.T
        eigenvalues, _ = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(0)  # descending
        eigenvalues = eigenvalues.clamp(min=0)  # numerical stability
        singular_values = torch.sqrt(eigenvalues).numpy()

        # Effective rank
        sv_pos = singular_values[singular_values > 1e-12]
        if len(sv_pos) > 0:
            p = sv_pos / sv_pos.sum()
            entropy = -np.sum(p * np.log(p))
            eff_rank = float(np.exp(entropy))
        else:
            eff_rank = 0.0

        # Cumulative variance explained
        total_var = float(eigenvalues.sum())
        cum_var = np.cumsum(singular_values ** 2) / total_var if total_var > 0 else np.zeros(len(singular_values))

        # How many SVs needed for 90%, 95%, 99%
        n_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        n_95 = int(np.searchsorted(cum_var, 0.95)) + 1
        n_99 = int(np.searchsorted(cum_var, 0.99)) + 1

        # Short label for display
        short_key = type_key.replace("visual.transformer.resblocks.*.", "")
        sv_str = ", ".join(f"{s:.2f}" for s in singular_values[:5])

        logger.info(f"  {short_key:<60s}  ({n_tasks}x{task_matrix.shape[1]:>7d})  "
                     f"{eff_rank:10.2f}  [{sv_str}]")

        results[type_key] = {
            "short_key": short_key,
            "n_blocks": n_blocks,
            "matrix_shape": [n_tasks, int(task_matrix.shape[1])],
            "singular_values": singular_values.tolist(),
            "effective_rank": eff_rank,
            "cumulative_variance": cum_var.tolist(),
            "n_for_90pct": n_90,
            "n_for_95pct": n_95,
            "n_for_99pct": n_99,
        }

        del task_matrix, gram  # free memory

    return results


# --------------------------------------------------------------------------- #
# Analysis 5: Layer-pair Co-variation
# --------------------------------------------------------------------------- #

def analyze_layer_pair_covariation(task_vectors, datasets, quick=False):
    """For each pair of same-type layers at different depths, compute the
    correlation of task vector values across tasks."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 5: Layer-pair Co-variation Across Tasks")
    logger.info("=" * 60)

    all_layers = list(task_vectors[datasets[0]].keys())

    # Group layers by type
    layer_type_groups = defaultdict(list)
    for name in all_layers:
        depth = get_block_depth(name)
        if depth >= 0:
            type_key = get_layer_type_key(name)
            layer_type_groups[type_key].append((name, depth))

    for key in layer_type_groups:
        layer_type_groups[key].sort(key=lambda x: x[1])

    if quick:
        keys = list(layer_type_groups.keys())
        layer_type_groups = {k: layer_type_groups[k] for k in keys[:3]}

    n_tasks = len(datasets)
    results = {}

    for type_key, layer_list in sorted(layer_type_groups.items()):
        n_blocks = len(layer_list)
        if n_blocks < 2:
            continue

        # For each layer in this group, extract per-task values
        # Shape per layer: (n_tasks, n_params)
        per_block_data = {}
        n_params = task_vectors[datasets[0]][layer_list[0][0]].numel()
        for layer_name, depth in layer_list:
            vals = torch.stack([task_vectors[ds][layer_name].float().flatten() for ds in datasets])
            per_block_data[depth] = vals  # (n_tasks, n_params)

        # For each pair of blocks, compute correlation across tasks for each parameter,
        # then average. To save memory, sample parameters if there are too many.
        max_params_sample = 10000
        if n_params > max_params_sample:
            param_indices = np.random.RandomState(42).choice(n_params, max_params_sample, replace=False)
        else:
            param_indices = np.arange(n_params)

        depths = sorted(per_block_data.keys())
        pair_corrs = np.zeros((n_blocks, n_blocks))

        for i, d1 in enumerate(depths):
            for j, d2 in enumerate(depths):
                if i >= j:
                    if i == j:
                        pair_corrs[i, j] = 1.0
                    continue

                v1 = per_block_data[d1][:, param_indices].numpy()  # (n_tasks, n_sampled)
                v2 = per_block_data[d2][:, param_indices].numpy()  # (n_tasks, n_sampled)

                # For each parameter position, compute correlation across tasks
                # This is: for each column p, corr(v1[:, p], v2[:, p])
                # Vectorized: standardize each column, then dot product / n_tasks
                if n_tasks > 2:
                    v1_centered = v1 - v1.mean(axis=0, keepdims=True)
                    v2_centered = v2 - v2.mean(axis=0, keepdims=True)
                    v1_std = np.sqrt((v1_centered ** 2).sum(axis=0) + 1e-12)
                    v2_std = np.sqrt((v2_centered ** 2).sum(axis=0) + 1e-12)
                    per_param_corr = (v1_centered * v2_centered).sum(axis=0) / (v1_std * v2_std)
                    avg_corr = float(np.mean(per_param_corr))
                else:
                    avg_corr = 0.0

                pair_corrs[i, j] = avg_corr
                pair_corrs[j, i] = avg_corr

        short_key = type_key.replace("visual.transformer.resblocks.*.", "")

        # Summary: adjacent vs distant correlations
        adjacent_corrs = [pair_corrs[i, i + 1] for i in range(n_blocks - 1)]
        all_off_diag = [pair_corrs[i, j] for i in range(n_blocks) for j in range(i + 1, n_blocks)]

        logger.info(f"  {short_key:<50s}: "
                     f"adjacent_avg={np.mean(adjacent_corrs):.4f}, "
                     f"all_pairs_avg={np.mean(all_off_diag):.4f}")

        results[type_key] = {
            "short_key": short_key,
            "depths": depths,
            "pair_correlation_matrix": pair_corrs.tolist(),
            "adjacent_avg_corr": float(np.mean(adjacent_corrs)),
            "all_pairs_avg_corr": float(np.mean(all_off_diag)),
            "n_params_sampled": len(param_indices),
        }

        # Free memory
        del per_block_data

    return results


# --------------------------------------------------------------------------- #
# Analysis 6: Principal Task Modes Across Layers (Full PCA)
# --------------------------------------------------------------------------- #

def analyze_full_pca(task_vectors, datasets):
    """PCA on the 8 x total_params matrix. Uses Gram matrix eigendecomposition
    for memory efficiency."""
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 6: Principal Task Modes (Full PCA)")
    logger.info("=" * 60)

    n_tasks = len(datasets)

    # Build Gram matrix: G[i,j] = <tv_i, tv_j> where tv is the full concatenated task vector
    logger.info("  Computing Gram matrix of full task vectors...")
    gram = np.zeros((n_tasks, n_tasks))
    all_layers = list(task_vectors[datasets[0]].keys())

    # Compute Gram matrix layer by layer to avoid materializing full vectors
    for layer_name in all_layers:
        flat_vecs = []
        for ds in datasets:
            flat_vecs.append(task_vectors[ds][layer_name].float().flatten())

        for i in range(n_tasks):
            for j in range(i, n_tasks):
                dot = float(torch.dot(flat_vecs[i], flat_vecs[j]))
                gram[i, j] += dot
                if i != j:
                    gram[j, i] += dot

    # Eigendecompose Gram matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negatives (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative_ratio = np.cumsum(explained_ratio)

    # Singular values = sqrt(eigenvalues)
    singular_values = np.sqrt(eigenvalues)

    logger.info(f"  Total variance: {total_var:.4f}")
    logger.info(f"  Singular values: [{', '.join(f'{s:.4f}' for s in singular_values)}]")
    logger.info(f"  Variance explained: [{', '.join(f'{r:.4f}' for r in explained_ratio)}]")
    logger.info(f"  Cumulative: [{', '.join(f'{c:.4f}' for c in cumulative_ratio)}]")

    # PCA projection: 2D coordinates for each task
    # The eigenvectors of the Gram matrix give coordinates in PCA space
    # Project: coords_i = eigenvectors[i, :] * sqrt(eigenvalues)
    pca_coords_2d = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    task_coords = {ds: pca_coords_2d[i].tolist() for i, ds in enumerate(datasets)}

    logger.info("\n  2D PCA coordinates:")
    for ds, coords in task_coords.items():
        logger.info(f"    {ds:12s}: ({coords[0]:.4f}, {coords[1]:.4f})")

    # Cosine similarity matrix from the Gram matrix
    norms = np.sqrt(np.diag(gram))
    cosine_sim = gram / (np.outer(norms, norms) + 1e-12)
    logger.info("\n  Full task vector cosine similarity matrix:")
    header = "".ljust(12) + "".join(ds[:7].ljust(8) for ds in datasets)
    logger.info(header)
    for i, ds in enumerate(datasets):
        row = ds[:11].ljust(12) + "".join(f"{cosine_sim[i, j]:.3f}   " for j in range(n_tasks))
        logger.info(row)

    # Effective dimensionality
    sv_pos = singular_values[singular_values > 1e-12]
    if len(sv_pos) > 0:
        p = sv_pos / sv_pos.sum()
        eff_dim = float(np.exp(-np.sum(p * np.log(p))))
    else:
        eff_dim = 0.0
    logger.info(f"\n  Effective dimensionality: {eff_dim:.2f} / {n_tasks}")

    results = {
        "eigenvalues": eigenvalues.tolist(),
        "singular_values": singular_values.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_variance_ratio": cumulative_ratio.tolist(),
        "effective_dimensionality": eff_dim,
        "pca_2d_coords": task_coords,
        "cosine_similarity_matrix": cosine_sim.tolist(),
        "gram_matrix": gram.tolist(),
    }
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_layer_magnitude_profiles(profile_results, datasets, output_dir):
    """Plot layer magnitude profiles for all tasks overlaid."""
    logger.info("Plotting layer magnitude profiles...")

    profiles = profile_results["profiles"]
    block_indices = profile_results["block_indices"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ds in datasets:
        ax.plot(block_indices, profiles[ds], "o-", label=ds, markersize=4, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Transformer Block", fontsize=11)
    ax.set_ylabel("Relative Change (||TV|| / ||pretrained||)", fontsize=11)
    ax.set_title("Layer Magnitude Profile Per Task\n(Frobenius norm of task vector / pretrained weight norm)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(block_indices)
    plt.tight_layout()
    plt.savefig(output_dir / "layer_magnitude_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_inter_layer_correlation(corr_results, output_dir):
    """Plot heatmap of inter-layer cosine similarity."""
    logger.info("Plotting inter-layer correlation heatmap...")

    sim_matrix = np.array(corr_results["sim_matrix"])
    short_labels = corr_results["short_labels"]
    n = len(short_labels)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.35), max(10, n * 0.3)))
    vmax = max(abs(sim_matrix.min()), abs(sim_matrix.max()))
    vmax = min(vmax, 1.0)
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(short_labels, fontsize=5)

    plt.colorbar(im, label="Cosine Similarity (avg across tasks)")
    ax.set_title("Inter-layer Cosine Similarity of Task Vectors\n(averaged across tasks)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "inter_layer_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_task_profile_similarity(profile_sim_results, datasets, output_dir):
    """Plot 8x8 heatmap of task profile similarity."""
    logger.info("Plotting task profile similarity heatmap...")

    sim_matrix = np.array(profile_sim_results["profile_similarity_matrix"])
    n_tasks = len(datasets)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(datasets, fontsize=9)

    for i in range(n_tasks):
        for j in range(n_tasks):
            text_color = "white" if abs(sim_matrix[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    plt.colorbar(im, label="Pearson Correlation of Layer Profiles")
    ax.set_title("Cross-task Layer Profile Similarity", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "task_profile_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_type_svd(svd_results, output_dir):
    """Bar chart showing effective rank and variance explained for each layer type."""
    logger.info("Plotting layer type SVD results...")

    type_keys = sorted(svd_results.keys())
    short_keys = [svd_results[k]["short_key"] for k in type_keys]
    eff_ranks = [svd_results[k]["effective_rank"] for k in type_keys]
    n_90s = [svd_results[k]["n_for_90pct"] for k in type_keys]
    n_95s = [svd_results[k]["n_for_95pct"] for k in type_keys]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(type_keys) * 0.8), 8))

    x = np.arange(len(type_keys))

    # Top: effective rank
    bars = axes[0].bar(x, eff_ranks, color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Effective Rank", fontsize=11)
    axes[0].set_title("Effective Rank of (Tasks x Parameters) Matrix per Layer Type", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_keys, rotation=45, ha="right", fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, eff_ranks):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Bottom: number of components for 90% and 95% variance
    width = 0.35
    axes[1].bar(x - width / 2, n_90s, width, label="Components for 90%", color="coral", alpha=0.8)
    axes[1].bar(x + width / 2, n_95s, width, label="Components for 95%", color="goldenrod", alpha=0.8)
    axes[1].set_ylabel("Number of Components", fontsize=11)
    axes[1].set_xlabel("Layer Type", fontsize=11)
    axes[1].set_title("Components Needed for Variance Explained Thresholds", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_keys, rotation=45, ha="right", fontsize=7)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "layer_type_svd.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_full_tv_pca(pca_results, datasets, output_dir):
    """Scree plot + 2D PCA projection of the 8 tasks in weight space."""
    logger.info("Plotting full task vector PCA...")

    explained = np.array(pca_results["explained_variance_ratio"])
    cumulative = np.array(pca_results["cumulative_variance_ratio"])
    coords = pca_results["pca_2d_coords"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scree plot
    n_components = len(explained)
    x = np.arange(1, n_components + 1)
    axes[0].bar(x, explained, color="steelblue", alpha=0.8, label="Individual")
    axes[0].plot(x, cumulative, "ro-", markersize=6, label="Cumulative")
    axes[0].set_xlabel("Principal Component", fontsize=11)
    axes[0].set_ylabel("Variance Explained", fontsize=11)
    axes[0].set_title("Scree Plot: PCA of Full Task Vectors", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(y=0.9, color="gray", linestyle="--", alpha=0.4, label="90%")
    # Annotate effective dimensionality
    eff_dim = pca_results["effective_dimensionality"]
    axes[0].text(0.95, 0.95, f"Eff. dim = {eff_dim:.2f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Right: 2D projection
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    for i, ds in enumerate(datasets):
        c = coords[ds]
        axes[1].scatter(c[0], c[1], s=100, color=colors[i], zorder=5)
        axes[1].annotate(ds, (c[0], c[1]), textcoords="offset points", xytext=(8, 5),
                         fontsize=9, color=colors[i], fontweight="bold")

    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%} var)", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%} var)", fontsize=11)
    axes[1].set_title("2D PCA Projection of Task Vectors", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.2)
    axes[1].axvline(x=0, color="gray", linestyle="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "full_tv_pca.png", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Cross-layer structure analysis of task vectors from N8 ViT-B-32 models"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 tasks, subset of layers")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        datasets = N8_DATASETS[:3]
        logger.info(f"QUICK MODE: Using {len(datasets)} datasets: {datasets}")
    else:
        datasets = N8_DATASETS
        logger.info(f"Full analysis: {len(datasets)} datasets: {datasets}")

    # Load task vectors and base state dict
    task_vectors, base_sd = load_task_vectors(MODEL_NAME, datasets)

    all_results = {}
    json_path = OUTPUT_DIR / "cross_layer_analysis.json"

    def save_intermediate():
        with open(json_path, "w") as f:
            json.dump(make_json_serializable(all_results), f, indent=2)
        logger.info(f"Intermediate results saved to: {json_path}")

    # --- Analysis 1: Layer magnitude profiles ---
    profile_results = analyze_layer_magnitude_profiles(task_vectors, base_sd, datasets)
    all_results["layer_magnitude_profiles"] = profile_results
    save_intermediate()

    # Free base_sd -- no longer needed
    del base_sd

    # --- Analysis 2: Inter-layer correlation matrix ---
    inter_layer_results = analyze_inter_layer_correlation(task_vectors, datasets, quick=args.quick)
    all_results["inter_layer_correlation"] = inter_layer_results
    save_intermediate()

    # --- Analysis 3: Cross-task layer profile similarity ---
    profile_sim_results = analyze_cross_task_profile_similarity(profile_results, datasets)
    all_results["cross_task_profile_similarity"] = profile_sim_results
    save_intermediate()

    # --- Analysis 4: SVD across layers ---
    svd_results = analyze_layer_type_svd(task_vectors, datasets, quick=args.quick)
    all_results["layer_type_svd"] = svd_results
    save_intermediate()

    # --- Analysis 5: Layer-pair co-variation ---
    covar_results = analyze_layer_pair_covariation(task_vectors, datasets, quick=args.quick)
    all_results["layer_pair_covariation"] = covar_results
    save_intermediate()

    # --- Analysis 6: Full PCA ---
    pca_results = analyze_full_pca(task_vectors, datasets)
    all_results["full_pca"] = pca_results
    save_intermediate()

    logger.info(f"\nFull results saved to: {json_path}")

    # --- Generate plots ---
    logger.info("\nGenerating plots...")
    plot_layer_magnitude_profiles(profile_results, datasets, OUTPUT_DIR)
    plot_inter_layer_correlation(inter_layer_results, OUTPUT_DIR)
    plot_task_profile_similarity(profile_sim_results, datasets, OUTPUT_DIR)
    plot_layer_type_svd(svd_results, OUTPUT_DIR)
    plot_full_tv_pca(pca_results, datasets, OUTPUT_DIR)

    logger.info(f"\nAll plots saved to: {OUTPUT_DIR}")
    logger.info("Cross-layer analysis complete.")


if __name__ == "__main__":
    main()
