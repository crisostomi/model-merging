"""
Head Swapping Analysis: Does Restoring Fine-Tuned Block 11 Weights Recover Accuracy?

Tests whether replacing specific layers of the merged model with fine-tuned weights
recovers per-task accuracy. For each task k, we create "hybrid" models:

1. Merged (baseline) -- the standard MP-edge + isotropic merged model
2. Attn-swap -- merged model with block 11 attention weights from task k's fine-tuned model
3. Block-swap -- merged model with all of block 11 from task k's fine-tuned model
4. Late-blocks-swap -- merged model with blocks 9-11 from task k's fine-tuned model

Evaluation uses linear probes (LogisticRegression) on extracted features, which is
fast, GPU-efficient, and diagnostic.

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_head_swapping.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_head_swapping.py --quick
"""

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "head_swapping"

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

# Layer key patterns for swapping
BLOCK_11_ATTN_PREFIX = "model.visual.transformer.resblocks.11.attn."
BLOCK_11_PREFIX = "model.visual.transformer.resblocks.11."
LATE_BLOCK_PREFIXES = [
    "model.visual.transformer.resblocks.9.",
    "model.visual.transformer.resblocks.10.",
    "model.visual.transformer.resblocks.11.",
]

N_TRAIN = 5000
N_TEST = 1000
BATCH_SIZE = 256


# --------------------------------------------------------------------------- #
# Model loading (reuses patterns from existing analysis scripts)
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def create_encoder(model_name=MODEL_NAME):
    return ImageEncoder(model_name)


def load_finetuned_encoder(dataset_name, model_name=MODEL_NAME):
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
        use_mp_edge=True, use_isotropic=True, mp_min_rank=4, mp_max_rank=128
    )
    # Patch functions that default to device="cuda" so they work on CPU-only nodes
    if not torch.cuda.is_available():
        import functools
        import model_merging.merger.interference_aware_merger as _iam
        import model_merging.utils.utils as _utils

        _orig_agg = _iam.aggregate_interference_aware
        _orig_apply = _utils.apply_dict_to_model

        @functools.wraps(_orig_agg)
        def _cpu_agg(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_agg(*args, **kwargs)

        @functools.wraps(_orig_apply)
        def _cpu_apply(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return _orig_apply(*args, **kwargs)

        _iam.aggregate_interference_aware = _cpu_agg
        # Patch both the module-level reference and the one imported in the merger module
        _utils.apply_dict_to_model = _cpu_apply
        _iam.apply_dict_to_model = _cpu_apply

    try:
        merged = merger.merge(encoder, finetuned)
    finally:
        if not torch.cuda.is_available():
            _iam.aggregate_interference_aware = _orig_agg
            _utils.apply_dict_to_model = _orig_apply
            _iam.apply_dict_to_model = _orig_apply
    return merged


# --------------------------------------------------------------------------- #
# Weight swapping logic
# --------------------------------------------------------------------------- #

def swap_weights(merged_sd, finetuned_sd, key_filter_fn):
    """Create a hybrid state dict: merged weights with specific keys replaced by finetuned.

    Args:
        merged_sd: merged model state dict
        finetuned_sd: fine-tuned model state dict
        key_filter_fn: function(key) -> bool; True means replace with finetuned

    Returns:
        New state dict with swapped weights
    """
    hybrid_sd = OrderedDict()
    swapped_keys = []
    for key in merged_sd:
        if key_filter_fn(key) and key in finetuned_sd:
            hybrid_sd[key] = finetuned_sd[key].clone()
            swapped_keys.append(key)
        else:
            hybrid_sd[key] = merged_sd[key].clone()
    return hybrid_sd, swapped_keys


def is_block11_attn(key):
    """Match block 11 attention weights (in_proj, out_proj)."""
    return key.startswith(BLOCK_11_ATTN_PREFIX)


def is_block11_all(key):
    """Match all of block 11 (attn + MLP + LayerNorm)."""
    return key.startswith(BLOCK_11_PREFIX)


def is_late_blocks(key):
    """Match blocks 9, 10, 11 entirely."""
    return any(key.startswith(prefix) for prefix in LATE_BLOCK_PREFIXES)


def create_hybrid_encoder(merged_encoder, finetuned_sd, key_filter_fn, swap_name=""):
    """Create a hybrid encoder by swapping specific weights.

    Args:
        merged_encoder: the merged ImageEncoder
        finetuned_sd: fine-tuned model state dict (raw, from HF)
        key_filter_fn: which keys to swap
        swap_name: descriptive name for logging

    Returns:
        New ImageEncoder with hybrid weights
    """
    merged_sd = merged_encoder.state_dict()
    hybrid_sd, swapped_keys = swap_weights(merged_sd, finetuned_sd, key_filter_fn)
    logger.info(f"  [{swap_name}] Swapped {len(swapped_keys)} keys")
    if swapped_keys:
        logger.info(f"    First few: {swapped_keys[:5]}")

    hybrid_encoder = create_encoder(MODEL_NAME)
    hybrid_encoder.load_state_dict(hybrid_sd, strict=False)
    return hybrid_encoder


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataset_samples(dataset_name, preprocess, n_train=N_TRAIN, n_test=N_TEST):
    """Load train and test samples from a dataset via HuggingFace.

    Returns:
        train_images, train_labels, test_images, test_labels
    """
    from datasets import load_dataset as hf_load_dataset

    hf_name, config, img_col, lbl_col = DATASET_HF_MAP[dataset_name]
    logger.info(f"Loading {dataset_name} ({hf_name}, config={config})...")

    def load_split(split_name, n_samples):
        if config:
            ds = hf_load_dataset(hf_name, config, split=split_name)
        else:
            ds = hf_load_dataset(hf_name, split=split_name)

        # Subsample with random indices to ensure class diversity
        # (ds.shuffle() can silently fail on some dataset formats)
        n_avail = len(ds)
        n_use = min(n_samples, n_avail)
        rng = np.random.default_rng(42)
        indices = rng.choice(n_avail, size=n_use, replace=False).tolist()
        ds = ds.select(indices)

        images, labels = [], []
        for sample in ds:
            try:
                img = preprocess(sample[img_col].convert("RGB"))
                images.append(img)
                labels.append(sample[lbl_col])
            except Exception:
                continue
        return torch.stack(images), np.array(labels)

    train_imgs, train_labels = load_split("train", n_train)
    test_imgs, test_labels = load_split("test", n_test)
    logger.info(f"  {dataset_name}: train={len(train_imgs)}, test={len(test_imgs)}, "
                f"classes={len(np.unique(train_labels))}")
    return train_imgs, train_labels, test_imgs, test_labels


# --------------------------------------------------------------------------- #
# Feature extraction + linear probe evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_features(model, images, device, batch_size=BATCH_SIZE):
    """Extract features from images using the model."""
    model.eval()
    model.to(device)
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        feat = model(batch)
        features.append(feat.cpu())
    model.cpu()
    torch.cuda.empty_cache()
    return torch.cat(features, dim=0).numpy()


def train_and_evaluate_probe(train_features, train_labels, test_features, test_labels, C=0.316):
    """Train a logistic regression probe and return test accuracy."""
    train_norm = normalize(train_features, norm="l2")
    test_norm = normalize(test_features, norm="l2")

    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(train_norm, train_labels)
    train_acc = clf.score(train_norm, train_labels)
    test_acc = clf.score(test_norm, test_labels)
    return float(train_acc), float(test_acc)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

SWAP_COLORS = {
    "merged": "#4e79a7",
    "attn_swap": "#f28e2b",
    "block_swap": "#e15759",
    "late_blocks_swap": "#76b7b2",
}

SWAP_LABELS = {
    "merged": "Merged (baseline)",
    "attn_swap": "Block 11 Attn swap",
    "block_swap": "Block 11 full swap",
    "late_blocks_swap": "Blocks 9-11 swap",
}


def plot_comparison(results, output_dir):
    """Grouped bar chart: tasks x [merged, attn_swap, block_swap, late_blocks_swap]."""
    tasks = list(results.keys())
    variants = ["merged", "attn_swap", "block_swap", "late_blocks_swap"]
    n_tasks = len(tasks)
    n_variants = len(variants)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 6))
    x = np.arange(n_tasks)
    width = 0.8 / n_variants

    for i, variant in enumerate(variants):
        accs = [results[t][variant]["test_acc"] for t in tasks]
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=SWAP_LABELS[variant],
                      color=SWAP_COLORS[variant], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Linear Probe Test Accuracy", fontsize=11)
    ax.set_title("Head Swapping: Restoring Fine-Tuned Weights in Merged Model", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / "head_swapping_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_delta(results, output_dir):
    """Delta from merged baseline for each swap variant."""
    tasks = list(results.keys())
    variants = ["attn_swap", "block_swap", "late_blocks_swap"]
    n_tasks = len(tasks)
    n_variants = len(variants)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.5), 6))
    x = np.arange(n_tasks)
    width = 0.8 / n_variants

    for i, variant in enumerate(variants):
        deltas = [
            results[t][variant]["test_acc"] - results[t]["merged"]["test_acc"]
            for t in tasks
        ]
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=SWAP_LABELS[variant],
                      color=SWAP_COLORS[variant], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, deltas):
            y_pos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=7, rotation=45)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Delta Accuracy (vs Merged Baseline)", fontsize=11)
    ax.set_title("Accuracy Change from Swapping Fine-Tuned Weights into Merged Model", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "head_swapping_delta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


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
        description="Head swapping analysis: restore fine-tuned block 11 weights in merged model"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 tasks, 200 test samples")
    parser.add_argument("--n-train", type=int, default=N_TRAIN,
                        help=f"Training samples per task for probes (default: {N_TRAIN})")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help=f"Test samples per task (default: {N_TEST})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if args.quick:
        tasks = ["SVHN", "EuroSAT", "MNIST"]
        n_train = 1000
        n_test = 200
    else:
        tasks = N8_DATASETS
        n_train = args.n_train
        n_test = args.n_test

    logger.info(f"Tasks: {tasks}")
    logger.info(f"Samples: train={n_train}, test={n_test}")

    # ======================================================================= #
    # Phase 1: Create models
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 1: Creating merged model")
    logger.info("=" * 70)

    merged_encoder = create_merged_model(N8_DATASETS, MODEL_NAME)
    merged_sd = merged_encoder.state_dict()
    preprocess = merged_encoder.val_preprocess
    print_memory("after merged model")

    # Load fine-tuned state dicts for all tasks we will evaluate
    logger.info("Loading fine-tuned state dicts...")
    finetuned_sds = {}
    for ds in tasks:
        logger.info(f"  Loading: {ds}")
        finetuned_sds[ds] = load_state_dict_from_hf_cache(MODEL_NAME, ds)
    print_memory("after loading finetuned sds")

    # ======================================================================= #
    # Phase 2: Load data for each task
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 2: Loading data for all tasks")
    logger.info("=" * 70)

    task_data = {}
    for ds in tasks:
        train_imgs, train_labels, test_imgs, test_labels = load_dataset_samples(
            ds, preprocess, n_train=n_train, n_test=n_test
        )
        task_data[ds] = {
            "train_imgs": train_imgs,
            "train_labels": train_labels,
            "test_imgs": test_imgs,
            "test_labels": test_labels,
        }
    print_memory("after loading data")

    # ======================================================================= #
    # Phase 3: Extract features from merged model (baseline, computed once)
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting merged model features (baseline)")
    logger.info("=" * 70)

    merged_features = {}
    for ds in tasks:
        logger.info(f"  Extracting features for {ds}...")
        train_feats = extract_features(merged_encoder, task_data[ds]["train_imgs"], device)
        test_feats = extract_features(merged_encoder, task_data[ds]["test_imgs"], device)
        merged_features[ds] = {"train": train_feats, "test": test_feats}
    print_memory("after merged feature extraction")

    # ======================================================================= #
    # Phase 4: Train probes and evaluate baseline
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 4: Baseline evaluation (merged model)")
    logger.info("=" * 70)

    results = {}
    for ds in tasks:
        train_acc, test_acc = train_and_evaluate_probe(
            merged_features[ds]["train"], task_data[ds]["train_labels"],
            merged_features[ds]["test"], task_data[ds]["test_labels"],
        )
        logger.info(f"  {ds}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
        results[ds] = {
            "merged": {
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        }

    # Save intermediate results
    _save_intermediate(results, OUTPUT_DIR, "after_baseline")

    # ======================================================================= #
    # Phase 5: Head swapping experiments
    # ======================================================================= #
    swap_configs = [
        ("attn_swap", is_block11_attn, "Block 11 attention"),
        ("block_swap", is_block11_all, "Block 11 full"),
        ("late_blocks_swap", is_late_blocks, "Blocks 9-11"),
    ]

    for swap_name, key_filter, description in swap_configs:
        logger.info("=" * 70)
        logger.info(f"PHASE 5: {description} swapping")
        logger.info("=" * 70)

        for ds in tasks:
            logger.info(f"\n--- {ds}: {description} swap ---")

            # Create hybrid encoder
            hybrid_encoder = create_hybrid_encoder(
                merged_encoder, finetuned_sds[ds], key_filter, swap_name=f"{ds}/{swap_name}"
            )

            # Extract features
            train_feats = extract_features(hybrid_encoder, task_data[ds]["train_imgs"], device)
            test_feats = extract_features(hybrid_encoder, task_data[ds]["test_imgs"], device)

            # Train probe and evaluate
            train_acc, test_acc = train_and_evaluate_probe(
                train_feats, task_data[ds]["train_labels"],
                test_feats, task_data[ds]["test_labels"],
            )

            delta = test_acc - results[ds]["merged"]["test_acc"]
            logger.info(f"  {ds} [{swap_name}]: test_acc={test_acc:.4f} "
                        f"(delta={delta:+.4f} vs merged)")

            results[ds][swap_name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "delta_vs_merged": delta,
            }

            # Free memory
            del hybrid_encoder, train_feats, test_feats
            torch.cuda.empty_cache()

        # Save intermediate results after each swap type
        _save_intermediate(results, OUTPUT_DIR, f"after_{swap_name}")

    # ======================================================================= #
    # Phase 6: Cross-task interference check
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 6: Cross-task interference (optional)")
    logger.info("=" * 70)
    logger.info("Checking if swapping task k's weights hurts OTHER tasks...")

    # For each swap type, pick one representative task and measure effect on all others.
    # Use the most aggressive swap (late_blocks_swap) for this check.
    cross_task_results = {}
    # Pick a few representative source tasks
    source_tasks = tasks[:3] if len(tasks) > 3 else tasks

    for source_ds in source_tasks:
        logger.info(f"\n--- Source: {source_ds} (late_blocks_swap) ---")
        hybrid_encoder = create_hybrid_encoder(
            merged_encoder, finetuned_sds[source_ds], is_late_blocks,
            swap_name=f"cross/{source_ds}"
        )

        cross_task_results[source_ds] = {}
        for target_ds in tasks:
            train_feats = extract_features(hybrid_encoder, task_data[target_ds]["train_imgs"], device)
            test_feats = extract_features(hybrid_encoder, task_data[target_ds]["test_imgs"], device)

            _, test_acc = train_and_evaluate_probe(
                train_feats, task_data[target_ds]["train_labels"],
                test_feats, task_data[target_ds]["test_labels"],
            )
            baseline = results[target_ds]["merged"]["test_acc"]
            delta = test_acc - baseline
            logger.info(f"  Target {target_ds}: test_acc={test_acc:.4f} "
                        f"(delta={delta:+.4f} vs merged)")
            cross_task_results[source_ds][target_ds] = {
                "test_acc": test_acc,
                "delta_vs_merged": delta,
            }

        del hybrid_encoder
        torch.cuda.empty_cache()

    # ======================================================================= #
    # Phase 7: Summary
    # ======================================================================= #
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    header = f"{'Task':<12} {'Merged':>8} {'Attn':>8} {'Block':>8} {'Late':>8} {'Best':>8} {'Best Var':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for ds in tasks:
        merged_acc = results[ds]["merged"]["test_acc"]
        attn_acc = results[ds]["attn_swap"]["test_acc"]
        block_acc = results[ds]["block_swap"]["test_acc"]
        late_acc = results[ds]["late_blocks_swap"]["test_acc"]

        best_acc = max(attn_acc, block_acc, late_acc)
        best_var = "attn" if best_acc == attn_acc else ("block" if best_acc == block_acc else "late")
        best_delta = best_acc - merged_acc

        logger.info(f"{ds:<12} {merged_acc:>8.4f} {attn_acc:>8.4f} {block_acc:>8.4f} "
                    f"{late_acc:>8.4f} {best_delta:>+8.4f} {best_var:>12}")

    # Cross-task interference summary
    if cross_task_results:
        logger.info("\nCross-task interference (late_blocks_swap):")
        logger.info(f"{'Source':<12} {'Self delta':>10} {'Mean other delta':>16} {'Worst other':>14}")
        logger.info("-" * 55)
        for source_ds in source_tasks:
            self_delta = cross_task_results[source_ds][source_ds]["delta_vs_merged"]
            other_deltas = [
                cross_task_results[source_ds][t]["delta_vs_merged"]
                for t in tasks if t != source_ds
            ]
            mean_other = np.mean(other_deltas) if other_deltas else 0.0
            worst_other = min(other_deltas) if other_deltas else 0.0
            logger.info(f"{source_ds:<12} {self_delta:>+10.4f} {mean_other:>+16.4f} {worst_other:>+14.4f}")

    # ======================================================================= #
    # Phase 8: Generate plots + save results
    # ======================================================================= #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 8: Generating plots and saving results")
    logger.info("=" * 70)

    plot_comparison(results, OUTPUT_DIR)
    plot_delta(results, OUTPUT_DIR)

    # Save final results
    final_output = {
        "metadata": {
            "model": MODEL_NAME,
            "merge_tasks": N8_DATASETS,
            "eval_tasks": tasks,
            "n_train": n_train,
            "n_test": n_test,
            "merger": "InterferenceAwareMerger(mp_edge=True, isotropic=True)",
            "swap_configs": {
                "attn_swap": "Block 11 attention (in_proj, out_proj)",
                "block_swap": "Block 11 full (attn + MLP + LN)",
                "late_blocks_swap": "Blocks 9-11 full",
            },
            "evaluation": "Linear probe (LogisticRegression, C=0.316, L2-normalized features)",
            "quick_mode": args.quick,
        },
        "per_task_results": results,
        "cross_task_interference": cross_task_results,
    }

    json_path = OUTPUT_DIR / "head_swapping_results.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(final_output), f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info("Head swapping analysis complete.")


def _save_intermediate(results, output_dir, stage_name):
    """Save intermediate results in case the script is killed."""
    path = output_dir / f"_intermediate_{stage_name}.json"
    with open(path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info(f"  Intermediate results saved: {path}")


if __name__ == "__main__":
    main()
