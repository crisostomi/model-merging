"""
Feature Alignment Analysis: Can per-task alignment transforms recover merged-model accuracy?

The merged model (MP-edge + isotropic on N8 ViT-B-32) loses accuracy because its features
are rotated relative to what fine-tuned classification heads expect. This script tests
whether lightweight alignment transforms can undo that rotation:

1. Procrustes alignment: orthogonal rotation R that minimizes ||merged @ R - finetuned||
2. Affine alignment: W, b that minimize ||merged @ W + b - finetuned|| (rotation + scaling)
3. Retrained probe: linear probe trained directly on merged features (no alignment)
4. Aligned + probe: alignment transform then probe (trained on fine-tuned features)
5. Identity baseline: merged features + probe trained on merged features
6. Fine-tuned baseline: fine-tuned features + probe (upper bound)

For each approach, we evaluate test accuracy to quantify how much of the accuracy loss
is due to feature rotation vs genuine information loss.

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_feature_alignment.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_feature_alignment.py --quick
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "feature_alignment"

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

N_TRAIN = 5000
N_TEST = 1000
BATCH_SIZE = 256

# Alignment method names and display labels
METHOD_NAMES = [
    "merged_probe",
    "procrustes_probe",
    "affine_probe",
    "finetuned_probe",
]

METHOD_LABELS = {
    "merged_probe": "Merged (probe on merged feats)",
    "procrustes_probe": "Procrustes align + FT probe",
    "affine_probe": "Affine align + FT probe",
    "finetuned_probe": "Fine-tuned (upper bound)",
}

METHOD_COLORS = {
    "merged_probe": "#4e79a7",
    "procrustes_probe": "#f28e2b",
    "affine_probe": "#e15759",
    "finetuned_probe": "#59a14f",
}


# --------------------------------------------------------------------------- #
# Model loading (same patterns as analyze_head_swapping.py)
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
# Data loading (same as analyze_head_swapping.py)
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
# Feature extraction
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


# --------------------------------------------------------------------------- #
# Alignment methods
# --------------------------------------------------------------------------- #

def compute_procrustes_rotation(merged_feats, finetuned_feats):
    """Compute orthogonal Procrustes rotation: R = V @ U^T.

    Finds the orthogonal matrix R that minimizes ||merged @ R - finetuned||_F.

    Args:
        merged_feats: (N, D) array of merged model features
        finetuned_feats: (N, D) array of fine-tuned model features

    Returns:
        R: (D, D) orthogonal rotation matrix
        residual: Frobenius norm of the residual after alignment
    """
    # Center features for better Procrustes fit
    merged_mean = merged_feats.mean(axis=0)
    ft_mean = finetuned_feats.mean(axis=0)
    M = merged_feats - merged_mean
    F = finetuned_feats - ft_mean

    # SVD of cross-covariance matrix: M^T @ F = U S V^T
    # Optimal rotation: R = V @ U^T
    cross_cov = M.T @ F  # (D, D)
    U, S, Vt = np.linalg.svd(cross_cov)
    R = Vt.T @ U.T  # (D, D)

    # Ensure proper rotation (det = +1), not reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute residual
    aligned = (merged_feats - merged_mean) @ R + ft_mean
    residual = np.linalg.norm(aligned - finetuned_feats, "fro")

    return R, merged_mean, ft_mean, float(residual)


def apply_procrustes(features, R, merged_mean, ft_mean):
    """Apply Procrustes transform: center, rotate, re-center."""
    return (features - merged_mean) @ R + ft_mean


def compute_affine_transform(merged_feats, finetuned_feats, reg_lambda=1e-3):
    """Compute affine alignment: W, b that minimize ||merged @ W + b - finetuned||.

    Uses ridge regression (regularized least squares) for stability.

    Args:
        merged_feats: (N, D) array of merged model features
        finetuned_feats: (N, D) array of fine-tuned model features
        reg_lambda: regularization strength

    Returns:
        W: (D, D) weight matrix
        b: (D,) bias vector
        residual: Frobenius norm of the residual
    """
    N, D = merged_feats.shape

    # Augment X with ones column for bias: X_aug = [merged, 1]
    X_aug = np.hstack([merged_feats, np.ones((N, 1))])  # (N, D+1)
    Y = finetuned_feats  # (N, D)

    # Ridge regression: (X^T X + lambda I)^{-1} X^T Y
    XtX = X_aug.T @ X_aug  # (D+1, D+1)
    XtX += reg_lambda * np.eye(D + 1)
    XtY = X_aug.T @ Y  # (D+1, D)
    params = np.linalg.solve(XtX, XtY)  # (D+1, D)

    W = params[:D, :]  # (D, D)
    b = params[D, :]   # (D,)

    # Compute residual
    aligned = merged_feats @ W + b
    residual = np.linalg.norm(aligned - finetuned_feats, "fro")

    return W, b, float(residual)


def apply_affine(features, W, b):
    """Apply affine transform: features @ W + b."""
    return features @ W + b


# --------------------------------------------------------------------------- #
# Linear probe training and evaluation
# --------------------------------------------------------------------------- #

def train_probe(train_features, train_labels, C=0.316):
    """Train a logistic regression probe and return the fitted model."""
    train_norm = normalize(train_features, norm="l2")
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(train_norm, train_labels)
    return clf


def evaluate_probe(clf, features, labels):
    """Evaluate a fitted probe on given features."""
    feat_norm = normalize(features, norm="l2")
    return float(clf.score(feat_norm, labels))


def train_and_evaluate_probe(train_features, train_labels, test_features, test_labels, C=0.316):
    """Train a logistic regression probe and return train/test accuracy."""
    clf = train_probe(train_features, train_labels, C=C)
    train_acc = evaluate_probe(clf, train_features, train_labels)
    test_acc = evaluate_probe(clf, test_features, test_labels)
    return float(train_acc), float(test_acc)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_comparison(results, output_dir):
    """Grouped bar chart: tasks x alignment methods."""
    tasks = list(results.keys())
    methods = METHOD_NAMES
    n_tasks = len(tasks)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.8), 6))
    x = np.arange(n_tasks)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        accs = [results[t][method]["test_acc"] for t in tasks]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Linear Probe Test Accuracy", fontsize=11)
    ax.set_title("Feature Alignment: Can Transforms Recover Merged-Model Accuracy?", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / "alignment_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_delta(results, output_dir):
    """Delta from merged baseline for each alignment method."""
    tasks = list(results.keys())
    methods = [m for m in METHOD_NAMES if m != "merged_probe"]
    n_tasks = len(tasks)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.8), 6))
    x = np.arange(n_tasks)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        deltas = [
            results[t][method]["test_acc"] - results[t]["merged_probe"]["test_acc"]
            for t in tasks
        ]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, deltas):
            y_pos = bar.get_height() + 0.003 if val >= 0 else bar.get_height() - 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=7, rotation=45)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Delta Accuracy (vs Merged Probe Baseline)", fontsize=11)
    ax.set_title("Feature Alignment: Accuracy Gain over Merged Baseline", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "alignment_delta.png"
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
# Intermediate saving
# --------------------------------------------------------------------------- #

def _save_intermediate(results, output_dir, stage_name):
    """Save intermediate results in case the script is killed."""
    path = output_dir / f"_intermediate_{stage_name}.json"
    with open(path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info(f"  Intermediate results saved: {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Feature alignment analysis: test whether per-task alignment "
                    "transforms can recover accuracy lost in model merging"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 tasks (SVHN, EuroSAT, MNIST), 1000 train, 200 test")
    parser.add_argument("--n-train", type=int, default=N_TRAIN,
                        help=f"Training samples per task (default: {N_TRAIN})")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help=f"Test samples per task (default: {N_TEST})")
    parser.add_argument("--reg-lambda", type=float, default=1e-3,
                        help="Regularization for affine alignment (default: 1e-3)")
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
    logger.info(f"Affine regularization: {args.reg_lambda}")

    # ======================================================================= #
    # Phase 1: Create merged model
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 1: Creating merged model")
    logger.info("=" * 70)

    merged_encoder = create_merged_model(N8_DATASETS, MODEL_NAME)
    preprocess = merged_encoder.val_preprocess
    print_memory("after merged model")

    # ======================================================================= #
    # Phase 2: Load fine-tuned encoders
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 2: Loading fine-tuned encoders")
    logger.info("=" * 70)

    finetuned_encoders = {}
    for ds in tasks:
        logger.info(f"  Loading fine-tuned encoder: {ds}")
        finetuned_encoders[ds] = load_finetuned_encoder(ds, MODEL_NAME)
    print_memory("after loading finetuned encoders")

    # ======================================================================= #
    # Phase 3: Load data for each task
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 3: Loading data for all tasks")
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
    # Phase 4: Extract features from merged and fine-tuned models
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 4: Extracting features from all models")
    logger.info("=" * 70)

    merged_features = {}
    finetuned_features = {}

    # Extract merged features (one model, all tasks)
    logger.info("Extracting merged model features...")
    for ds in tasks:
        logger.info(f"  Merged features for {ds}...")
        train_feats = extract_features(merged_encoder, task_data[ds]["train_imgs"], device)
        test_feats = extract_features(merged_encoder, task_data[ds]["test_imgs"], device)
        merged_features[ds] = {"train": train_feats, "test": test_feats}
    print_memory("after merged feature extraction")

    # Extract fine-tuned features (one model per task)
    logger.info("Extracting fine-tuned model features...")
    for ds in tasks:
        logger.info(f"  Fine-tuned features for {ds}...")
        ft_encoder = finetuned_encoders[ds]
        train_feats = extract_features(ft_encoder, task_data[ds]["train_imgs"], device)
        test_feats = extract_features(ft_encoder, task_data[ds]["test_imgs"], device)
        finetuned_features[ds] = {"train": train_feats, "test": test_feats}
    print_memory("after finetuned feature extraction")

    # Free fine-tuned encoders (we only need their features now)
    del finetuned_encoders
    torch.cuda.empty_cache()

    # ======================================================================= #
    # Phase 5: Compute alignment transforms and evaluate
    # ======================================================================= #
    logger.info("=" * 70)
    logger.info("PHASE 5: Computing alignment transforms and evaluating")
    logger.info("=" * 70)

    results = {}
    alignment_diagnostics = {}

    for ds in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Task: {ds}")
        logger.info(f"{'='*50}")

        m_train = merged_features[ds]["train"]
        m_test = merged_features[ds]["test"]
        ft_train = finetuned_features[ds]["train"]
        ft_test = finetuned_features[ds]["test"]
        train_labels = task_data[ds]["train_labels"]
        test_labels = task_data[ds]["test_labels"]

        results[ds] = {}
        alignment_diagnostics[ds] = {}

        # --------------------------------------------------------------- #
        # Method 1: Merged probe (baseline)
        # Train probe on merged features, evaluate on merged features
        # --------------------------------------------------------------- #
        logger.info(f"  [merged_probe] Training probe on merged features...")
        train_acc, test_acc = train_and_evaluate_probe(m_train, train_labels, m_test, test_labels)
        results[ds]["merged_probe"] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        logger.info(f"  [merged_probe] train={train_acc:.4f}, test={test_acc:.4f}")

        # --------------------------------------------------------------- #
        # Train the fine-tuned probe (used by alignment methods)
        # Train on fine-tuned features, will evaluate on aligned merged feats
        # --------------------------------------------------------------- #
        logger.info(f"  Training fine-tuned probe (for alignment evaluation)...")
        ft_probe = train_probe(ft_train, train_labels)
        ft_probe_train_acc = evaluate_probe(ft_probe, ft_train, train_labels)
        ft_probe_test_on_ft = evaluate_probe(ft_probe, ft_test, test_labels)
        logger.info(f"  FT probe on FT features: train={ft_probe_train_acc:.4f}, "
                    f"test={ft_probe_test_on_ft:.4f}")

        # Also evaluate FT probe directly on unaligned merged features (to show misalignment)
        ft_probe_on_merged = evaluate_probe(ft_probe, m_test, test_labels)
        logger.info(f"  FT probe on unaligned merged features: test={ft_probe_on_merged:.4f}")
        alignment_diagnostics[ds]["ft_probe_on_merged_feats"] = ft_probe_on_merged

        # --------------------------------------------------------------- #
        # Method 2: Procrustes alignment
        # Rotate merged features toward fine-tuned space, then use FT probe
        # --------------------------------------------------------------- #
        logger.info(f"  [procrustes] Computing Procrustes rotation...")
        R, m_mean, ft_mean, proc_residual = compute_procrustes_rotation(m_train, ft_train)
        alignment_diagnostics[ds]["procrustes_residual"] = proc_residual

        # Apply Procrustes to test features
        proc_train_aligned = apply_procrustes(m_train, R, m_mean, ft_mean)
        proc_test_aligned = apply_procrustes(m_test, R, m_mean, ft_mean)

        # Evaluate FT probe on Procrustes-aligned features
        proc_train_acc = evaluate_probe(ft_probe, proc_train_aligned, train_labels)
        proc_test_acc = evaluate_probe(ft_probe, proc_test_aligned, test_labels)
        results[ds]["procrustes_probe"] = {
            "train_acc": proc_train_acc,
            "test_acc": proc_test_acc,
        }
        logger.info(f"  [procrustes] train={proc_train_acc:.4f}, test={proc_test_acc:.4f}")

        # Compute feature-space diagnostics
        cosine_before = _mean_cosine_similarity(m_test, ft_test)
        cosine_after_proc = _mean_cosine_similarity(proc_test_aligned, ft_test)
        alignment_diagnostics[ds]["cosine_merged_vs_ft"] = cosine_before
        alignment_diagnostics[ds]["cosine_procrustes_vs_ft"] = cosine_after_proc
        logger.info(f"  Cosine sim (merged vs FT): {cosine_before:.4f}")
        logger.info(f"  Cosine sim (Procrustes vs FT): {cosine_after_proc:.4f}")

        # --------------------------------------------------------------- #
        # Method 3: Affine alignment
        # Linear transform merged features toward fine-tuned space
        # --------------------------------------------------------------- #
        logger.info(f"  [affine] Computing affine transform (lambda={args.reg_lambda})...")
        W, b, affine_residual = compute_affine_transform(m_train, ft_train, reg_lambda=args.reg_lambda)
        alignment_diagnostics[ds]["affine_residual"] = affine_residual

        affine_train_aligned = apply_affine(m_train, W, b)
        affine_test_aligned = apply_affine(m_test, W, b)

        affine_train_acc = evaluate_probe(ft_probe, affine_train_aligned, train_labels)
        affine_test_acc = evaluate_probe(ft_probe, affine_test_aligned, test_labels)
        results[ds]["affine_probe"] = {
            "train_acc": affine_train_acc,
            "test_acc": affine_test_acc,
        }
        logger.info(f"  [affine] train={affine_train_acc:.4f}, test={affine_test_acc:.4f}")

        cosine_after_affine = _mean_cosine_similarity(affine_test_aligned, ft_test)
        alignment_diagnostics[ds]["cosine_affine_vs_ft"] = cosine_after_affine
        logger.info(f"  Cosine sim (Affine vs FT): {cosine_after_affine:.4f}")

        # --------------------------------------------------------------- #
        # Method 4: Fine-tuned probe (upper bound)
        # Probe trained and evaluated on fine-tuned features
        # --------------------------------------------------------------- #
        results[ds]["finetuned_probe"] = {
            "train_acc": ft_probe_train_acc,
            "test_acc": ft_probe_test_on_ft,
        }
        logger.info(f"  [finetuned_probe] train={ft_probe_train_acc:.4f}, "
                    f"test={ft_probe_test_on_ft:.4f}")

        # Compute deltas
        merged_base = results[ds]["merged_probe"]["test_acc"]
        ft_upper = results[ds]["finetuned_probe"]["test_acc"]
        proc_delta = results[ds]["procrustes_probe"]["test_acc"] - merged_base
        affine_delta = results[ds]["affine_probe"]["test_acc"] - merged_base
        gap = ft_upper - merged_base

        results[ds]["summary"] = {
            "merged_to_ft_gap": gap,
            "procrustes_recovery": proc_delta,
            "affine_recovery": affine_delta,
            "procrustes_pct_recovered": (proc_delta / gap * 100) if gap > 0 else 0.0,
            "affine_pct_recovered": (affine_delta / gap * 100) if gap > 0 else 0.0,
        }

        logger.info(f"\n  Summary for {ds}:")
        logger.info(f"    Gap (FT - Merged): {gap:.4f}")
        logger.info(f"    Procrustes recovery: {proc_delta:+.4f} ({results[ds]['summary']['procrustes_pct_recovered']:.1f}%)")
        logger.info(f"    Affine recovery: {affine_delta:+.4f} ({results[ds]['summary']['affine_pct_recovered']:.1f}%)")

        # Save intermediate
        _save_intermediate(results, OUTPUT_DIR, f"after_{ds}")

    # ======================================================================= #
    # Phase 6: Summary
    # ======================================================================= #
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    header = (f"{'Task':<12} {'Merged':>8} {'Procrust':>8} {'Affine':>8} "
              f"{'FT(UB)':>8} {'Gap':>8} {'Proc%':>7} {'Aff%':>7}")
    logger.info(header)
    logger.info("-" * len(header))

    for ds in tasks:
        r = results[ds]
        s = r["summary"]
        logger.info(
            f"{ds:<12} "
            f"{r['merged_probe']['test_acc']:>8.4f} "
            f"{r['procrustes_probe']['test_acc']:>8.4f} "
            f"{r['affine_probe']['test_acc']:>8.4f} "
            f"{r['finetuned_probe']['test_acc']:>8.4f} "
            f"{s['merged_to_ft_gap']:>8.4f} "
            f"{s['procrustes_pct_recovered']:>6.1f}% "
            f"{s['affine_pct_recovered']:>6.1f}%"
        )

    # Averages
    avg_merged = np.mean([results[ds]["merged_probe"]["test_acc"] for ds in tasks])
    avg_proc = np.mean([results[ds]["procrustes_probe"]["test_acc"] for ds in tasks])
    avg_affine = np.mean([results[ds]["affine_probe"]["test_acc"] for ds in tasks])
    avg_ft = np.mean([results[ds]["finetuned_probe"]["test_acc"] for ds in tasks])
    avg_proc_pct = np.mean([results[ds]["summary"]["procrustes_pct_recovered"] for ds in tasks])
    avg_affine_pct = np.mean([results[ds]["summary"]["affine_pct_recovered"] for ds in tasks])

    logger.info("-" * len(header))
    logger.info(
        f"{'AVERAGE':<12} "
        f"{avg_merged:>8.4f} "
        f"{avg_proc:>8.4f} "
        f"{avg_affine:>8.4f} "
        f"{avg_ft:>8.4f} "
        f"{(avg_ft - avg_merged):>8.4f} "
        f"{avg_proc_pct:>6.1f}% "
        f"{avg_affine_pct:>6.1f}%"
    )

    # Alignment diagnostics summary
    logger.info("\nAlignment Diagnostics:")
    logger.info(f"{'Task':<12} {'cos(M,FT)':>10} {'cos(P,FT)':>10} {'cos(A,FT)':>10} "
                f"{'P_resid':>10} {'A_resid':>10}")
    logger.info("-" * 65)
    for ds in tasks:
        d = alignment_diagnostics[ds]
        logger.info(
            f"{ds:<12} "
            f"{d['cosine_merged_vs_ft']:>10.4f} "
            f"{d['cosine_procrustes_vs_ft']:>10.4f} "
            f"{d['cosine_affine_vs_ft']:>10.4f} "
            f"{d['procrustes_residual']:>10.2f} "
            f"{d['affine_residual']:>10.2f}"
        )

    # ======================================================================= #
    # Phase 7: Plots and final save
    # ======================================================================= #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 7: Generating plots and saving results")
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
            "affine_reg_lambda": args.reg_lambda,
            "methods": {
                "merged_probe": "Probe trained on merged features, evaluated on merged features",
                "procrustes_probe": "Orthogonal rotation (Procrustes) to align merged->FT, then FT probe",
                "affine_probe": "Affine transform (W,b) to align merged->FT, then FT probe",
                "finetuned_probe": "Probe trained on fine-tuned features (upper bound)",
            },
            "evaluation": "LogisticRegression (C=0.316, L2-normalized features)",
            "quick_mode": args.quick,
        },
        "per_task_results": results,
        "alignment_diagnostics": alignment_diagnostics,
        "averages": {
            "merged_probe": avg_merged,
            "procrustes_probe": avg_proc,
            "affine_probe": avg_affine,
            "finetuned_probe": avg_ft,
            "procrustes_pct_recovered": avg_proc_pct,
            "affine_pct_recovered": avg_affine_pct,
        },
    }

    json_path = OUTPUT_DIR / "feature_alignment_results.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(final_output), f, indent=2)
    logger.info(f"Results saved to: {json_path}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info("Feature alignment analysis complete.")


# --------------------------------------------------------------------------- #
# Utility: mean cosine similarity between two feature matrices
# --------------------------------------------------------------------------- #

def _mean_cosine_similarity(A, B):
    """Compute mean per-sample cosine similarity between two feature matrices."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    cosines = np.sum(A_norm * B_norm, axis=1)
    return float(np.mean(cosines))


if __name__ == "__main__":
    main()
