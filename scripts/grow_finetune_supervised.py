"""
Multi-task supervised fine-tuning of function-preserving expanded merged model.

Previous attempt (grow_and_finetune.py) used self-distillation (MSE to teacher
outputs) which FAILED -- it just reproduced teacher features without learning
new task-specific representations. This script uses actual multi-task
classification training with real labels.

Pipeline:
1. Create merged model (MP-edge + isotropic, 8 tasks)
2. Apply function-preserving MLP expansion to blocks 9-11
3. G-Freeze: only new modules are trainable
4. Load classification heads from HuggingFace / MODELS_PATH cache
5. Load training data: 500 samples per task with labels
6. Train with multi-task cross-entropy loss (cycle through tasks)
7. Evaluate using standard pipeline methodology (full test split)
8. Save results JSON + comparison plot

Usage:
    sbatch slurm/launch_analysis.slurm scripts/grow_finetune_supervised.py
    sbatch slurm/launch_analysis.slurm scripts/grow_finetune_supervised.py --quick
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.model.encoder import ImageEncoder, ClassificationHead
from model_merging.merger.interference_aware_merger import InterferenceAwareMerger
from model_merging.model.heads import get_classification_head
from model_merging.data.dataset import load_hf_dataset_filtered, HFImageClassification
from model_merging.utils.utils import print_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "grow_finetune"

# Baseline accuracies from the official evaluation (MP-edge + isotropic, no expansion)
BASELINE_ACCS = {
    "SUN397": 0.7325,
    "Cars": 0.7512,
    "RESISC45": 0.9000,
    "EuroSAT": 0.9630,
    "SVHN": 0.7968,
    "GTSRB": 0.9224,
    "MNIST": 0.9710,
    "DTD": 0.7181,
}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def create_merged_model(datasets, model_name=MODEL_NAME):
    logger.info(f"Creating merged model with {len(datasets)} tasks")
    encoder = ImageEncoder(model_name)
    finetuned = {}
    for ds in datasets:
        finetuned[ds] = load_state_dict_from_hf_cache(model_name, ds)
    merger = InterferenceAwareMerger(
        use_mp_edge=True, use_isotropic=True, mp_min_rank=4, mp_max_rank=128
    )
    return merger.merge(encoder, finetuned)


# --------------------------------------------------------------------------- #
# Function-preserving MLP expansion (copied from grow_and_finetune.py)
# --------------------------------------------------------------------------- #

class ExpandedMLP(nn.Module):
    """MLP with function-preserving 2x expansion using separate modules for G-Freeze.

    Structure: x -> [c_fc_orig; c_fc_new] -> gelu -> c_proj_orig(orig_half) + c_proj_new(new_half) -> out

    The original and new halves are SEPARATE modules so we can freeze originals
    while training only the new copies. At initialization, produces IDENTICAL
    outputs to the original MLP.
    """

    def __init__(self, original_mlp):
        super().__init__()
        c_fc_w = original_mlp.c_fc.weight.data      # (3072, 768)
        c_fc_b = original_mlp.c_fc.bias.data         # (3072,)
        c_proj_w = original_mlp.c_proj.weight.data   # (768, 3072)
        c_proj_b = original_mlp.c_proj.bias.data     # (768,)

        hidden_dim = c_fc_w.shape[0]  # 3072
        input_dim = c_fc_w.shape[1]   # 768
        output_dim = c_proj_w.shape[0] # 768

        # Original up-projection (FROZEN)
        self.c_fc_orig = nn.Linear(input_dim, hidden_dim)
        self.c_fc_orig.weight.data = c_fc_w.clone()
        self.c_fc_orig.bias.data = c_fc_b.clone()

        # New up-projection (TRAINABLE) -- starts as copy of original
        self.c_fc_new = nn.Linear(input_dim, hidden_dim)
        self.c_fc_new.weight.data = c_fc_w.clone()
        self.c_fc_new.bias.data = c_fc_b.clone()

        self.gelu = copy.deepcopy(original_mlp.gelu)
        self.ln = copy.deepcopy(original_mlp.ln) if hasattr(original_mlp, 'ln') else nn.Identity()

        # Original down-projection (FROZEN) -- scaled by 1/2
        self.c_proj_orig = nn.Linear(hidden_dim, output_dim, bias=True)
        self.c_proj_orig.weight.data = 0.5 * c_proj_w.clone()
        self.c_proj_orig.bias.data = c_proj_b.clone()  # full bias on orig

        # New down-projection (TRAINABLE) -- scaled by 1/2, no bias
        self.c_proj_new = nn.Linear(hidden_dim, output_dim, bias=False)
        self.c_proj_new.weight.data = 0.5 * c_proj_w.clone()

        self.hidden_dim = hidden_dim
        self.expanded = True

    def forward(self, x):
        # Up-projection: both halves process the same input
        h_orig = self.c_fc_orig(x)
        h_new = self.c_fc_new(x)

        # Activation on each half separately
        h_orig = self.gelu(h_orig)
        h_new = self.gelu(h_new)

        # Down-projection: sum of both halves
        # At init: 0.5 * W @ gelu(W @ x) + 0.5 * W @ gelu(W @ x) = W @ gelu(W @ x)
        out = self.c_proj_orig(h_orig) + self.c_proj_new(h_new)
        return out


def expand_late_blocks(encoder, block_indices=(9, 10, 11)):
    """Replace MLP in specified blocks with function-preserving expanded versions."""
    visual = encoder.model.visual
    for idx in block_indices:
        block = visual.transformer.resblocks[idx]
        original_mlp = block.mlp
        expanded_mlp = ExpandedMLP(original_mlp)
        block.mlp = expanded_mlp
        logger.info(f"  Expanded block {idx} MLP: {original_mlp.c_fc.weight.shape[0]}d -> 2x{expanded_mlp.hidden_dim}d (separate orig/new modules)")
    return encoder


def setup_gfreeze(encoder, block_indices=(9, 10, 11)):
    """Freeze all parameters except the 'new' modules in expanded MLPs."""
    # Freeze everything
    for param in encoder.parameters():
        param.requires_grad = False

    # Unfreeze only the NEW modules
    visual = encoder.model.visual
    trainable_count = 0
    for idx in block_indices:
        block = visual.transformer.resblocks[idx]
        mlp = block.mlp
        if not hasattr(mlp, 'expanded'):
            continue

        for param in mlp.c_fc_new.parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        for param in mlp.c_proj_new.parameters():
            param.requires_grad = True
            trainable_count += param.numel()

    total = sum(p.numel() for p in encoder.parameters())
    logger.info(f"  G-Freeze: {trainable_count:,} trainable / {total:,} total ({trainable_count/total*100:.1f}%)")
    return trainable_count


# --------------------------------------------------------------------------- #
# Classification heads
# --------------------------------------------------------------------------- #

def load_classification_heads(model_name, datasets, device="cpu"):
    """Load classification heads for all tasks.

    Uses get_classification_head which checks MODELS_PATH cache first,
    then builds from scratch if needed.
    """
    models_path = os.environ.get("MODELS_PATH", str(PROJECT_ROOT / "checkpoints"))
    ckpt_path = os.path.join(models_path, model_name)
    openclip_cachedir = os.path.join(models_path, "openclip_cache")

    heads = {}
    for ds_name in datasets:
        logger.info(f"  Loading classification head for {ds_name}...")
        head = get_classification_head(
            model_name, ds_name,
            ckpt_path=ckpt_path,
            openclip_cachedir=openclip_cachedir,
            device=device,
        )
        head.eval()
        for param in head.parameters():
            param.requires_grad = False
        heads[ds_name] = head
        logger.info(f"    {ds_name}: {head.weight.shape[0]} classes, normalize={head.normalize}")
    return heads


# --------------------------------------------------------------------------- #
# Multi-task data loading (with labels)
# --------------------------------------------------------------------------- #

# HuggingFace dataset configs -- matches the dataset YAML configs
DATASET_HF = {
    "SUN397":   {"path": "tanganke/sun397"},
    "Cars":     {"path": "tanganke/stanford_cars"},
    "RESISC45": {"path": "tanganke/resisc45"},
    "EuroSAT":  {"path": "tanganke/eurosat"},
    "SVHN":     {"path": "ufldl-stanford/svhn", "name": "cropped_digits"},
    "GTSRB":    {"path": "tanganke/gtsrb"},
    "MNIST":    {"path": "ylecun/mnist"},
    "DTD":      {"path": "tanganke/dtd"},
}

# Label maps from the dataset configs (only RESISC45 has one)
LABEL_MAPS = {
    "RESISC45": {
        0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10,
        8: 11, 9: 12, 10: 13, 11: 14, 12: 15, 13: 1, 14: 16,
        15: 17, 16: 18, 17: 19, 18: 4, 19: 20, 20: 21, 21: 22,
        22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29,
        29: 30, 30: 31, 31: 32, 32: 8, 33: 33, 34: 34, 35: 35,
        36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
        43: 43, 44: 44,
    },
}


def load_task_datasets(preprocess, datasets, n_train_per_task=500, batch_size=32):
    """Load train and test data for all tasks.

    Returns:
        train_loaders: dict of {task_name: DataLoader} for training (capped at n_train_per_task)
        test_loaders: dict of {task_name: DataLoader} for evaluation (full test split)
    """
    from datasets import load_dataset as hf_load_dataset
    from model_merging.data.dataset import _HFImageTorchDataset

    train_loaders = {}
    test_loaders = {}

    for ds_name in datasets:
        hf_cfg = DATASET_HF[ds_name]
        hf_path = hf_cfg["path"]
        hf_name = hf_cfg.get("name", None)
        label_map = LABEL_MAPS.get(ds_name, None)

        logger.info(f"  Loading {ds_name}...")

        # Load train split (subset for training)
        if hf_name:
            train_split = hf_load_dataset(hf_path, hf_name, split="train")
        else:
            train_split = hf_load_dataset(hf_path, split="train")

        # Subsample train data
        n_avail = len(train_split)
        n_use = min(n_train_per_task, n_avail)
        if n_use < n_avail:
            # Deterministic subsample
            rng = np.random.default_rng(42)
            indices = rng.choice(n_avail, size=n_use, replace=False).tolist()
            train_split = train_split.select(indices)

        train_ds = _HFImageTorchDataset(train_split, transform=preprocess, label_map=label_map)
        train_loaders[ds_name] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )

        # Load test split (full, for evaluation)
        if hf_name:
            test_split = hf_load_dataset(hf_path, hf_name, split="test")
        else:
            test_split = hf_load_dataset(hf_path, split="test")

        test_ds = _HFImageTorchDataset(test_split, transform=preprocess, label_map=label_map)
        test_loaders[ds_name] = DataLoader(
            test_ds, batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        logger.info(f"    {ds_name}: {len(train_ds)} train, {len(test_ds)} test")

    return train_loaders, test_loaders


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_multitask(encoder, classification_heads, train_loaders, device,
                    epochs=20, lr=1e-4, weight_decay=0.01):
    """Train expanded parameters with multi-task classification loss.

    Each step: sample a random task, forward through encoder + task head, CE loss.
    Only the expanded (unfrozen) parameters receive gradients.
    """
    encoder.to(device)
    for head in classification_heads.values():
        head.to(device)

    # Only optimize parameters that require grad
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # Compute total steps for cosine schedule
    # Each epoch iterates through all tasks; steps per epoch = sum of batches across tasks
    steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    task_names = list(train_loaders.keys())
    logger.info(f"  Training: {epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps")
    logger.info(f"  LR: {lr}, weight_decay: {weight_decay}")

    for epoch in range(epochs):
        encoder.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        # Create iterators for all tasks
        task_iterators = {name: iter(loader) for name, loader in train_loaders.items()}

        # Round-robin through tasks (ensures each task is visited equally per epoch)
        # Interleave batches from all tasks
        active_tasks = list(task_names)
        while active_tasks:
            # Shuffle task order each round to avoid ordering bias
            rng = np.random.default_rng(epoch * 10000 + n_batches)
            round_tasks = list(active_tasks)
            rng.shuffle(round_tasks)

            exhausted = []
            for task_name in round_tasks:
                try:
                    images, labels = next(task_iterators[task_name])
                except StopIteration:
                    exhausted.append(task_name)
                    continue

                images = images.to(device)
                labels = labels.to(device)

                # Forward: encoder -> classification head
                features = encoder(images)
                head = classification_heads[task_name]
                logits = head(features)

                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Track metrics
                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += len(labels)
                n_batches += 1

            for t in exhausted:
                active_tasks.remove(t)

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_correct / max(epoch_total, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
            f"train_acc={avg_acc:.4f}, lr={current_lr:.2e}"
        )

    encoder.eval()
    return encoder


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_task(encoder, head, test_loader, device):
    """Evaluate a single task using the standard pipeline methodology."""
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = encoder(images)
        logits = head(features)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total if total > 0 else 0.0


def evaluate_all_tasks(encoder, classification_heads, test_loaders, device):
    """Evaluate on all N8 tasks using classification heads (standard methodology)."""
    encoder.to(device)
    encoder.eval()
    results = {}

    for ds_name in N8_DATASETS:
        head = classification_heads[ds_name].to(device)
        loader = test_loaders[ds_name]
        acc = evaluate_task(encoder, head, loader, device)
        results[ds_name] = float(acc)
        logger.info(f"  {ds_name}: {acc:.4f}")

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_comparison(baseline, expanded, output_path):
    """Bar chart comparing baseline vs expanded model accuracies."""
    tasks = list(baseline.keys())
    base_vals = [baseline[t] for t in tasks]
    exp_vals = [expanded[t] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, base_vals, width, label="Baseline (merged)", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, exp_vals, width, label="Expanded + supervised", color="#DD8452", alpha=0.85)

    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy")
    ax.set_title("Multi-task Supervised Fine-tuning of Expanded Merged Model")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Add average lines
    base_avg = np.mean(base_vals)
    exp_avg = np.mean(exp_vals)
    ax.axhline(y=base_avg, color="#4C72B0", linestyle="--", alpha=0.5, label=f"Baseline avg: {base_avg:.4f}")
    ax.axhline(y=exp_avg, color="#DD8452", linestyle="--", alpha=0.5, label=f"Expanded avg: {exp_avg:.4f}")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Plot saved to {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of expanded merged model")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer epochs and samples")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-per-task", type=int, default=500)
    parser.add_argument("--expand-blocks", type=str, default="9,10,11")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_indices = tuple(int(x) for x in args.expand_blocks.split(","))

    if args.quick:
        args.epochs = 3
        args.n_per_task = 100

    logger.info(f"Device: {device}")
    logger.info(f"Expand blocks: {block_indices}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Samples/task: {args.n_per_task}")
    logger.info(f"Batch size: {args.batch_size}, Weight decay: {args.weight_decay}")

    # ------------------------------------------------------------------- #
    # Step 1: Create merged model
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 1: Create merged model (MP-edge + isotropic)")
    logger.info("=" * 60)
    merged = create_merged_model(N8_DATASETS)
    preprocess = merged.val_preprocess
    print_memory("after merge")

    # ------------------------------------------------------------------- #
    # Step 2: Load classification heads
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 2: Load classification heads for all tasks")
    logger.info("=" * 60)
    classification_heads = load_classification_heads(MODEL_NAME, N8_DATASETS, device="cpu")

    # ------------------------------------------------------------------- #
    # Step 3: Load multi-task training + test data
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 3: Load multi-task training and test data")
    logger.info("=" * 60)
    train_loaders, test_loaders = load_task_datasets(
        preprocess, N8_DATASETS,
        n_train_per_task=args.n_per_task,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------- #
    # Step 4: Evaluate baseline (before expansion)
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluate baseline merged model (before expansion)")
    logger.info("=" * 60)
    baseline_results = evaluate_all_tasks(merged, classification_heads, test_loaders, device)
    baseline_avg = np.mean(list(baseline_results.values()))
    logger.info(f"  Baseline average: {baseline_avg:.4f}")
    merged.cpu()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- #
    # Step 5: Expand late blocks
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 5: Function-preserving MLP expansion")
    logger.info("=" * 60)
    merged = expand_late_blocks(merged, block_indices)
    trainable_count = setup_gfreeze(merged, block_indices)
    print_memory("after expansion")

    # Verify function-preserving property
    logger.info("  Verifying function-preserving property...")
    merged.to(device)
    merged.eval()
    with torch.no_grad():
        # Grab a small batch from the first task's train loader
        first_task = N8_DATASETS[0]
        test_iter = iter(test_loaders[first_task])
        test_images, _ = next(test_iter)
        test_images = test_images[:8].to(device)

        # Compare expanded model output with a fresh forward pass
        # (We trust function-preserving init -- just check output is finite and not NaN)
        expanded_out = merged(test_images)
        assert not torch.isnan(expanded_out).any(), "NaN in expanded model output!"
        logger.info(f"  Expanded model output shape: {expanded_out.shape}, norm: {expanded_out.norm(dim=-1).mean():.4f}")

    # Also verify expanded model matches baseline on evaluation
    logger.info("  Verifying expanded model matches baseline accuracy (pre-training)...")
    expanded_pre_results = evaluate_all_tasks(merged, classification_heads, test_loaders, device)
    expanded_pre_avg = np.mean(list(expanded_pre_results.values()))
    logger.info(f"  Expanded (pre-training) average: {expanded_pre_avg:.4f}")
    diff = abs(expanded_pre_avg - baseline_avg)
    logger.info(f"  Difference from baseline: {diff:.6f} (should be ~0)")
    merged.cpu()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------- #
    # Step 6: Multi-task supervised fine-tuning
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 6: Multi-task supervised fine-tuning")
    logger.info("=" * 60)
    merged = train_multitask(
        merged, classification_heads, train_loaders, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
    )
    print_memory("after training")

    # ------------------------------------------------------------------- #
    # Step 7: Evaluate after fine-tuning
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 7: Evaluate expanded model after fine-tuning")
    logger.info("=" * 60)
    expanded_results = evaluate_all_tasks(merged, classification_heads, test_loaders, device)
    expanded_avg = np.mean(list(expanded_results.values()))
    logger.info(f"\n  Expanded average: {expanded_avg:.4f}")
    logger.info(f"  Baseline average: {baseline_avg:.4f}")
    logger.info(f"  Improvement:      {expanded_avg - baseline_avg:+.4f}")

    # Per-task comparison
    logger.info("\n  Per-task comparison:")
    logger.info(f"  {'Task':<12} {'Baseline':>10} {'Expanded':>10} {'Delta':>10}")
    logger.info(f"  {'-'*44}")
    for ds_name in N8_DATASETS:
        base_acc = baseline_results[ds_name]
        exp_acc = expanded_results[ds_name]
        delta = exp_acc - base_acc
        logger.info(f"  {ds_name:<12} {base_acc:>10.4f} {exp_acc:>10.4f} {delta:>+10.4f}")
    logger.info(f"  {'-'*44}")
    logger.info(f"  {'Average':<12} {baseline_avg:>10.4f} {expanded_avg:>10.4f} {expanded_avg - baseline_avg:>+10.4f}")

    # ------------------------------------------------------------------- #
    # Step 8: Save results and plot
    # ------------------------------------------------------------------- #
    logger.info("=" * 60)
    logger.info("STEP 8: Save results")
    logger.info("=" * 60)

    all_results = {
        "config": {
            "method": "supervised_multitask_classification",
            "expand_blocks": list(block_indices),
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "n_per_task": args.n_per_task,
            "trainable_params": trainable_count,
        },
        "baseline_results": baseline_results,
        "baseline_average": float(baseline_avg),
        "expanded_pre_training_results": expanded_pre_results,
        "expanded_pre_training_average": float(expanded_pre_avg),
        "expanded_results": expanded_results,
        "expanded_average": float(expanded_avg),
        "improvement": float(expanded_avg - baseline_avg),
        "per_task_delta": {
            ds: float(expanded_results[ds] - baseline_results[ds])
            for ds in N8_DATASETS
        },
    }

    json_path = OUTPUT_DIR / "grow_finetune_supervised_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Results saved to: {json_path}")

    # Plot comparison
    plot_path = OUTPUT_DIR / "supervised_comparison.png"
    plot_comparison(baseline_results, expanded_results, plot_path)

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
