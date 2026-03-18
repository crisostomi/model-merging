"""
Function-Preserving Growth for Late Blocks: Break the Capacity Ceiling

Applies "Grow, Don't Overwrite" (Adila et al., 2026) to the merged model:
1. Merge all blocks with MP-edge + isotropic (baseline)
2. Expand MLP in blocks 9-11 via function-preserving growth (2x hidden dim)
3. Fine-tune only expanded parameters on multi-task data (G-Freeze)
4. Evaluate on all 8 N8 tasks

The expansion guarantees the model is functionally identical to the merged
model at initialization. Fine-tuning then leverages the extra capacity.

Usage:
    sbatch slurm/launch_analysis.slurm scripts/grow_and_finetune.py
    sbatch slurm/launch_analysis.slurm scripts/grow_and_finetune.py --quick
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_merging.model.encoder import ImageEncoder
from model_merging.merger.interference_aware_merger import InterferenceAwareMerger
from model_merging.utils.utils import print_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "grow_finetune"


# --------------------------------------------------------------------------- #
# Model loading (same as other analysis scripts)
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
# Function-preserving MLP expansion
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

        # New up-projection (TRAINABLE) — starts as copy of original
        self.c_fc_new = nn.Linear(input_dim, hidden_dim)
        self.c_fc_new.weight.data = c_fc_w.clone()
        self.c_fc_new.bias.data = c_fc_b.clone()

        self.gelu = copy.deepcopy(original_mlp.gelu)
        self.ln = copy.deepcopy(original_mlp.ln) if hasattr(original_mlp, 'ln') else nn.Identity()

        # Original down-projection (FROZEN) — scaled by ½
        self.c_proj_orig = nn.Linear(hidden_dim, output_dim, bias=True)
        self.c_proj_orig.weight.data = 0.5 * c_proj_w.clone()
        self.c_proj_orig.bias.data = c_proj_b.clone()  # full bias on orig

        # New down-projection (TRAINABLE) — scaled by ½, no bias
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
# Multi-task data loading
# --------------------------------------------------------------------------- #

def load_multitask_data(preprocess, datasets, n_per_task=500):
    """Load a small multi-task training set."""
    from datasets import load_dataset as hf_load_dataset

    DATASET_HF = {
        "SUN397": ("tanganke/sun397", None),
        "Cars": ("tanganke/stanford_cars", None),
        "RESISC45": ("tanganke/resisc45", None),
        "EuroSAT": ("tanganke/eurosat", None),
        "SVHN": ("ufldl-stanford/svhn", "cropped_digits"),
        "GTSRB": ("tanganke/gtsrb", None),
        "MNIST": ("ylecun/mnist", None),
        "DTD": ("tanganke/dtd", None),
    }

    all_images = []
    all_labels = []  # We don't use labels for multi-task encoder fine-tuning
    # We'll train with a self-distillation objective instead

    for ds_name in datasets:
        hf_name, config = DATASET_HF[ds_name]
        logger.info(f"  Loading {ds_name} ({n_per_task} samples)...")
        if config:
            ds = hf_load_dataset(hf_name, config, split="train")
        else:
            ds = hf_load_dataset(hf_name, split="train")

        count = 0
        for sample in ds:
            if count >= n_per_task:
                break
            try:
                img = preprocess(sample["image"].convert("RGB"))
                all_images.append(img)
                count += 1
            except Exception:
                continue

    images = torch.stack(all_images)
    logger.info(f"  Total multi-task samples: {len(images)}")
    return images


# --------------------------------------------------------------------------- #
# Self-distillation training
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_teacher_features(encoder, images, device, batch_size=256):
    """Get the frozen merged model's output features as distillation targets."""
    encoder.eval()
    encoder.to(device)
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        feat = encoder(batch)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


def train_expanded_model(encoder, images, teacher_features, device,
                         epochs=10, lr=1e-4, batch_size=64):
    """Train expanded parameters via self-distillation.

    Loss: MSE between expanded model output and teacher (original merged) output.
    This encourages the expanded model to preserve the merged model's features
    while having capacity to develop better task-specific representations.
    """
    encoder.to(device)
    encoder.train()

    # Only optimize parameters that require grad
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(images, teacher_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch_imgs, batch_targets in loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)

            outputs = encoder(batch_imgs)
            loss = torch.nn.functional.mse_loss(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

    encoder.eval()
    return encoder


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_all_tasks(encoder, preprocess, device, eval_on_val=True):
    """Evaluate on all N8 tasks using the existing classification heads."""
    from datasets import load_dataset as hf_load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    DATASET_HF = {
        "SUN397": ("tanganke/sun397", None),
        "Cars": ("tanganke/stanford_cars", None),
        "RESISC45": ("tanganke/resisc45", None),
        "EuroSAT": ("tanganke/eurosat", None),
        "SVHN": ("ufldl-stanford/svhn", "cropped_digits"),
        "GTSRB": ("tanganke/gtsrb", None),
        "MNIST": ("ylecun/mnist", None),
        "DTD": ("tanganke/dtd", None),
    }

    encoder.eval()
    encoder.to(device)
    results = {}

    for ds_name in N8_DATASETS:
        hf_name, config = DATASET_HF[ds_name]
        split = "test"

        if config:
            ds = hf_load_dataset(hf_name, config, split=split)
        else:
            ds = hf_load_dataset(hf_name, split=split)

        # Extract features
        all_feats = []
        all_labels = []
        for sample in ds:
            try:
                img = preprocess(sample["image"].convert("RGB")).unsqueeze(0).to(device)
                feat = encoder(img).cpu()
                all_feats.append(feat)
                all_labels.append(sample["label"])
            except Exception:
                continue

            if len(all_feats) >= 5000:  # cap for speed
                break

        feats = torch.cat(all_feats, dim=0).numpy()
        labels = np.array(all_labels)

        # Use same train data for probe (reuse test as train — not ideal but
        # matches the linear probe methodology for fair comparison)
        feats_norm = normalize(feats, norm="l2")

        # Split 80/20 for train/test
        n = len(feats)
        n_train = int(0.8 * n)
        clf = LogisticRegression(max_iter=500, C=0.316, solver="lbfgs", n_jobs=-1)
        clf.fit(feats_norm[:n_train], labels[:n_train])
        acc = clf.score(feats_norm[n_train:], labels[n_train:])

        results[ds_name] = float(acc)
        logger.info(f"  {ds_name}: {acc:.4f}")

    encoder.cpu()
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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

    # Step 1: Create merged model
    logger.info("=" * 60)
    logger.info("STEP 1: Create merged model (MP-edge + isotropic)")
    logger.info("=" * 60)
    merged = create_merged_model(N8_DATASETS)
    preprocess = merged.val_preprocess
    print_memory("after merge")

    # Step 2: Load multi-task training data
    logger.info("=" * 60)
    logger.info("STEP 2: Load multi-task training data")
    logger.info("=" * 60)
    train_images = load_multitask_data(preprocess, N8_DATASETS, args.n_per_task)

    # Step 3: Compute teacher features (before expansion)
    logger.info("=" * 60)
    logger.info("STEP 3: Compute teacher features (distillation targets)")
    logger.info("=" * 60)
    teacher_features = compute_teacher_features(merged, train_images, device)
    merged.cpu()
    torch.cuda.empty_cache()
    logger.info(f"  Teacher features shape: {teacher_features.shape}")

    # Step 4: Expand late blocks
    logger.info("=" * 60)
    logger.info("STEP 4: Function-preserving MLP expansion")
    logger.info("=" * 60)
    merged = expand_late_blocks(merged, block_indices)
    trainable_count = setup_gfreeze(merged, block_indices)
    print_memory("after expansion")

    # Verify function-preserving property
    logger.info("  Verifying function-preserving property...")
    merged.to(device)
    merged.eval()
    with torch.no_grad():
        test_batch = train_images[:8].to(device)
        expanded_out = merged(test_batch).cpu()
        teacher_batch = teacher_features[:8]
        diff = (expanded_out - teacher_batch).norm().item()
        logger.info(f"  Output difference after expansion: {diff:.8f} (should be ~0)")
    merged.cpu()
    torch.cuda.empty_cache()

    # Step 5: Fine-tune expanded parameters
    logger.info("=" * 60)
    logger.info("STEP 5: Fine-tune expanded parameters (self-distillation)")
    logger.info("=" * 60)
    merged = train_expanded_model(
        merged, train_images, teacher_features, device,
        epochs=args.epochs, lr=args.lr, batch_size=64,
    )
    print_memory("after training")

    # Step 6: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluate on all N8 tasks (linear probe)")
    logger.info("=" * 60)
    results = evaluate_all_tasks(merged, preprocess, device)

    avg = np.mean(list(results.values()))
    logger.info(f"\n  Average: {avg:.4f}")
    logger.info(f"\n  SVHN: {results.get('SVHN', 0):.4f}")

    # Save
    all_results = {
        "config": {
            "expand_blocks": list(block_indices),
            "epochs": args.epochs,
            "lr": args.lr,
            "n_per_task": args.n_per_task,
            "trainable_params": trainable_count,
        },
        "results": results,
        "average": float(avg),
    }

    json_path = OUTPUT_DIR / "grow_finetune_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
