"""
Split-Depth Inference & Exclusive Block-11 Ownership

Tests two zero-training, zero-param proposals for recovering SVHN accuracy:

Proposal A — Split-depth inference:
  Run all-8 merged model for blocks [0..K], then SVHN-FT blocks [K+1..11].
  Sweep K in {-1 (all FT), 7, 8, 9, 10, 11 (all merged)}.
  Hypothesis: merged blocks 0..K ≈ pretrained blocks 0..K ≈ SVHN-FT input at K+1,
  so SVHN-FT's late blocks should work correctly on the merged model's early output.

Proposal B — Exclusive block-11 ownership:
  Use 7-task merged model (excl. SVHN) for blocks 0..10; SVHN-FT for block 11.
  Block 11 is not contested by SVHN in the 7-task merge → SVHN gets its own slot.
  Equivalent to Proposal A K=10 but with 7-task merged model.

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_split_depth.py
    uv run python scripts/analyze_split_depth.py --quick
"""

import argparse
import json
import logging
import sys
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
N8 = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
N7_NO_SVHN = [d for d in N8 if d != "SVHN"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "split_depth"
N_HEADS = 12
HEAD_DIM = 64


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_sd(model_name, dataset_name="base"):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=f"crisostomi/{model_name}-{dataset_name}",
        filename="pytorch_model.bin",
    )
    return torch.load(path, map_location="cpu")


def make_encoder(model_name=MODEL_NAME):
    return ImageEncoder(model_name)


def make_ft_encoder(dataset_name, model_name=MODEL_NAME):
    enc = make_encoder(model_name)
    enc.load_state_dict(load_sd(model_name, dataset_name), strict=False)
    return enc


def make_merged(datasets, model_name=MODEL_NAME):
    log.info(f"  Merging {datasets}")
    enc = make_encoder(model_name)
    fts = {d: load_sd(model_name, d) for d in datasets}
    merger = InterferenceAwareMerger(use_mp_edge=True, use_isotropic=True, mp_min_rank=4, mp_max_rank=128)
    return merger.merge(enc, fts)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_svhn(preprocess, n_train=5000, n_test=2000):
    from datasets import load_dataset as hf_load
    log.info(f"Loading SVHN (train={n_train}, test={n_test})")
    train_ds = hf_load("ufldl-stanford/svhn", "cropped_digits", split="train")
    test_ds  = hf_load("ufldl-stanford/svhn", "cropped_digits", split="test")

    def process(ds, n):
        imgs, labels = [], []
        for i, s in enumerate(ds):
            if i >= n:
                break
            imgs.append(preprocess(s["image"].convert("RGB")))
            labels.append(int(s["label"]))
        return torch.stack(imgs), np.array(labels)

    return (*process(train_ds, n_train), *process(test_ds, n_test))


# ──────────────────────────────────────────────────────────────────────────────
# Forward passes
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def standard_features(encoder, images, device, batch_size=128):
    """Standard full forward pass → output features."""
    encoder.eval().to(device)
    feats = []
    for i in range(0, len(images), batch_size):
        feats.append(encoder(images[i:i+batch_size].to(device)).cpu())
    encoder.cpu()
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


@torch.no_grad()
def split_depth_features(merged_enc, ft_enc, images, split_k, device, batch_size=128):
    """
    merged embedding + blocks[0..split_k] → ft blocks[split_k+1..11] → ft ln_post + proj.

    split_k = -1  → all blocks from ft_enc  (upper bound: pure SVHN-FT)
    split_k = 11  → all blocks from merged   (baseline: pure merged)
    """
    m = merged_enc.model.visual
    f = ft_enc.model.visual
    merged_enc.eval().to(device)
    ft_enc.eval().to(device)

    feats = []
    for i in range(0, len(images), batch_size):
        x = images[i:i+batch_size].to(device)

        # Shared embedding (use merged)
        x = m.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = m.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + m.positional_embedding.to(x.dtype)
        x = m.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD → LND

        # Transformer blocks
        for k in range(12):
            vis = m if k <= split_k else f
            x = vis.transformer.resblocks[k](x)

        # Final layers (use ft)
        x = x.permute(1, 0, 2)       # LND → NLD
        x = f.ln_post(x[:, 0, :])    # CLS token
        if f.proj is not None:
            x = x @ f.proj

        feats.append(x.cpu())

    merged_enc.cpu()
    ft_enc.cpu()
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Probing
# ──────────────────────────────────────────────────────────────────────────────

def probe(train_X, train_y, test_X, test_y, C=0.316):
    tr = normalize(train_X)
    te = normalize(test_X)
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(tr, train_y)
    return float(clf.score(te, test_y))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer samples")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    n_train = 1000 if args.quick else 5000
    n_test  = 500  if args.quick else 2000

    # ── Load models ──────────────────────────────────────────────────────────
    log.info("Loading pretrained encoder for preprocess...")
    pretrained = make_encoder()
    preprocess = pretrained.val_preprocess
    del pretrained

    log.info("Loading SVHN-FT...")
    svhn_ft = make_ft_encoder("SVHN")

    log.info("Loading 8-task merged model...")
    merged_8 = make_merged(N8)

    log.info("Loading 7-task merged model (excl. SVHN)...")
    merged_7 = make_merged(N7_NO_SVHN)

    # ── Load SVHN data ────────────────────────────────────────────────────────
    tr_imgs, tr_labels, te_imgs, te_labels = load_svhn(preprocess, n_train, n_test)
    log.info(f"SVHN: {len(tr_imgs)} train, {len(te_imgs)} test")

    results = {}

    # ── Experiment A: split-depth sweep (8-task merged → SVHN-FT) ────────────
    log.info("\n" + "="*60)
    log.info("EXPERIMENT A: Split-depth sweep (8-task merged + SVHN-FT)")
    log.info("="*60)

    split_ks = [-1, 7, 8, 9, 10, 11]
    labels_a = {
        -1: "All FT (UB)",
        7:  "Split K=7",
        8:  "Split K=8",
        9:  "Split K=9",
        10: "Split K=10",
        11: "All merged (baseline)",
    }

    for k in split_ks:
        tag = f"split_8task_k{k}"
        log.info(f"\n--- K={k} ({labels_a[k]}) ---")
        tr_f = split_depth_features(merged_8, svhn_ft, tr_imgs, k, device)
        te_f = split_depth_features(merged_8, svhn_ft, te_imgs, k, device)
        acc = probe(tr_f, tr_labels, te_f, te_labels)
        log.info(f"  SVHN probe acc: {acc:.4f}")
        results[tag] = {"acc": acc, "label": labels_a[k], "experiment": "A", "split_k": k}

    # ── Experiment B: 7-task merged + SVHN-FT block 11 only ─────────────────
    log.info("\n" + "="*60)
    log.info("EXPERIMENT B: 7-task merged (excl SVHN) + SVHN-FT block 11")
    log.info("="*60)

    # K=10 means: use merged_7 blocks 0..10, then SVHN-FT block 11
    log.info("--- 7-task merged K=10 (exclusive block-11 ownership) ---")
    tr_f = split_depth_features(merged_7, svhn_ft, tr_imgs, 10, device)
    te_f = split_depth_features(merged_7, svhn_ft, te_imgs, 10, device)
    acc = probe(tr_f, tr_labels, te_f, te_labels)
    log.info(f"  SVHN probe acc: {acc:.4f}")
    results["excl_block11_k10"] = {"acc": acc, "label": "7-task merge + FT block 11", "experiment": "B", "split_k": 10}

    # Also try 7-task with K=9 (FT handles blocks 10+11)
    log.info("--- 7-task merged K=9 (FT handles last 2 blocks) ---")
    tr_f = split_depth_features(merged_7, svhn_ft, tr_imgs, 9, device)
    te_f = split_depth_features(merged_7, svhn_ft, te_imgs, 9, device)
    acc = probe(tr_f, tr_labels, te_f, te_labels)
    log.info(f"  SVHN probe acc: {acc:.4f}")
    results["excl_block11_k9"] = {"acc": acc, "label": "7-task merge + FT blocks 10-11", "experiment": "B", "split_k": 9}

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info("="*60)
    log.info(f"{'Configuration':<40} {'SVHN Probe':>12}")
    log.info("-" * 55)
    for tag, r in results.items():
        log.info(f"{r['label']:<40} {r['acc']:>12.4f}")

    # Reference: FT upper bound
    ft_ref = results["split_8task_k-1"]["acc"]
    merged_ref = results["split_8task_k11"]["acc"]
    log.info(f"\nRecovery vs FT upper bound:")
    for tag, r in results.items():
        recovery = (r["acc"] - merged_ref) / (ft_ref - merged_ref + 1e-8) * 100
        log.info(f"  {r['label']:<38}: +{r['acc']-merged_ref:+.4f} ({recovery:.1f}% of gap recovered)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Experiment A sweep
    ax = axes[0]
    ks_a = sorted([r["split_k"] for r in results.values() if r["experiment"] == "A"])
    accs_a = [next(r["acc"] for r in results.values() if r["experiment"] == "A" and r["split_k"] == k) for k in ks_a]
    xlabels = [labels_a[k] for k in ks_a]
    colors = ["green" if k == -1 else ("red" if k == 11 else "steelblue") for k in ks_a]
    ax.bar(range(len(ks_a)), accs_a, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for j, (acc, k) in enumerate(zip(accs_a, ks_a)):
        ax.text(j, acc + 0.003, f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(ks_a)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("SVHN Probe Accuracy")
    ax.set_title("Proposal A: Split-Depth Inference\n(8-task merged + SVHN-FT late blocks)")
    ax.set_ylim(0.7, 1.02)
    ax.axhline(accs_a[-1], color="red", linestyle="--", alpha=0.4, label="All merged")
    ax.axhline(accs_a[0],  color="green", linestyle="--", alpha=0.4, label="All FT")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: Experiment B comparison
    ax = axes[1]
    b_configs = [
        ("All merged (baseline)",       results["split_8task_k11"]["acc"],   "red"),
        ("7-task + FT blocks 10-11",    results["excl_block11_k9"]["acc"],   "steelblue"),
        ("7-task + FT block 11",        results["excl_block11_k10"]["acc"],  "darkorange"),
        ("All FT (upper bound)",        results["split_8task_k-1"]["acc"],   "green"),
    ]
    labels_b, accs_b, colors_b = zip(*b_configs)
    ax.bar(range(len(b_configs)), accs_b, color=colors_b, alpha=0.85, edgecolor="black", linewidth=0.5)
    for j, acc in enumerate(accs_b):
        ax.text(j, acc + 0.003, f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(b_configs)))
    ax.set_xticklabels(labels_b, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("SVHN Probe Accuracy")
    ax.set_title("Proposal B: Exclusive Block-11 Ownership\n(7-task merge excl. SVHN + FT late blocks)")
    ax.set_ylim(0.7, 1.02)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "split_depth_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {fig_path}")

    # Save JSON
    json_path = OUTPUT_DIR / "split_depth_results.json"
    with open(json_path, "w") as f:
        json.dump({"n_train": n_train, "n_test": n_test, "results": results}, f, indent=2)
    log.info(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
