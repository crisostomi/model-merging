"""
QK Routing — Selective Head Transplant for Block 11

Proposal C: In block 11, replace Q and K projection weights of SVHN-selective
attention heads {3, 4, 8} (0-indexed) from SVHN-FT into the merged model.
Keep V and out_proj as merged (preserving multi-task representation space).

The insight: merged block 11 lacks task-specific routing capability because
attention heads are homogenized (max selectivity ≤0.075 vs 0.15-0.18 in FT).
Transplanting QK for the selective heads lets the merged model attend to
SVHN-relevant tokens while keeping V/out_proj compatible with other tasks.

Variants tested:
  qk_only       — swap Q, K of heads {3, 4, 8} only
  qkv           — swap Q, K, V of heads {3, 4, 8}
  qkv_out       — swap Q, K, V + corresponding out_proj rows
  full_attn     — swap entire in_proj + out_proj of block 11 (sanity check)
  full_block    — swap entire block 11 (sanity check — expected to fail)

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_qk_routing.py
    uv run python scripts/analyze_qk_routing.py --quick
"""

import argparse
import copy
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "qk_routing"

# ViT-B-32: hidden=768, n_heads=12, head_dim=64
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = 768

# Heads identified as SVHN-selective in SVHN-FT (0-indexed)
SVHN_SELECTIVE_HEADS = [3, 4, 8]


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
# QK routing surgery
# ──────────────────────────────────────────────────────────────────────────────

def get_block11_attn_key(prefix="model.visual.transformer.resblocks.11"):
    """Return key names for block 11 attention."""
    return {
        "in_proj": f"{prefix}.attn.in_proj_weight",
        "in_proj_bias": f"{prefix}.attn.in_proj_bias",
        "out_proj": f"{prefix}.attn.out_proj.weight",
        "out_proj_bias": f"{prefix}.attn.out_proj.bias",
    }


def transplant_qk(merged_sd, donor_sd, heads, block=11):
    """
    Replace Q and K rows (in in_proj_weight) for the specified heads in the
    merged model with those from the donor (SVHN-FT).

    in_proj_weight layout: [3*H*D, H*D] where H=n_heads, D=head_dim
      Q: rows [0 : H*D]
      K: rows [H*D : 2*H*D]
      V: rows [2*H*D : 3*H*D]

    For head h:
      Q rows: [h*D : (h+1)*D]
      K rows: [H*D + h*D : H*D + (h+1)*D]
      V rows: [2*H*D + h*D : 2*H*D + (h+1)*D]
    """
    prefix = f"model.visual.transformer.resblocks.{block}"
    in_proj_key = f"{prefix}.attn.in_proj_weight"

    sd = copy.deepcopy(merged_sd)
    W = sd[in_proj_key].clone()  # [3*768, 768]
    W_donor = donor_sd[in_proj_key]

    for h in heads:
        q_start = h * HEAD_DIM
        k_start = HIDDEN + h * HEAD_DIM
        W[q_start : q_start + HEAD_DIM] = W_donor[q_start : q_start + HEAD_DIM]
        W[k_start : k_start + HEAD_DIM] = W_donor[k_start : k_start + HEAD_DIM]

    sd[in_proj_key] = W
    return sd


def transplant_qkv(merged_sd, donor_sd, heads, block=11):
    """Replace Q, K, V for the specified heads."""
    prefix = f"model.visual.transformer.resblocks.{block}"
    in_proj_key = f"{prefix}.attn.in_proj_weight"

    sd = copy.deepcopy(merged_sd)
    W = sd[in_proj_key].clone()
    W_donor = donor_sd[in_proj_key]

    for h in heads:
        q_start = h * HEAD_DIM
        k_start = HIDDEN + h * HEAD_DIM
        v_start = 2 * HIDDEN + h * HEAD_DIM
        W[q_start : q_start + HEAD_DIM] = W_donor[q_start : q_start + HEAD_DIM]
        W[k_start : k_start + HEAD_DIM] = W_donor[k_start : k_start + HEAD_DIM]
        W[v_start : v_start + HEAD_DIM] = W_donor[v_start : v_start + HEAD_DIM]

    sd[in_proj_key] = W
    return sd


def transplant_qkv_out(merged_sd, donor_sd, heads, block=11):
    """Replace Q, K, V and corresponding out_proj columns for the specified heads."""
    prefix = f"model.visual.transformer.resblocks.{block}"
    in_proj_key = f"{prefix}.attn.in_proj_weight"
    out_proj_key = f"{prefix}.attn.out_proj.weight"

    sd = transplant_qkv(merged_sd, donor_sd, heads, block)

    # out_proj: [hidden, hidden] — head h contributes columns [h*D : (h+1)*D]
    W_out = sd[out_proj_key].clone()
    W_out_donor = donor_sd[out_proj_key]
    for h in heads:
        col_start = h * HEAD_DIM
        W_out[:, col_start : col_start + HEAD_DIM] = W_out_donor[:, col_start : col_start + HEAD_DIM]
    sd[out_proj_key] = W_out

    return sd


def transplant_full_attn(merged_sd, donor_sd, block=11):
    """Transplant entire attention (in_proj + out_proj) from donor."""
    prefix = f"model.visual.transformer.resblocks.{block}"
    keys = [
        f"{prefix}.attn.in_proj_weight",
        f"{prefix}.attn.in_proj_bias",
        f"{prefix}.attn.out_proj.weight",
        f"{prefix}.attn.out_proj.bias",
    ]
    sd = copy.deepcopy(merged_sd)
    for k in keys:
        if k in donor_sd:
            sd[k] = donor_sd[k].clone()
    return sd


def transplant_full_block(merged_sd, donor_sd, block=11):
    """Transplant entire block from donor (sanity check)."""
    prefix = f"model.visual.transformer.resblocks.{block}"
    sd = copy.deepcopy(merged_sd)
    for k in list(donor_sd.keys()):
        if k.startswith(prefix):
            sd[k] = donor_sd[k].clone()
    return sd


def apply_sd_to_encoder(encoder, sd):
    """Return a copy of encoder with the given state dict."""
    enc = copy.deepcopy(encoder)
    enc.load_state_dict(sd, strict=False)
    return enc


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
# Feature extraction & probing
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_features(encoder, images, device, batch_size=128):
    encoder.eval().to(device)
    feats = []
    for i in range(0, len(images), batch_size):
        feats.append(encoder(images[i:i+batch_size].to(device)).cpu())
    encoder.cpu()
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


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

    log.info("Loading SVHN-FT...")
    svhn_ft_sd = load_sd(MODEL_NAME, "SVHN")
    svhn_ft_enc = make_encoder()
    svhn_ft_enc.load_state_dict(svhn_ft_sd, strict=False)

    log.info("Loading 8-task merged model...")
    merged_8 = make_merged(N8)
    merged_8_sd = merged_8.state_dict()

    del pretrained
    torch.cuda.empty_cache()

    # ── Load SVHN data ────────────────────────────────────────────────────────
    tr_imgs, tr_labels, te_imgs, te_labels = load_svhn(preprocess, n_train, n_test)
    log.info(f"SVHN: {len(tr_imgs)} train, {len(te_imgs)} test")

    results = {}

    # ── Baseline: pure merged ────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("Baseline: pure 8-task merged model")
    tr_f = get_features(merged_8, tr_imgs, device)
    te_f = get_features(merged_8, te_imgs, device)
    acc = probe(tr_f, tr_labels, te_f, te_labels)
    log.info(f"  Merged baseline acc: {acc:.4f}")
    results["merged_baseline"] = {"acc": acc, "label": "Merged (baseline)"}

    # ── Upper bound: pure SVHN-FT ─────────────────────────────────────────────
    log.info("Upper bound: pure SVHN-FT")
    tr_f = get_features(svhn_ft_enc, tr_imgs, device)
    te_f = get_features(svhn_ft_enc, te_imgs, device)
    acc = probe(tr_f, tr_labels, te_f, te_labels)
    log.info(f"  SVHN-FT acc: {acc:.4f}")
    results["svhn_ft"] = {"acc": acc, "label": "SVHN-FT (upper bound)"}

    # ── Variants ─────────────────────────────────────────────────────────────
    variants = [
        ("qk_heads348",    transplant_qk,         {"merged_sd": merged_8_sd, "donor_sd": svhn_ft_sd, "heads": SVHN_SELECTIVE_HEADS}),
        ("qkv_heads348",   transplant_qkv,        {"merged_sd": merged_8_sd, "donor_sd": svhn_ft_sd, "heads": SVHN_SELECTIVE_HEADS}),
        ("qkv_out_heads348", transplant_qkv_out,  {"merged_sd": merged_8_sd, "donor_sd": svhn_ft_sd, "heads": SVHN_SELECTIVE_HEADS}),
        ("full_attn",      transplant_full_attn,  {"merged_sd": merged_8_sd, "donor_sd": svhn_ft_sd}),
        ("full_block",     transplant_full_block, {"merged_sd": merged_8_sd, "donor_sd": svhn_ft_sd}),
    ]

    labels = {
        "qk_heads348":     "QK heads {3,4,8} from FT",
        "qkv_heads348":    "QKV heads {3,4,8} from FT",
        "qkv_out_heads348": "QKV+out heads {3,4,8} from FT",
        "full_attn":       "Full attn block 11 from FT",
        "full_block":      "Full block 11 from FT",
    }

    for name, fn, kwargs in variants:
        log.info(f"\n--- {labels[name]} ---")
        patched_sd = fn(**kwargs)
        patched_enc = apply_sd_to_encoder(merged_8, patched_sd)

        tr_f = get_features(patched_enc, tr_imgs, device)
        te_f = get_features(patched_enc, te_imgs, device)
        acc = probe(tr_f, tr_labels, te_f, te_labels)
        log.info(f"  Probe acc: {acc:.4f}")
        results[name] = {"acc": acc, "label": labels[name]}

        del patched_enc
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SUMMARY")
    log.info("="*60)
    baseline = results["merged_baseline"]["acc"]
    ub = results["svhn_ft"]["acc"]
    log.info(f"{'Configuration':<45} {'Acc':>8} {'vs merged':>10} {'Recovery':>10}")
    log.info("-" * 75)
    for tag, r in results.items():
        delta = r["acc"] - baseline
        recovery = (r["acc"] - baseline) / (ub - baseline + 1e-8) * 100
        log.info(f"{r['label']:<45} {r['acc']:>8.4f} {delta:>+10.4f} {recovery:>9.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    order = ["merged_baseline", "qk_heads348", "qkv_heads348", "qkv_out_heads348",
             "full_attn", "full_block", "svhn_ft"]
    plot_labels = [results[k]["label"] for k in order]
    plot_accs = [results[k]["acc"] for k in order]
    colors = ["red"] + ["steelblue"] * 4 + ["darkorange", "green"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(order)), plot_accs, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for j, acc in enumerate(plot_accs):
        ax.text(j, acc + 0.003, f"{acc:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(plot_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("SVHN Probe Accuracy")
    ax.set_title("Proposal C: QK Routing — Selective Head Transplant (Block 11)\n"
                 "Transplanting Q,K weights of SVHN-selective heads {3,4,8} into merged model")
    ax.set_ylim(0.7, 1.02)
    ax.axhline(results["merged_baseline"]["acc"], color="red", linestyle="--",
               alpha=0.4, label="Merged baseline")
    ax.axhline(results["svhn_ft"]["acc"], color="green", linestyle="--",
               alpha=0.4, label="SVHN-FT upper bound")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "qk_routing_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {fig_path}")

    # Save JSON
    json_path = OUTPUT_DIR / "qk_routing_results.json"
    with open(json_path, "w") as f:
        json.dump({"n_train": n_train, "n_test": n_test, "results": results}, f, indent=2)
    log.info(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
