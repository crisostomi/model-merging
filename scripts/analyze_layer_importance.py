"""
Layer Importance & Depth Efficiency Analysis

For each N8 task and its fine-tuned model, extracts features at all 12 transformer
blocks + final output using forward hooks. Trains linear probes at each depth.
Computes depth efficiency: how much of the final accuracy is captured at each block.

Key questions:
  - How many layers do you need to reach 90% of the fine-tuned accuracy? Which ones?
  - Do different tasks rely on different depths?
  - Where does the merged model lag the FT model most?

Outputs:
  - results/layer_importance/depth_efficiency.json  — full results
  - results/layer_importance/depth_efficiency_heatmap.png  — tasks × blocks heatmap
  - results/layer_importance/depth_curves.png  — per-task probe acc curves
  - results/layer_importance/merged_vs_ft.png  — merged vs FT at each depth

Usage:
    sbatch slurm/launch_long.slurm scripts/analyze_layer_importance.py
    uv run python scripts/analyze_layer_importance.py --quick
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "layer_importance"
N_BLOCKS = 12


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

HF_DATASET_MAP = {
    "SUN397":   ("tanganke/sun397",      None,             "train", "test"),
    "Cars":     ("tanganke/stanford_cars", None,           "train", "test"),
    "RESISC45": ("tanganke/resisc45",    None,             "train", "test"),
    "EuroSAT":  ("tanganke/eurosat",     None,             "train", "test"),
    "SVHN":     ("ufldl-stanford/svhn",  "cropped_digits", "train", "test"),
    "GTSRB":    ("tanganke/gtsrb",       None,             "train", "test"),
    "MNIST":    ("ylecun/mnist",          None,             "train", "test"),
    "DTD":      ("tanganke/dtd",          None,             "train", "test"),
}

LABEL_FIELD_MAP = {
    "SUN397": "label", "Cars": "label", "RESISC45": "label", "EuroSAT": "label",
    "SVHN": "label", "GTSRB": "label", "MNIST": "label", "DTD": "label",
}

IMAGE_FIELD_MAP = {
    "SUN397": "image", "Cars": "image", "RESISC45": "image", "EuroSAT": "image",
    "SVHN": "image", "GTSRB": "image", "MNIST": "image", "DTD": "image",
}


def load_dataset(dataset_name, preprocess, n_train=1000, n_test=500):
    from datasets import load_dataset as hf_load
    repo, config, train_split, test_split = HF_DATASET_MAP[dataset_name]
    img_field = IMAGE_FIELD_MAP[dataset_name]
    lbl_field = LABEL_FIELD_MAP[dataset_name]

    log.info(f"  Loading {dataset_name} (train={n_train}, test={n_test})")
    load_kwargs = {"path": repo, "split": train_split}
    if config:
        load_kwargs["name"] = config
    train_ds = hf_load(**load_kwargs)
    load_kwargs["split"] = test_split
    test_ds = hf_load(**load_kwargs)

    def process(ds, n):
        imgs, labels = [], []
        for i, s in enumerate(ds):
            if i >= n:
                break
            img = s[img_field]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            else:
                from PIL import Image
                img = Image.fromarray(np.array(img)).convert("RGB")
            imgs.append(preprocess(img))
            labels.append(int(s[lbl_field]))
        return torch.stack(imgs), np.array(labels)

    tr_imgs, tr_labels = process(train_ds, n_train)
    te_imgs, te_labels = process(test_ds, n_test)
    return tr_imgs, tr_labels, te_imgs, te_labels


# ──────────────────────────────────────────────────────────────────────────────
# Block-level feature extraction via hooks
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_block_features(encoder, images, device, batch_size=128):
    """
    Extract CLS-token features at the output of each transformer block (0..11)
    plus the final projected output.

    Returns: dict {block_idx: np.array of shape (N, D)}
             plus "output" for the final projected features.

    Note on ViT-B-32 forward pass (open_clip):
      conv1 → class_embedding + positional_embedding → ln_pre →
      transformer.resblocks[0..11] → ln_post(CLS) → proj

    Intermediate tensor format in transformer: LND (seq_len, batch, hidden).
    """
    encoder.eval().to(device)
    vis = encoder.model.visual

    # Storage for intermediate features
    block_feats = {i: [] for i in range(N_BLOCKS)}
    output_feats = []

    handles = []

    def make_hook(block_idx):
        def hook(module, input, output):
            # output shape: [seq_len, batch, hidden] (LND)
            # CLS token is at position 0
            cls = output[0].detach().cpu()  # [batch, hidden]
            block_feats[block_idx].append(cls)
        return hook

    for i in range(N_BLOCKS):
        h = vis.transformer.resblocks[i].register_forward_hook(make_hook(i))
        handles.append(h)

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        out = encoder(batch)  # standard full forward pass
        output_feats.append(out.detach().cpu())

    for h in handles:
        h.remove()

    encoder.cpu()
    torch.cuda.empty_cache()

    result = {}
    for i in range(N_BLOCKS):
        result[i] = torch.cat(block_feats[i]).numpy()
    result["output"] = torch.cat(output_feats).numpy()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Probing
# ──────────────────────────────────────────────────────────────────────────────

def probe(train_X, train_y, test_X, test_y, C=0.316):
    tr = normalize(train_X)
    te = normalize(test_X)
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(tr, train_y)
    return float(clf.score(te, test_y))


def probe_all_depths(tr_block_feats, tr_labels, te_block_feats, te_labels):
    """Train probes at each block and at the final output. Returns dict depth→acc."""
    accs = {}
    for depth in list(range(N_BLOCKS)) + ["output"]:
        acc = probe(tr_block_feats[depth], tr_labels, te_block_feats[depth], te_labels)
        accs[str(depth)] = acc
    return accs


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer samples")
    parser.add_argument("--tasks", nargs="+", default=None, help="Subset of tasks (default: all N8)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    n_train = 500 if args.quick else 2000
    n_test  = 200 if args.quick else 1000

    tasks = args.tasks or N8

    # ── Load pretrained encoder for preprocess ────────────────────────────────
    log.info("Loading pretrained encoder for preprocess...")
    pretrained = make_encoder()
    preprocess = pretrained.val_preprocess
    del pretrained
    torch.cuda.empty_cache()

    # ── Load merged model ─────────────────────────────────────────────────────
    log.info("Loading 8-task merged model...")
    merged_8 = make_merged(N8)

    # ── Main analysis loop ────────────────────────────────────────────────────
    results = {}

    for dataset_name in tasks:
        log.info(f"\n{'='*60}")
        log.info(f"TASK: {dataset_name}")
        log.info("="*60)

        # Load data
        try:
            tr_imgs, tr_labels, te_imgs, te_labels = load_dataset(
                dataset_name, preprocess, n_train, n_test
            )
        except Exception as e:
            log.warning(f"Failed to load {dataset_name}: {e}")
            continue

        # Load FT encoder
        log.info(f"  Loading {dataset_name}-FT encoder...")
        ft_enc = make_ft_encoder(dataset_name)

        # Extract block features for FT
        log.info(f"  Extracting block features from FT model...")
        ft_tr_feats = extract_block_features(ft_enc, tr_imgs, device)
        ft_te_feats = extract_block_features(ft_enc, te_imgs, device)

        # Extract block features for merged
        log.info(f"  Extracting block features from merged model...")
        merged_tr_feats = extract_block_features(merged_8, tr_imgs, device)
        merged_te_feats = extract_block_features(merged_8, te_imgs, device)

        # Probe at each depth
        log.info(f"  Probing FT at each depth...")
        ft_accs = probe_all_depths(ft_tr_feats, tr_labels, ft_te_feats, te_labels)

        log.info(f"  Probing merged at each depth...")
        merged_accs = probe_all_depths(merged_tr_feats, tr_labels, merged_te_feats, te_labels)

        # Compute depth efficiency (relative to FT output)
        ft_final_acc = ft_accs["output"]
        depth_efficiency = {k: v / (ft_final_acc + 1e-8) for k, v in ft_accs.items()}

        # Find min depth for 90% efficiency
        threshold = 0.90
        min_depth_90 = None
        for d in range(N_BLOCKS):
            if depth_efficiency[str(d)] >= threshold:
                min_depth_90 = d
                break

        log.info(f"  FT final acc: {ft_final_acc:.4f}")
        log.info(f"  Merged final acc (output): {merged_accs['output']:.4f}")
        log.info(f"  Min depth for {threshold*100:.0f}% efficiency: block {min_depth_90}")
        log.info(f"  Depth efficiencies (blocks):")
        for d in range(N_BLOCKS):
            eff = depth_efficiency[str(d)]
            ft_a = ft_accs[str(d)]
            mg_a = merged_accs[str(d)]
            log.info(f"    Block {d:2d}: FT={ft_a:.4f}  Merged={mg_a:.4f}  Eff={eff:.3f}")
        log.info(f"    Output: FT={ft_accs['output']:.4f}  Merged={merged_accs['output']:.4f}")

        results[dataset_name] = {
            "ft_accs": ft_accs,
            "merged_accs": merged_accs,
            "depth_efficiency": depth_efficiency,
            "min_depth_90pct": min_depth_90,
            "ft_final_acc": ft_final_acc,
        }

        del ft_enc
        torch.cuda.empty_cache()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "depth_efficiency.json"
    with open(json_path, "w") as f:
        json.dump({"n_train": n_train, "n_test": n_test, "results": results}, f, indent=2)
    log.info(f"\nResults saved: {json_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("SUMMARY: Min depth for 90% efficiency")
    log.info("="*60)
    for ds, r in results.items():
        log.info(f"  {ds:<12}: block {r['min_depth_90pct']}  (FT final: {r['ft_final_acc']:.4f})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not results:
        log.warning("No results to plot")
        return

    completed_tasks = list(results.keys())
    depths = list(range(N_BLOCKS)) + ["output"]
    depth_labels = [str(d) for d in range(N_BLOCKS)] + ["out"]

    # 1. Depth efficiency heatmap (tasks × blocks)
    fig, ax = plt.subplots(figsize=(14, max(4, len(completed_tasks) * 0.6 + 1)))
    # Only use integer blocks for the heatmap (exclude "output" since it's always 1.0)
    heatmap_data = np.array([
        [results[ds]["depth_efficiency"][str(d)] for d in range(N_BLOCKS)]
        for ds in completed_tasks
    ])
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(N_BLOCKS))
    ax.set_xticklabels([f"B{d}" for d in range(N_BLOCKS)], fontsize=9)
    ax.set_yticks(range(len(completed_tasks)))
    ax.set_yticklabels(completed_tasks, fontsize=9)
    ax.set_xlabel("Block index")
    ax.set_title("Depth Efficiency: FT probe acc (block) / FT probe acc (output)\n"
                 "Green = task already well-solved; red = still building")
    # Annotate cells
    for i in range(len(completed_tasks)):
        for j in range(N_BLOCKS):
            val = heatmap_data[i, j]
            color = "black" if val > 0.75 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, label="Depth efficiency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depth_efficiency_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved depth_efficiency_heatmap.png")

    # 2. Per-task FT depth curves
    n_tasks = len(completed_tasks)
    ncols = min(4, n_tasks)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    x_numeric = list(range(N_BLOCKS + 1))  # 0..11 + "output" as 12
    x_labels = [str(d) for d in range(N_BLOCKS)] + ["out"]

    for idx, ds in enumerate(completed_tasks):
        ax = axes[idx]
        r = results[ds]
        ft_curve = [r["ft_accs"][str(d)] for d in range(N_BLOCKS)] + [r["ft_accs"]["output"]]
        merged_curve = [r["merged_accs"][str(d)] for d in range(N_BLOCKS)] + [r["merged_accs"]["output"]]

        ax.plot(x_numeric, ft_curve, "b-o", markersize=4, label="FT", linewidth=1.5)
        ax.plot(x_numeric, merged_curve, "r--s", markersize=4, label="Merged", linewidth=1.5)
        ax.axhline(r["ft_final_acc"] * 0.90, color="gray", linestyle=":", alpha=0.7, label="90% FT")

        if r["min_depth_90pct"] is not None:
            ax.axvline(r["min_depth_90pct"], color="green", linestyle="--", alpha=0.5)

        ax.set_title(f"{ds}", fontsize=10)
        ax.set_xticks(x_numeric[::2])
        ax.set_xticklabels(x_labels[::2], fontsize=8)
        ax.set_ylabel("Probe acc", fontsize=8)
        ax.set_ylim(bottom=max(0, min(ft_curve + merged_curve) - 0.05))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for idx in range(len(completed_tasks), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Probe Accuracy at Each Depth: FT vs Merged Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depth_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved depth_curves.png")

    # 3. Gap (merged - FT) at each depth heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(completed_tasks) * 0.6 + 1)))
    gap_data = np.array([
        [results[ds]["merged_accs"][str(d)] - results[ds]["ft_accs"][str(d)]
         for d in range(N_BLOCKS)]
        for ds in completed_tasks
    ])
    vmax = max(abs(gap_data).max(), 0.05)
    im = ax.imshow(gap_data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_BLOCKS))
    ax.set_xticklabels([f"B{d}" for d in range(N_BLOCKS)], fontsize=9)
    ax.set_yticks(range(len(completed_tasks)))
    ax.set_yticklabels(completed_tasks, fontsize=9)
    ax.set_xlabel("Block index")
    ax.set_title("Merged − FT probe accuracy at each block\n"
                 "Blue = merged better; red = merged worse")
    for i in range(len(completed_tasks)):
        for j in range(N_BLOCKS):
            val = gap_data[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="Merged − FT")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "merged_vs_ft_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved merged_vs_ft_gap.png")

    # 4. Min depth for 90% bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    min_depths = [results[ds]["min_depth_90pct"] for ds in completed_tasks]
    bars = ax.bar(range(len(completed_tasks)), min_depths, color="steelblue",
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    for j, (d, ds) in enumerate(zip(min_depths, completed_tasks)):
        label = f"B{d}" if d is not None else "N/A"
        ax.text(j, (d or 0) + 0.1, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(completed_tasks)))
    ax.set_xticklabels(completed_tasks, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Block index")
    ax.set_title("Minimum block depth to reach 90% of FT final accuracy")
    ax.set_ylim(0, N_BLOCKS)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "min_depth_90pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved min_depth_90pct.png")

    log.info(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
