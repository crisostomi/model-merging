"""
Linear Probe: Is SVHN Information Preserved but Rotated in Merged Representations?

Tests whether the merged model's representations still contain SVHN-discriminative
information by training a fresh linear classifier on extracted features.

If probe_acc >> original_head_acc: information preserved, alignment is bottleneck
If probe_acc ~ original_head_acc: information genuinely lost

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_linear_probe.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_linear_probe.py --quick
"""

import argparse
import json
import logging
import os
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
from model_merging.utils.utils import print_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "linear_probe"


def load_state_dict_from_hf_cache(model_name, dataset_name="base"):
    from huggingface_hub import hf_hub_download
    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    return torch.load(ckpt_path, map_location="cpu")


def create_encoder(model_name=MODEL_NAME):
    return ImageEncoder(model_name)


def load_finetuned_encoder(dataset_name, model_name=MODEL_NAME):
    encoder = create_encoder(model_name)
    ft_sd = load_state_dict_from_hf_cache(model_name, dataset_name)
    encoder.load_state_dict(ft_sd, strict=False)
    return encoder


def create_merged_model(datasets, model_name=MODEL_NAME):
    logger.info(f"Creating merged model with {len(datasets)} tasks: {datasets}")
    encoder = create_encoder(model_name)
    finetuned = {}
    for ds in datasets:
        logger.info(f"  Loading finetuned: {ds}")
        finetuned[ds] = load_state_dict_from_hf_cache(model_name, ds)
    merger = InterferenceAwareMerger(
        use_mp_edge=True, use_isotropic=True, mp_min_rank=4, mp_max_rank=128
    )
    merged = merger.merge(encoder, finetuned)
    return merged


def load_svhn_data(preprocess, n_train=10000, n_test=10000):
    from datasets import load_dataset as hf_load_dataset
    logger.info(f"Loading SVHN data (train={n_train}, test={n_test})...")
    train_ds = hf_load_dataset("ufldl-stanford/svhn", "cropped_digits", split="train")
    test_ds = hf_load_dataset("ufldl-stanford/svhn", "cropped_digits", split="test")

    def process_split(ds, n):
        images, labels = [], []
        for i, sample in enumerate(ds):
            if i >= n:
                break
            images.append(preprocess(sample["image"].convert("RGB")))
            labels.append(sample["label"])
        return torch.stack(images), np.array(labels)

    train_imgs, train_labels = process_split(train_ds, n_train)
    test_imgs, test_labels = process_split(test_ds, n_test)
    logger.info(f"  Train: {len(train_imgs)}, Test: {len(test_imgs)}, Classes: {len(np.unique(train_labels))}")
    return train_imgs, train_labels, test_imgs, test_labels


@torch.no_grad()
def extract_features(model, images, device, batch_size=256):
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


@torch.no_grad()
def extract_multilayer_features(model, images, device, batch_size=256):
    """Extract features at multiple layers: block outputs, ln_post, and final output."""
    from collections import defaultdict
    model.eval()
    model.to(device)

    all_features = defaultdict(list)
    visual = model.model.visual

    # Register hooks for intermediate layers
    hooks = []
    hook_data = {}

    # Hook last 3 blocks + ln_post
    for block_idx in [9, 10, 11]:
        name = f"block_{block_idx}"
        def make_hook(n):
            def hook_fn(module, input, output):
                # output: (seq_len, batch, hidden) — CLS at index 0
                hook_data[n] = output[0].detach().cpu()  # CLS token
            return hook_fn
        hooks.append(visual.transformer.resblocks[block_idx].register_forward_hook(make_hook(name)))

    def ln_post_hook(module, input, output):
        hook_data["ln_post"] = output.detach().cpu()
    hooks.append(visual.ln_post.register_forward_hook(ln_post_hook))

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        out = model(batch)
        all_features["output"].append(out.cpu())
        for name, data in hook_data.items():
            all_features[name].append(data)
        hook_data.clear()

    for h in hooks:
        h.remove()
    model.cpu()
    torch.cuda.empty_cache()

    return {name: torch.cat(tensors, dim=0).float().numpy() for name, tensors in all_features.items()}


def train_linear_probe(train_features, train_labels, C=0.316):
    """Train logistic regression. C=0.316 (sqrt(0.1)) is a common CLIP probe default."""
    train_norm = normalize(train_features, norm="l2")
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(train_norm, train_labels)
    train_acc = clf.score(train_norm, train_labels)
    logger.info(f"  Probe train acc: {train_acc:.4f}")
    return clf


def evaluate_probe(clf, test_features, test_labels):
    test_norm = normalize(test_features, norm="l2")
    return clf.score(test_norm, test_labels)


def plot_results(results, output_dir):
    models = list(results.keys())
    probe_accs = [results[m]["probe_accuracy"] for m in models]

    model_labels = {
        "pretrained": "Pretrained\nCLIP",
        "svhn_ft": "SVHN\nFine-tuned",
        "merged_8": "Merged\nAll-8",
        "merged_7": "Merged\nNo-EuroSAT",
    }
    colors = {
        "pretrained": "gray",
        "svhn_ft": "tab:brown",
        "merged_8": "tab:red",
        "merged_7": "tab:green",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, probe_accs, color=[colors.get(m, "gray") for m in models],
                  alpha=0.85, edgecolor="black", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, probe_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([model_labels.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("SVHN Test Accuracy (Linear Probe)", fontsize=11)
    ax.set_title("Information Content: Linear Probe on Model Features", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / "linear_probe_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--multilayer", action="store_true", help="Also probe at intermediate layers")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_train = 2000 if args.quick else args.n_train
    n_test = 2000 if args.quick else args.n_test

    # Load models
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading models")
    logger.info("=" * 60)

    pretrained = create_encoder(MODEL_NAME)
    preprocess = pretrained.val_preprocess
    svhn_ft = load_finetuned_encoder("SVHN", MODEL_NAME)
    print_memory("after svhn_ft")

    merged_8 = create_merged_model(N8_DATASETS, MODEL_NAME)
    print_memory("after merged_8")

    no_eurosat = [d for d in N8_DATASETS if d != "EuroSAT"]
    merged_7 = create_merged_model(no_eurosat, MODEL_NAME)
    print_memory("after merged_7")

    models = {"pretrained": pretrained, "svhn_ft": svhn_ft, "merged_8": merged_8, "merged_7": merged_7}

    # Load data
    logger.info("=" * 60)
    logger.info("PHASE 2: Loading SVHN data")
    logger.info("=" * 60)
    train_imgs, train_labels, test_imgs, test_labels = load_svhn_data(preprocess, n_train, n_test)

    # Extract features and train probes
    logger.info("=" * 60)
    logger.info("PHASE 3: Feature extraction + linear probes")
    logger.info("=" * 60)

    results = {}
    for model_name, model in models.items():
        logger.info(f"\n--- {model_name} ---")
        train_feats = extract_features(model, train_imgs, device)
        test_feats = extract_features(model, test_imgs, device)
        logger.info(f"  Features shape: {train_feats.shape}")

        clf = train_linear_probe(train_feats, train_labels)
        probe_acc = evaluate_probe(clf, test_feats, test_labels)
        logger.info(f"  PROBE TEST ACC: {probe_acc:.4f}")

        results[model_name] = {
            "probe_accuracy": float(probe_acc),
            "feature_dim": train_feats.shape[1],
        }
        print_memory(f"after {model_name}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'Probe Acc':>10}")
    logger.info("-" * 35)
    for m, r in results.items():
        logger.info(f"{m:<20} {r['probe_accuracy']:>10.4f}")

    m8 = results["merged_8"]["probe_accuracy"]
    ft = results["svhn_ft"]["probe_accuracy"]
    pre = results["pretrained"]["probe_accuracy"]
    logger.info(f"\nRecovery: merged_8 probe / svhn_ft probe = {m8/ft:.4f}")
    logger.info(f"Gain over pretrained: {m8 - pre:+.4f}")

    if m8 > 0.90:
        logger.info("CONCLUSION: Merged model preserves most SVHN information — the issue is classifier alignment.")
    elif m8 > 0.80:
        logger.info("CONCLUSION: Merged model preserves significant SVHN info but some is lost.")
    else:
        logger.info("CONCLUSION: Merged model has lost substantial SVHN-discriminative information.")

    # Save
    plot_results(results, OUTPUT_DIR)
    json_path = OUTPUT_DIR / "linear_probe_results.json"
    with open(json_path, "w") as f:
        json.dump({"metadata": {"n_train": n_train, "n_test": n_test, "model": MODEL_NAME}, "results": results}, f, indent=2)
    logger.info(f"Results saved to: {json_path}")

    # --- Multi-layer probing ---
    if args.multilayer:
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-LAYER PROBING")
        logger.info("=" * 60)

        focus_models = {"svhn_ft": models["svhn_ft"], "merged_8": models["merged_8"]}
        layer_results = {}

        for model_name, model in focus_models.items():
            logger.info(f"\n--- {model_name} (multilayer) ---")
            train_ml = extract_multilayer_features(model, train_imgs, device)
            test_ml = extract_multilayer_features(model, test_imgs, device)

            layer_results[model_name] = {}
            for layer_name in sorted(train_ml.keys(), key=lambda x: (
                0 if x.startswith("block_") else 1 if x == "ln_post" else 2,
                int(x.split("_")[1]) if x.startswith("block_") else 999,
            )):
                logger.info(f"  Layer: {layer_name} (dim={train_ml[layer_name].shape[1]})")
                clf = train_linear_probe(train_ml[layer_name], train_labels)
                acc = evaluate_probe(clf, test_ml[layer_name], test_labels)
                logger.info(f"    PROBE ACC: {acc:.4f}")
                layer_results[model_name][layer_name] = {
                    "probe_accuracy": float(acc),
                    "feature_dim": train_ml[layer_name].shape[1],
                }

        # Summary table
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-LAYER RESULTS")
        logger.info("=" * 60)
        layers = list(layer_results["svhn_ft"].keys())
        logger.info(f"{'Layer':<12} {'Dim':>5} {'SVHN-FT':>10} {'Merged-8':>10} {'Gap':>8}")
        logger.info("-" * 50)
        for layer in layers:
            ft_acc = layer_results["svhn_ft"][layer]["probe_accuracy"]
            m8_acc = layer_results["merged_8"][layer]["probe_accuracy"]
            dim = layer_results["svhn_ft"][layer]["feature_dim"]
            gap = m8_acc - ft_acc
            logger.info(f"{layer:<12} {dim:>5} {ft_acc:>10.4f} {m8_acc:>10.4f} {gap:>+8.4f}")

        # Plot multi-layer comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(layers))
        width = 0.35
        ft_accs = [layer_results["svhn_ft"][l]["probe_accuracy"] for l in layers]
        m8_accs = [layer_results["merged_8"][l]["probe_accuracy"] for l in layers]

        bars1 = ax.bar(x - width/2, ft_accs, width, label="SVHN Fine-tuned", color="tab:brown", alpha=0.85)
        bars2 = ax.bar(x + width/2, m8_accs, width, label="Merged All-8", color="tab:red", alpha=0.85)

        for bar, val in zip(bars1, ft_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, m8_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        dims = [layer_results["svhn_ft"][l]["feature_dim"] for l in layers]
        ax.set_xticklabels([f"{l}\n({d}d)" for l, d in zip(layers, dims)], fontsize=9)
        ax.set_ylabel("SVHN Probe Accuracy", fontsize=11)
        ax.set_title("Multi-Layer Probing: Where Is Information Lost?", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = OUTPUT_DIR / "multilayer_probe_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

        # Save multilayer results
        ml_json_path = OUTPUT_DIR / "multilayer_probe_results.json"
        with open(ml_json_path, "w") as f:
            json.dump(layer_results, f, indent=2)
        logger.info(f"Multilayer results saved to: {ml_json_path}")


if __name__ == "__main__":
    main()
