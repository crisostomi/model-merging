"""
Mechanistic Analysis of Block 11: Attention vs MLP Contributions

Block 11 (final transformer block) single-handedly creates ~42 pp of SVHN
classification accuracy in the merged model (probe accuracy jumps from 41.8%
at block 9 to 83.8% at block 11).  In the SVHN fine-tuned model, features
build gradually (93.2% at block 9 -> 96.7% at block 11).

This script dissects block 11 to answer:
  - Is it the attention heads or MLP (or both) that create task-relevant features?

Hooks:
  1. Block 10 output (input to block 11)
  2. After attention + residual (before MLP)
  3. After full block 11 (attention + MLP + residuals)

For each hook point, a linear probe (LogisticRegression) is trained on SVHN
train data and evaluated on SVHN test data.

Outputs (results/activation_analysis/):
  - block11_mechanistic.json          — all probe accuracies
  - block11_subcomponent_probes.png   — bar chart: block_10 -> +attn -> +MLP

Usage:
    sbatch slurm/launch_analysis.slurm scripts/analyze_block11_mechanistic.py
    sbatch slurm/launch_analysis.slurm scripts/analyze_block11_mechanistic.py --quick
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "activation_analysis"
BATCH_SIZE = 64
BLOCK_IDX = 11  # The block we are dissecting


# --------------------------------------------------------------------------- #
# Model loading  (same pattern as other analysis scripts)
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_svhn_data(preprocess, n_train=5000, n_test=5000):
    """Load SVHN train and test splits separately for proper probe evaluation."""
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
    logger.info(
        f"  Train: {len(train_imgs)}, Test: {len(test_imgs)}, "
        f"Classes: {len(np.unique(train_labels))}"
    )
    return train_imgs, train_labels, test_imgs, test_labels


# --------------------------------------------------------------------------- #
# Hook-based activation extraction
# --------------------------------------------------------------------------- #

def register_block11_hooks(model):
    """Register hooks to capture block 11 sub-component activations.

    Hook points (all CLS-token, seq-first format):
        block_10_out   : output of block 10 = input to block 11
        block_11_out   : full block 11 output (post-MLP + residual)
        attn_output    : raw attention output before residual addition
    """
    activations = {}
    handles = []
    visual = model.model.visual

    # 1. Block 10 output (= block 11 input)
    def block10_hook(module, input, output):
        activations["block_10_out"] = output.detach().cpu()  # (seq, batch, hidden)
    handles.append(
        visual.transformer.resblocks[10].register_forward_hook(block10_hook)
    )

    # 2. Full block 11 output
    def block11_hook(module, input, output):
        activations["block_11_out"] = output.detach().cpu()
    handles.append(
        visual.transformer.resblocks[BLOCK_IDX].register_forward_hook(block11_hook)
    )

    # 3. Attention output (before residual).
    #    The attention call inside ResidualAttentionBlock is:
    #      x = x + self.ls_1(self.attention(self.ln_1(x)))
    #    self.attention is a method that calls self.attn (nn.MultiheadAttention).
    #    We hook the full nn.MultiheadAttention to get the attention output.
    #    Its output is a tuple: (attn_output, attn_weights).
    #    attn_output shape: (seq_len, batch, hidden_dim).
    block11 = visual.transformer.resblocks[BLOCK_IDX]

    def mha_hook(module, input, output):
        # output is (attn_output, attn_weights_or_None)
        attn_output = output[0] if isinstance(output, tuple) else output
        activations["attn_output"] = attn_output.detach().cpu()  # (seq, batch, hidden)
    handles.append(block11.attn.register_forward_hook(mha_hook))

    return activations, handles


@torch.no_grad()
def extract_block11_activations(model, images, device, batch_size=BATCH_SIZE):
    """Extract block 11 sub-component activations for all images.

    Returns dict with keys:
        block_10_out        : (n_samples, 768) CLS token after block 10
        block_11_post_attn  : (n_samples, 768) CLS token after attention+residual
        block_11_out        : (n_samples, 768) CLS token after full block 11
    """
    model.eval()
    model.to(device)

    collectors = defaultdict(list)
    activations_dict, handles = register_block11_hooks(model)

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        _ = model(batch)

        # --- Block 10 output (CLS token at index 0, seq-first) ---
        block10 = activations_dict["block_10_out"]  # (seq, batch, hidden)
        collectors["block_10_out"].append(block10[0].clone())  # (batch, hidden)

        # --- Full block 11 output ---
        block11 = activations_dict["block_11_out"]  # (seq, batch, hidden)
        collectors["block_11_out"].append(block11[0].clone())

        # --- Attention output (raw, before residual) ---
        attn_out = activations_dict["attn_output"]  # (seq, batch, hidden)
        # Post-attention = block_10_out + attn_output  (ls_1 is Identity for ViT-B-32)
        post_attn = block10[0] + attn_out[0]  # (batch, hidden)
        collectors["block_11_post_attn"].append(post_attn.clone())

        activations_dict.clear()

    for h in handles:
        h.remove()
    model.cpu()
    torch.cuda.empty_cache()

    result = {}
    for key in ["block_10_out", "block_11_post_attn", "block_11_out"]:
        result[key] = torch.cat(collectors[key], dim=0).float()  # (n, 768)

    return result


# --------------------------------------------------------------------------- #
# Linear probes
# --------------------------------------------------------------------------- #

def train_linear_probe(train_features, train_labels, C=0.316):
    """Train L2-regularized logistic regression (C=0.316 is a common CLIP default)."""
    train_norm = normalize(train_features, norm="l2")
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs", n_jobs=-1)
    clf.fit(train_norm, train_labels)
    return clf


def evaluate_probe(clf, test_features, test_labels):
    test_norm = normalize(test_features, norm="l2")
    return clf.score(test_norm, test_labels)


def run_probes(train_acts, test_acts, train_labels, test_labels, model_name):
    """Train and evaluate linear probes at each sub-component for one model.

    Returns dict of {probe_point: accuracy}.
    """
    results = {}

    # Sub-component probes (768d each)
    for key in ["block_10_out", "block_11_post_attn", "block_11_out"]:
        logger.info(f"  [{model_name}] Probing {key} (dim={train_acts[key].shape[1]})")
        train_feat = train_acts[key].numpy()
        test_feat = test_acts[key].numpy()
        clf = train_linear_probe(train_feat, train_labels)
        acc = evaluate_probe(clf, test_feat, test_labels)
        results[key] = float(acc)
        logger.info(f"    -> {acc:.4f}")

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

MODEL_DISPLAY = {
    "pretrained": "Pretrained",
    "svhn_ft": "SVHN Fine-tuned",
    "merged_8": "Merged All-8",
}

MODEL_COLORS = {
    "pretrained": "gray",
    "svhn_ft": "#8B4513",
    "merged_8": "tab:red",
}


def plot_subcomponent_bars(all_results, output_dir):
    """Bar chart: probe accuracy at block_10 -> +attention -> +MLP for each model."""
    models = list(all_results.keys())
    probe_points = ["block_10_out", "block_11_post_attn", "block_11_out"]
    probe_labels = ["Block 10\n(input to B11)", "Block 10 + Attn\n(post-attention)", "Block 11\n(post-MLP)"]
    n_models = len(models)
    n_points = len(probe_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_points)
    width = 0.8 / n_models

    for i, model in enumerate(models):
        accs = [all_results[model][pp] for pp in probe_points]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, accs, width,
            label=MODEL_DISPLAY.get(model, model),
            color=MODEL_COLORS.get(model, f"C{i}"),
            alpha=0.85, edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    # Draw arrows showing the gain from attention and MLP
    for i, model in enumerate(models):
        accs = [all_results[model][pp] for pp in probe_points]
        offset = (i - n_models / 2 + 0.5) * width
        attn_gain = accs[1] - accs[0]
        mlp_gain = accs[2] - accs[1]
        # Annotate gains between bars
        mid_x_attn = (x[0] + offset + x[1] + offset) / 2
        mid_y_attn = max(accs[0], accs[1]) + 0.04
        ax.annotate(
            f"+{attn_gain:.1%}",
            xy=(mid_x_attn, mid_y_attn),
            fontsize=7, ha="center", color=MODEL_COLORS.get(model, f"C{i}"),
            fontweight="bold",
        )
        mid_x_mlp = (x[1] + offset + x[2] + offset) / 2
        mid_y_mlp = max(accs[1], accs[2]) + 0.04
        ax.annotate(
            f"+{mlp_gain:.1%}",
            xy=(mid_x_mlp, mid_y_mlp),
            fontsize=7, ha="center", color=MODEL_COLORS.get(model, f"C{i}"),
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(probe_labels, fontsize=10)
    ax.set_ylabel("SVHN Probe Accuracy", fontsize=12)
    ax.set_title("Block 11 Dissection: Attention vs MLP Contribution", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.tight_layout()
    path = output_dir / "block11_subcomponent_probes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")



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
        description="Mechanistic analysis of block 11 sub-components"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer samples for faster iteration",
    )
    parser.add_argument("--n-train", type=int, default=5000, help="Training samples")
    parser.add_argument("--n-test", type=int, default=5000, help="Test samples")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_train = 1000 if args.quick else args.n_train
    n_test = 1000 if args.quick else args.n_test

    # =================================================================== #
    # Phase 1: Load models
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 1: Loading models")
    logger.info("=" * 70)

    pretrained = create_encoder(MODEL_NAME)
    preprocess = pretrained.val_preprocess
    print_memory("after pretrained")

    svhn_ft = load_finetuned_encoder("SVHN", MODEL_NAME)
    print_memory("after svhn_ft")

    merged_8 = create_merged_model(N8_DATASETS, MODEL_NAME)
    print_memory("after merged_8")

    models = {
        "pretrained": pretrained,
        "svhn_ft": svhn_ft,
        "merged_8": merged_8,
    }

    # =================================================================== #
    # Phase 2: Load SVHN data (train + test, separate splits)
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 2: Loading SVHN data")
    logger.info("=" * 70)

    train_imgs, train_labels, test_imgs, test_labels = load_svhn_data(
        preprocess, n_train=n_train, n_test=n_test
    )

    # =================================================================== #
    # Phase 3: Extract block-11 sub-component activations
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 3: Extracting block 11 sub-component activations")
    logger.info("=" * 70)

    train_activations = {}
    test_activations = {}

    for model_name, model in models.items():
        logger.info(f"Extracting activations: {model_name}")
        train_activations[model_name] = extract_block11_activations(
            model, train_imgs, device, batch_size=BATCH_SIZE
        )
        test_activations[model_name] = extract_block11_activations(
            model, test_imgs, device, batch_size=BATCH_SIZE
        )
        # Log shapes for sanity check
        for key, tensor in train_activations[model_name].items():
            logger.info(f"  {key}: {tensor.shape}")
        print_memory(f"after {model_name}")

    # =================================================================== #
    # Phase 4: Train and evaluate linear probes
    # =================================================================== #
    logger.info("=" * 70)
    logger.info("PHASE 4: Training linear probes")
    logger.info("=" * 70)

    all_results = {}
    for model_name in models:
        logger.info(f"\n--- {model_name} ---")
        all_results[model_name] = run_probes(
            train_activations[model_name],
            test_activations[model_name],
            train_labels,
            test_labels,
            model_name,
        )

    # =================================================================== #
    # Phase 5: Print summary
    # =================================================================== #
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info("\nSub-component probe accuracy:")
    logger.info(
        f"{'Model':<20} {'Block 10':>10} {'+ Attn':>10} {'+ MLP':>10} "
        f"{'Attn gain':>10} {'MLP gain':>10}"
    )
    logger.info("-" * 75)
    for model_name in models:
        r = all_results[model_name]
        b10 = r["block_10_out"]
        post_attn = r["block_11_post_attn"]
        b11 = r["block_11_out"]
        attn_gain = post_attn - b10
        mlp_gain = b11 - post_attn
        logger.info(
            f"{model_name:<20} {b10:>10.4f} {post_attn:>10.4f} {b11:>10.4f} "
            f"{attn_gain:>+10.4f} {mlp_gain:>+10.4f}"
        )

    # Key interpretation
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)
    m8 = all_results["merged_8"]
    attn_contribution = m8["block_11_post_attn"] - m8["block_10_out"]
    mlp_contribution = m8["block_11_out"] - m8["block_11_post_attn"]
    total_gain = m8["block_11_out"] - m8["block_10_out"]
    if total_gain > 0.01:
        attn_pct = attn_contribution / total_gain * 100
        mlp_pct = mlp_contribution / total_gain * 100
        logger.info(
            f"In the merged model, block 11 total gain: {total_gain:.1%}"
        )
        logger.info(
            f"  Attention contributes: {attn_contribution:.1%} ({attn_pct:.0f}% of total)"
        )
        logger.info(
            f"  MLP contributes: {mlp_contribution:.1%} ({mlp_pct:.0f}% of total)"
        )
        if attn_pct > 60:
            logger.info("  -> Attention is the dominant contributor.")
        elif mlp_pct > 60:
            logger.info("  -> MLP is the dominant contributor.")
        else:
            logger.info("  -> Both attention and MLP contribute substantially.")
    else:
        logger.info("Block 11 gain is negligible for the merged model.")

    # =================================================================== #
    # Phase 6: Generate plots
    # =================================================================== #
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: Generating plots")
    logger.info("=" * 70)

    plot_subcomponent_bars(all_results, OUTPUT_DIR)

    # =================================================================== #
    # Phase 7: Save results
    # =================================================================== #
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "n_train": n_train,
            "n_test": n_test,
            "block_analyzed": BLOCK_IDX,
            "merger": "InterferenceAwareMerger(mp_edge=True, isotropic=True)",
            "merge_tasks": N8_DATASETS,
            "probe_C": 0.316,
        },
        "results": all_results,
    }
    json_path = OUTPUT_DIR / "block11_mechanistic.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(output), f, indent=2)
    logger.info(f"\nResults saved to: {json_path}")
    logger.info(f"Plots saved to: {OUTPUT_DIR}")
    logger.info("Block 11 mechanistic analysis complete.")


if __name__ == "__main__":
    main()
