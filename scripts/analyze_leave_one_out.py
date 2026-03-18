#!/usr/bin/env python3
"""
Analyze leave-one-out merging experiments.

Computes the N x N task interaction matrix showing how each task's
presence affects every other task's accuracy in the merged model.

For each leave-one-out experiment (excluded task j):
  delta_ij = normalized_acc(task i, without j) - normalized_acc_baseline(task i)
  Positive delta => task i improves when task j is removed (j was hurting i)
  Negative delta => task i worsens when task j is removed (j was helping i)

Usage:
    uv run python scripts/analyze_leave_one_out.py
    uv run python scripts/analyze_leave_one_out.py --job-ids 37899255 37899257 ...
    uv run python scripts/analyze_leave_one_out.py --metric acc   # use raw accuracy instead
"""

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLURM_DIR = PROJECT_ROOT / "slurm"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"

N8_DATASETS = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]

# Baseline results (all 8 tasks, job 37874516, interference_aware + MP-edge + isotropic)
BASELINE_NORM_ACC = {
    "SUN397": 0.9830, "Cars": 0.9412, "RESISC45": 0.9444,
    "EuroSAT": 0.9701, "SVHN": 0.7766, "GTSRB": 0.9332,
    "MNIST": 0.9613, "DTD": 0.9388,
}
BASELINE_ACC = {
    "SUN397": 0.7360, "Cars": 0.7512, "RESISC45": 0.9000,
    "EuroSAT": 0.9630, "SVHN": 0.7507, "GTSRB": 0.9232,
    "MNIST": 0.9560, "DTD": 0.7340,
}
BASELINE_AVG_NORM_ACC = 0.9311
BASELINE_AVG_ACC = 0.8393

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def extract_excluded_task(filepath: Path) -> str | None:
    """Extract the excluded task name from Hydra overrides in a SLURM .out file.

    Looks for a line like:
        Hydra overrides: ... benchmark=N8_no_SUN397 ...
    """
    try:
        with open(filepath, "r", errors="replace") as f:
            for line in f:
                m = re.search(r"benchmark=N8_no_(\w+)", line)
                if m:
                    return m.group(1)
    except OSError:
        return None
    return None


def parse_results_dict(filepath: Path) -> dict | None:
    """Parse the results dict logged by Rich at the end of a SLURM .out file.

    The dict is logged via `pylogger.info(results)` which produces Rich-formatted
    output that wraps long lines and adds leading whitespace. The dict starts with
    a line containing "{'<TaskName>':" and ends with a line ending in "}".

    Strategy:
    1. Read the file from the end looking for the results dict.
    2. Strip Rich formatting (timestamps, log levels, alignment whitespace).
    3. Reassemble into a single string and parse with ast.literal_eval.
    """
    try:
        with open(filepath, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    if not lines:
        return None

    # Find the line range containing the results dict.
    # The dict is logged as a single INFO message spanning many lines.
    # Look for a line containing "{'SUN397':" or "{'Cars':" etc. (first task key)
    # The Rich formatter prefixes lines with timestamps like "[14:41:34] INFO"
    # or continuation lines with just whitespace.

    dict_start = None
    dict_end = None

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        # Look for the start of the results dict (contains a task name as first key)
        if re.search(r"\{['\"](" + "|".join(N8_DATASETS) + r")['\"]", line):
            dict_start = i
            break

    if dict_start is None:
        return None

    # Find the end: scan forward from start until we find the closing brace
    brace_depth = 0
    for i in range(dict_start, len(lines)):
        line = lines[i]
        # Count braces (simplified - not inside strings, but good enough for this format)
        for ch in line:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
        if brace_depth <= 0 and i > dict_start:
            dict_end = i
            break
        elif brace_depth == 0 and i == dict_start:
            dict_end = i
            break

    if dict_end is None:
        # Dict might not be complete
        return None

    # Extract and clean the lines, then smart-join respecting string boundaries.
    # Rich wraps long lines mid-token or mid-string-literal, e.g.:
    #   'normalized_acc/test/GTSRB
    #   ': 0.933...
    # Naive space-join produces 'normalized_acc/test/GTSRB ': ... which breaks.
    # Strategy: track whether we're inside a string literal and concatenate
    # continuation lines without a space when the previous line ended mid-string.
    raw_lines = lines[dict_start : dict_end + 1]
    cleaned_parts = []
    for line in raw_lines:
        stripped = line.rstrip()

        # Remove timestamp + log level prefix
        stripped = re.sub(r"^\s*\[\d+:\d+:\d+\]\s+(INFO|WARNING|DEBUG|ERROR)\s+", "", stripped)

        # Remove source file references at the end (e.g., "evaluate_multitask_merging.py:172")
        stripped = re.sub(r"\s+\S+\.py:\d+\s*$", "", stripped)

        # Remove leading whitespace that's just Rich alignment
        stripped = stripped.strip()

        if stripped:
            cleaned_parts.append(stripped)

    # Smart-join: if one part ends inside a string literal (odd number of quotes
    # so far), concatenate without space to avoid breaking the literal.
    dict_str = cleaned_parts[0] if cleaned_parts else ""
    for part in cleaned_parts[1:]:
        # Count unescaped single quotes in dict_str so far
        n_quotes = dict_str.count("'") - dict_str.count("\\'")
        if n_quotes % 2 == 1:
            # We're inside a string literal -- concatenate directly
            dict_str += part
        else:
            dict_str += " " + part

    # Additional fixups for Rich line-wrapping within identifiers that may
    # still have spurious spaces (e.g., "SUN39 7" from a wrap point).
    for ds in N8_DATASETS:
        for split_pos in range(2, len(ds)):
            broken = ds[:split_pos] + " " + ds[split_pos:]
            if broken in dict_str:
                dict_str = dict_str.replace(broken, ds)

    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError) as e:
        logger.warning("Failed to parse results dict from %s: %s", filepath, e)
        logger.debug("Dict string was: %s", dict_str[:500])

    return None



def extract_per_task_metrics(results_dict: dict, metric: str = "normalized_acc") -> dict:
    """Extract per-task metric values from a parsed results dict.

    Args:
        results_dict: Parsed results dict from SLURM output.
        metric: Either "normalized_acc" or "acc".

    Returns:
        Dict mapping task name -> metric value.
    """
    metrics = {}
    for task_name, task_results_list in results_dict.items():
        if task_name == "avg":
            continue
        if not task_results_list or not isinstance(task_results_list, list):
            continue
        task_results = task_results_list[0]
        key = f"{metric}/test/{task_name}"
        if key in task_results:
            metrics[task_name] = task_results[key]
        else:
            logger.warning("Key %s not found in results for task %s", key, task_name)
    return metrics


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_loo_files(slurm_dir: Path) -> dict[str, Path]:
    """Discover leave-one-out SLURM output files.

    Returns dict mapping excluded_task -> most recent .out file path.
    """
    candidates = defaultdict(list)

    for out_file in sorted(slurm_dir.glob("merging-eval-*.out")):
        excluded = extract_excluded_task(out_file)
        if excluded and excluded in N8_DATASETS:
            # Extract job ID for ordering (higher = more recent)
            m = re.search(r"merging-eval-(\d+)\.out", out_file.name)
            job_id = int(m.group(1)) if m else 0
            candidates[excluded].append((job_id, out_file))

    # Pick the most recent (highest job ID) for each excluded task
    result = {}
    for excluded, files in candidates.items():
        files.sort(key=lambda x: x[0], reverse=True)
        result[excluded] = files[0][1]

    return result


def load_loo_results(
    slurm_dir: Path,
    job_ids: list[int] | None = None,
    metric: str = "normalized_acc",
) -> dict[str, dict[str, float]]:
    """Load leave-one-out experiment results.

    Args:
        slurm_dir: Path to SLURM output directory.
        job_ids: Optional list of specific job IDs to use.
        metric: "normalized_acc" or "acc".

    Returns:
        Dict mapping excluded_task -> {remaining_task: metric_value}.
    """
    if job_ids:
        file_map = {}
        for jid in job_ids:
            fpath = slurm_dir / f"merging-eval-{jid}.out"
            if not fpath.exists():
                logger.warning("Job file not found: %s", fpath)
                continue
            excluded = extract_excluded_task(fpath)
            if excluded:
                file_map[excluded] = fpath
            else:
                logger.warning("Could not determine excluded task from %s", fpath)
    else:
        file_map = discover_loo_files(slurm_dir)

    logger.info("Found %d leave-one-out files:", len(file_map))
    for excluded, fpath in sorted(file_map.items()):
        logger.info("  Excluded %-10s -> %s", excluded, fpath.name)

    loo_results = {}
    for excluded, fpath in file_map.items():
        results_dict = parse_results_dict(fpath)
        if results_dict is None:
            logger.warning(
                "Could not parse results from %s (job may still be running)", fpath.name
            )
            continue
        per_task = extract_per_task_metrics(results_dict, metric=metric)
        if not per_task:
            logger.warning("No per-task metrics found in %s", fpath.name)
            continue
        loo_results[excluded] = per_task
        logger.info(
            "  Parsed %-10s: %d tasks, avg=%.4f",
            excluded,
            len(per_task),
            np.mean(list(per_task.values())),
        )

    return loo_results


# --------------------------------------------------------------------------- #
# Interaction matrix
# --------------------------------------------------------------------------- #

def build_interaction_matrix(
    loo_results: dict[str, dict[str, float]],
    baseline: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Build the N x N task interaction matrix.

    matrix[i, j] = delta when task j is excluded, for task i
                  = loo_acc(task i, without j) - baseline_acc(task i)

    Positive: task i improved when j was removed (j was hurting i)
    Negative: task i worsened when j was removed (j was helping i)

    The diagonal is NaN (task j is not evaluated when it's excluded).
    """
    tasks = N8_DATASETS
    n = len(tasks)
    matrix = np.full((n, n), np.nan)

    for j, excluded_task in enumerate(tasks):
        if excluded_task not in loo_results:
            continue
        per_task = loo_results[excluded_task]
        for i, eval_task in enumerate(tasks):
            if eval_task == excluded_task:
                # Task j is excluded, so no accuracy for it
                continue
            if eval_task in per_task and eval_task in baseline:
                matrix[i, j] = per_task[eval_task] - baseline[eval_task]

    return matrix, tasks


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def plot_interaction_heatmap(
    matrix: np.ndarray,
    tasks: list[str],
    output_path: Path,
    metric_name: str = "normalized_acc",
) -> None:
    """Plot and save the interaction matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Mask the diagonal (NaN values)
    masked = np.ma.array(matrix, mask=np.isnan(matrix))

    # Determine symmetric color limits
    vmax = np.nanmax(np.abs(matrix))
    if vmax == 0:
        vmax = 0.01

    im = ax.imshow(
        masked,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Delta {metric_name} (LOO - baseline)", fontsize=11)

    # Labels
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=10)

    ax.set_xlabel("Excluded task (j)", fontsize=12)
    ax.set_ylabel("Evaluated task (i)", fontsize=12)
    ax.set_title(
        "Leave-One-Out Interaction Matrix\n"
        "(positive = task i improves when j removed, i.e. j hurts i)",
        fontsize=13,
    )

    # Annotate cells with values
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(
                    j, i, f"{val:+.3f}",
                    ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold",
                )
            elif i == j:
                ax.text(
                    j, i, "---",
                    ha="center", va="center",
                    fontsize=8, color="gray",
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", output_path)


# --------------------------------------------------------------------------- #
# Summary analysis
# --------------------------------------------------------------------------- #

def print_summary(matrix: np.ndarray, tasks: list[str]) -> None:
    """Print analysis summary to stdout."""
    n = len(tasks)

    print("\n" + "=" * 72)
    print("LEAVE-ONE-OUT INTERACTION ANALYSIS")
    print("=" * 72)

    # --- Interaction matrix table ---
    print(f"\n{'Interaction Matrix (delta normalized_acc when column task is excluded)':^72}")
    print("-" * 72)

    label = "Eval \\ Excl"
    header = f"{label:>12}" + "".join(f"{t:>10}" for t in tasks)
    print(header)
    print("-" * len(header))

    for i, task_i in enumerate(tasks):
        row = f"{task_i:>12}"
        for j, task_j in enumerate(tasks):
            if np.isnan(matrix[i, j]):
                row += f"{'---':>10}"
            else:
                row += f"{matrix[i, j]:>+10.4f}"
        print(row)

    # --- Column means: avg effect of removing task j on others ---
    print("\n" + "-" * 72)
    print("Average effect of removing each task (column means, excluding diagonal):")
    print("-" * 72)

    col_means = []
    for j in range(n):
        col_vals = [matrix[i, j] for i in range(n) if not np.isnan(matrix[i, j])]
        mean_val = np.mean(col_vals) if col_vals else np.nan
        col_means.append(mean_val)
        direction = "HARMFUL" if mean_val > 0 else "helpful"
        print(f"  Remove {tasks[j]:>10}: avg delta = {mean_val:+.4f}  ({direction} to others)")

    # --- Most harmful task ---
    most_harmful_idx = np.nanargmax(col_means)
    most_helpful_idx = np.nanargmin(col_means)
    print(f"\n  >> Most HARMFUL task (others improve most when removed): "
          f"{tasks[most_harmful_idx]} (avg delta = {col_means[most_harmful_idx]:+.4f})")
    print(f"  >> Most HELPFUL task (others worsen most when removed):  "
          f"{tasks[most_helpful_idx]} (avg delta = {col_means[most_helpful_idx]:+.4f})")

    # --- Row means: how much each task is affected by removal of others ---
    print("\n" + "-" * 72)
    print("Average sensitivity of each task (row means, excluding diagonal):")
    print("-" * 72)

    row_means = []
    for i in range(n):
        row_vals = [matrix[i, j] for j in range(n) if not np.isnan(matrix[i, j])]
        mean_val = np.mean(row_vals) if row_vals else np.nan
        row_means.append(mean_val)
        direction = "generally benefits" if mean_val > 0 else "generally suffers"
        print(f"  {tasks[i]:>10}: avg delta = {mean_val:+.4f}  ({direction} from removals)")

    # --- Strongest pairwise interactions ---
    print("\n" + "-" * 72)
    print("Strongest pairwise interactions:")
    print("-" * 72)

    interactions = []
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                interactions.append((matrix[i, j], i, j))

    interactions.sort(key=lambda x: x[0], reverse=True)

    print("\n  Top 5 POSITIVE deltas (task i improves when task j removed -- j hurts i):")
    for val, i, j in interactions[:5]:
        print(f"    {tasks[i]:>10} improves by {val:+.4f} when {tasks[j]:<10} is removed")

    print("\n  Top 5 NEGATIVE deltas (task i worsens when task j removed -- j helps i):")
    for val, i, j in interactions[-5:]:
        print(f"    {tasks[i]:>10} worsens by {val:+.4f} when {tasks[j]:<10} is removed")

    # --- LOO average accuracy ---
    print("\n" + "-" * 72)
    print("LOO average accuracy (over remaining 7 tasks):")
    print("-" * 72)

    for j in range(n):
        col_vals = [matrix[i, j] for i in range(n) if not np.isnan(matrix[i, j])]
        if col_vals:
            # Reconstruct avg from baseline + delta
            baseline_vals = [BASELINE_NORM_ACC[tasks[i]] for i in range(n) if not np.isnan(matrix[i, j])]
            loo_avg = np.mean([b + d for b, d in zip(baseline_vals, col_vals)])
            baseline_avg = np.mean(baseline_vals)
            print(f"  Without {tasks[j]:>10}: avg norm_acc = {loo_avg:.4f}  "
                  f"(baseline subset avg = {baseline_avg:.4f}, delta = {loo_avg - baseline_avg:+.4f})")

    print("\n" + "=" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Analyze leave-one-out merging experiments")
    parser.add_argument(
        "--job-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific SLURM job IDs to use (default: auto-discover)",
    )
    parser.add_argument(
        "--metric",
        choices=["normalized_acc", "acc"],
        default="normalized_acc",
        help="Metric to use for the interaction matrix (default: normalized_acc)",
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=SLURM_DIR,
        help="Directory containing SLURM .out files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for output files",
    )
    args = parser.parse_args()

    # Select baseline based on metric
    if args.metric == "normalized_acc":
        baseline = BASELINE_NORM_ACC
    else:
        baseline = BASELINE_ACC

    # Load results
    loo_results = load_loo_results(
        slurm_dir=args.slurm_dir,
        job_ids=args.job_ids,
        metric=args.metric,
    )

    if not loo_results:
        logger.error(
            "No leave-one-out results found! "
            "Jobs may still be running, or output files may not exist yet.\n"
            "Looked in: %s",
            args.slurm_dir,
        )
        print("\nNo results to analyze. Provide --job-ids or wait for jobs to finish.")
        return

    logger.info("Successfully parsed %d / %d leave-one-out experiments", len(loo_results), len(N8_DATASETS))
    missing = set(N8_DATASETS) - set(loo_results.keys())
    if missing:
        logger.warning("Missing results for excluded tasks: %s", sorted(missing))

    # Build interaction matrix
    matrix, tasks = build_interaction_matrix(loo_results, baseline)

    # Print summary
    print_summary(matrix, tasks)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = args.output_dir / "leave_one_out_results.json"
    json_data = {
        "metric": args.metric,
        "baseline": baseline,
        "tasks": tasks,
        "interaction_matrix": [
            [None if np.isnan(v) else round(v, 6) for v in row]
            for row in matrix
        ],
        "loo_results": {
            excluded: {task: round(val, 6) for task, val in per_task.items()}
            for excluded, per_task in loo_results.items()
        },
        "column_means": {},
        "row_means": {},
    }

    # Add summary stats
    for j, task in enumerate(tasks):
        col_vals = [matrix[i, j] for i in range(len(tasks)) if not np.isnan(matrix[i, j])]
        json_data["column_means"][task] = round(np.mean(col_vals), 6) if col_vals else None

    for i, task in enumerate(tasks):
        row_vals = [matrix[i, j] for j in range(len(tasks)) if not np.isnan(matrix[i, j])]
        json_data["row_means"][task] = round(np.mean(row_vals), 6) if row_vals else None

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info("JSON results saved to %s", json_path)

    # Save heatmap
    heatmap_path = args.output_dir / "leave_one_out_interaction.png"
    plot_interaction_heatmap(matrix, tasks, heatmap_path, metric_name=args.metric)

    print(f"Output files:")
    print(f"  JSON:    {json_path}")
    print(f"  Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
