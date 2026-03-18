"""Fetch merging evaluation results from Weights & Biases.

Usage:
    uv run python scripts/fetch_wandb_results.py                          # all autoresearch runs
    uv run python scripts/fetch_wandb_results.py --tags autoresearch tsv  # filter by multiple tags
    uv run python scripts/fetch_wandb_results.py --merger iso-cts         # filter by merger
    uv run python scripts/fetch_wandb_results.py --benchmark N8           # filter by benchmark
    uv run python scripts/fetch_wandb_results.py --json                   # output as JSON
"""

import argparse
import json
import sys
from typing import Optional

import wandb

ENTITY = "gladia"
PROJECT = "dual-merging"

# Map merger _target_ to short name
MERGER_TARGET_MAP = {
    "IsotropicCommonTaskSpecific": "iso-cts",
    "IsotropicMerger": "isotropic",
    "TaskSingularVectors": "tsv",
    "TaskArithmetic": "weight_avg",
    "DualMerger": "dual",
    "DummyMerger": "dummy",
}


def get_merger_name(config: dict) -> str:
    target = config.get("merger/_target_", "")
    if not target:
        merger_cfg = config.get("merger", {})
        if isinstance(merger_cfg, dict):
            target = merger_cfg.get("_target_", "")
    for key, name in MERGER_TARGET_MAP.items():
        if key in target:
            return name
    return target.split(".")[-1] if target else "unknown"


def get_benchmark_name(run) -> str:
    tags = run.tags
    for t in tags:
        if t in ("N2", "N8", "N14", "N20", "hard"):
            return t
    num_tasks = run.config.get("num_tasks")
    if num_tasks:
        return f"N{num_tasks}"
    return "?"


def fetch_wandb_results(
    tags: Optional[list[str]] = None,
    merger: Optional[str] = None,
    benchmark: Optional[str] = None,
    entity: str = ENTITY,
    project: str = PROJECT,
) -> list[dict]:
    """Fetch evaluation results from wandb.

    Args:
        tags: Filter runs by ALL of these tags (AND logic).
        merger: Filter by merger short name (e.g. 'tsv', 'iso-cts').
        benchmark: Filter by benchmark name (e.g. 'N8', 'N20').
        entity: wandb entity.
        project: wandb project.

    Returns:
        List of result dicts with keys: merger, benchmark, num_tasks,
        acc, norm_acc, run_id, run_name, tags.
    """
    api = wandb.Api()

    filters = {}
    if tags:
        filters["$and"] = [{"tags": {"$eq": t}} for t in tags]

    runs = api.runs(f"{entity}/{project}", filters=filters, order="-created_at")

    results = []
    for r in runs:
        h = r.history(keys=["acc/test/avg", "normalized_acc/test/avg"])
        if h.empty:
            continue

        acc = h.iloc[-1].get("acc/test/avg", 0)
        norm_acc = h.iloc[-1].get("normalized_acc/test/avg", 0)
        m = get_merger_name(r.config)
        b = get_benchmark_name(r)
        n = r.config.get("num_tasks", "?")

        if merger and m != merger:
            continue
        if benchmark and b != benchmark:
            continue

        results.append({
            "merger": m,
            "benchmark": b,
            "num_tasks": n,
            "acc": round(acc * 100, 2),
            "norm_acc": round(norm_acc * 100, 2),
            "run_id": r.id,
            "run_name": r.name,
            "tags": r.tags,
        })

    return results


def print_table(results: list[dict]):
    if not results:
        print("No results found.")
        return

    print(f"{'Merger':<12} {'Bench':<6} {'Tasks':<6} {'Acc%':<8} {'NormAcc%':<10} {'Run'}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: (x["merger"], x.get("num_tasks", 0))):
        print(
            f"{r['merger']:<12} {r['benchmark']:<6} {r['num_tasks']:<6} "
            f"{r['acc']:<8.2f} {r['norm_acc']:<10.2f} {r['run_name']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Fetch merging results from wandb")
    parser.add_argument("--tags", nargs="+", default=["autoresearch"], help="Filter by tags (default: autoresearch)")
    parser.add_argument("--merger", type=str, default=None, help="Filter by merger name")
    parser.add_argument("--benchmark", type=str, default=None, help="Filter by benchmark")
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--project", type=str, default=PROJECT)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = fetch_wandb_results(
        tags=args.tags,
        merger=args.merger,
        benchmark=args.benchmark,
        entity=args.entity,
        project=args.project,
    )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print_table(results)


if __name__ == "__main__":
    main()
