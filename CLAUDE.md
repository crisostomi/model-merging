# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Core shared library for weight-space model merging research. Provides utilities and merging methods for combining fine-tuned vision models (primarily CLIP-based) into a single multi-task model. Used as a dependency by sibling research projects in the parent monorepo (cycle-consistent-model-merging, mass, iso-merging, task_singular_vectors, etc.).

## Build & Run Commands

```sh
# Install dependencies
uv sync

# Run multi-task merging evaluation (main entry point)
uv run scripts/evaluate_multitask_merging.py

# Run with Hydra overrides
uv run scripts/evaluate_multitask_merging.py merger=isotropic benchmark=N8 nn.encoder.model_name=ViT-B-32

# Finetune a model
uv run scripts/finetune.py

# Download datasets
uv run scripts/download_datasets.py

# Run tests
uv run pytest src/tests/

# Linting
uv run ruff check src/
```

## Architecture

### Core Pipeline

The main workflow (`scripts/evaluate_multitask_merging.py`) follows this pattern:
1. Load a pretrained CLIP encoder (`ImageEncoder`) from HuggingFace (`crisostomi/{model_name}-base`)
2. Load fine-tuned model state dicts for each task (`crisostomi/{model_name}-{dataset_name}`)
3. A `Merger` combines the pretrained + fine-tuned models into a single merged encoder
4. Evaluate the merged encoder on each task's test set using task-specific `ClassificationHead`s

### Key Abstractions

- **`ImageEncoder`** ([model/encoder.py](src/model_merging/model/encoder.py)): Wraps an OpenCLIP vision model. Loads pretrained weights and optionally strips the text transformer. All merging operates on its `state_dict()`.

- **`TaskVectorBasedMerger`** ([merger/merger.py](src/model_merging/merger/merger.py)): Base class for all merging strategies. Subclasses implement `merge(base_model, finetuned_models) -> ImageEncoder`. Mergers are instantiated via Hydra (`instantiate(cfg.merger)`).

- **Task Vectors** ([merging/task_vectors.py](src/model_merging/merging/task_vectors.py)): The difference `finetuned_state_dict - pretrained_state_dict`. This is the fundamental unit of merging.

- **SVD Decomposition** ([merging/structured.py](src/model_merging/merging/structured.py)): Many mergers (Isotropic, TSV, Dual) first decompose task vectors via SVD with optional compression (`get_svd_dict`), then aggregate the low-rank components. SVD results are cached to disk at `svd_path`.

- **`ImageClassifier`** ([model/image_classifier.py](src/model_merging/model/image_classifier.py)): PyTorch Lightning module combining an encoder + classification head. Handles training, evaluation, and metric logging.

### Merger Implementations

Each merger lives in `src/model_merging/merger/` with a corresponding Hydra config in `conf/merger/`:

| Merger | Config | Description |
|--------|--------|-------------|
| `TaskArithmeticMerger` | `weight_avg.yaml` | Simple weighted sum of task vectors |
| `IsotropicMerger` | `isotropic.yaml` | SVD-based isotropic merging (replaces singular values with their mean) |
| `IsotropicCommonTaskSpecificMerger` | `iso-cts.yaml` | Splits into common + task-specific SVD subspaces |
| `TaskSingularVectorsMerger` | `tsv.yaml` | Concatenates low-rank SVD components across tasks |
| `DualMerger` | `dual.yaml` | Applies duality-based normalization via modular network composition |
| `DummyMerger` | `dummy.yaml` | Returns the base model unchanged (baseline) |

### Adding a New Merger

1. Create a class in `src/model_merging/merger/` extending `TaskVectorBasedMerger`
2. Implement `merge(self, base_model, finetuned_models) -> ImageEncoder`
3. Add a Hydra config in `conf/merger/` with `_target_: model_merging.merger.your_module.YourMerger`
4. Select it via `merger=your_config` in the CLI or `multitask.yaml`

## Configuration

Hydra-based configuration rooted at `conf/`:
- `multitask.yaml` - Main config; sets benchmark, merger, encoder, and train defaults
- `benchmark/` - Task sets (N2, N8, N14, N20 datasets)
- `merger/` - Merging method configs
- `nn/encoder/` - Model architectures (b32, b16, l14)
- `dataset/` - Per-dataset configs (loaded dynamically via `${dataset:DatasetName}` OmegaConf resolver)

### Environment Variables

Requires a `.env` file (loaded automatically via `python-dotenv` in `__init__.py`):
- `PROJECT_ROOT` - Set automatically from git root
- `MODELS_PATH` - Path to model checkpoints (referenced in config as `${oc.env:MODELS_PATH}`)

### Logging

- Weights & Biases (wandb) for experiment tracking
- `nn-template-core` (`NNLogger`, `NNTemplateCore`) for boilerplate logging setup
- Rich-formatted console logging configured in `__init__.py`

## Code Style

- Ruff for linting (line-length 88, target Python 3.8)
- Python 3.11.8 required
