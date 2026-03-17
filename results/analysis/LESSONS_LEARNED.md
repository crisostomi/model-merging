# AutoResearch: Issues & Lessons Learned

## Session 2026-03-17

### Issue 1: Model Loading Inefficiency
**Problem**: `load_model_from_hf()` in `io_utils.py` creates a full `ImageEncoder` (which initializes the OpenCLIP architecture from scratch, loading ~150MB of weights) just to immediately overwrite with `load_state_dict()`. For analysis scripts that only need state dicts, this is ~3x slower and uses much more memory than necessary.
**Fix applied**: In `scripts/analyze_task_vectors.py`, replaced with direct `hf_hub_download` + `torch.load` to get state dicts without creating ImageEncoder.
**Recommendation for autoresearch.md**: Add a helper function or note that for analysis-only scripts (no inference/evaluation), load state dicts directly:
```python
from huggingface_hub import hf_hub_download
def load_state_dict(model_name, dataset_name="base"):
    path = hf_hub_download(f"crisostomi/{model_name}-{dataset_name}", "pytorch_model.bin")
    return torch.load(path, map_location="cpu")
```

### Issue 2: Empty Scratch Storage
**Problem**: `$MODELS_PATH` on Leonardo scratch is empty — no cached SVD dicts exist. First SLURM run for any merger will compute SVDs from scratch (slow, several minutes).
**Impact**: The rank ablation merger intentionally skips SVD caching since it uses a different parameterization (max_rank vs compression_factor). Each SLURM job computes SVD independently.

### Issue 3: Login Node Memory
**Problem**: Loading 8 ViT-B-32 state dicts (~450MB each) + computing task vectors + performing SVD on all layers consumes ~8GB+ RAM.
**Fixes applied**:
- Added `del` calls in `load_task_vectors()` to free intermediate data
- Added `--quick` flag to analyze subset (3 tasks, 5 layers) for iteration
- Added intermediate saving of JSON results so partial results survive if script is killed

### Issue 4: SLURM Script Auto-Tagging
**Observation**: The SLURM script (`slurm/launch_merging_eval.slurm`) auto-extracts tags from Hydra overrides like `merger=rank_ablation`. However, additional overrides like `merger.max_rank_per_task=8` won't be captured as separate tags. This means differentiating rank ablation runs by their rank parameter requires checking wandb config, not just tags.
**Recommendation**: Consider adding explicit tag support for sub-parameters (e.g., `core.tags=[autoresearch,rank_ablation,rank8]`).

### Issue 5: Analysis Script Runs on Login Node
**Constraint**: The autoresearch.md says "GPU workloads must be submitted via SLURM". Analysis scripts that only need CPU (SVD, statistics, plotting) can run on the login node, but they're subject to memory limits and may get killed.
**Solution**: Keep analysis scripts memory-efficient, use `--quick` for iteration, run full analyses with `run_in_background` and generous timeouts.

### Issue 6: SLURM Auto-Tag Overrides Explicit Tags
**Problem**: The SLURM script auto-generates `core.tags=[autoresearch,merger,benchmark]` and appends it AFTER user-supplied Hydra overrides. Since Hydra uses last-wins semantics, any explicit `core.tags=` override in the user args gets overwritten by the auto-generated one.
**Impact**: For the rank ablation sweep, the `rank${k}` tag in explicit `core.tags` is lost. The rank parameter is only visible in wandb config (`merger.max_rank_per_task`).
**Recommendation**: Modify the SLURM script to merge tags rather than overwrite. Or use a separate Hydra override for additional tags (e.g., `+core.extra_tags=[rank8]`).

### Lesson 1: Intermediate Saves are Essential
Long-running analysis scripts should save results after each phase. The first run was killed mid-analysis and produced no output files. After adding `save_intermediate()` calls, partial results are preserved.

### Lesson 2: Report Findings to File, Not Just Chat
Lessons learned, issues, and findings should be written to files (not just chat output) since chat context may be compressed or lost. Use `results/analysis/LESSONS_LEARNED.md` for meta-issues and flywheel nodes for research findings.

## Session 2026-03-17 (Layer Alpha Analysis)

### Issue 7: All Layers vs 2D-Only Layers
**Problem**: The existing `analyze_task_vectors.py` script only analyzed 2D (matrix) layers for most analyses, since SVD only applies to matrices. However, for layer-wise alpha analysis, 1D layers (biases, layernorm scale/shift) also matter -- they are part of the merged model and their scaling behavior differs from weight matrices.
**Fix applied**: In `scripts/analyze_layer_alpha.py`, all analyses run on ALL layers (1D and 2D). The Fisher proxy analysis handles 1D tensors by using the Frobenius norm as a degenerate spectral norm (since a 1D tensor has a single "singular value" equal to its norm). This ensures per-layer alpha recommendations cover the full model.

### Issue 8: Cosine Similarity Scale for Agreement Heatmaps
**Observation**: Pairwise cosine similarities between task vectors are typically very small (close to 0) for most layer pairs, since task vectors from different fine-tuning tasks operate in largely orthogonal directions. Using a symmetric colormap (RdBu_r) centered at 0 with a range of [-0.3, 0.3] is more informative than [0, 1] -- it reveals the subtle differences between slight agreement and slight disagreement that matter for alpha selection.

### Issue 9: Interference Score Interpretation
**Observation**: The "retained fraction" metric (cos(tau_i, sum(tau_j))) can be misleading for layers where one task dominates. If task A has a much larger norm than tasks B-H, then cos(tau_A, sum(tau)) will be ~1.0 regardless of interference, because the sum is dominated by task A. The "magnitude ratio" metric (||sum|| / sum(||tau_i||)) is a cleaner measure of cancellation but loses per-task information. Both should be considered together.

### Lesson 3: Reuse Patterns from Existing Scripts
The existing `scripts/analyze_task_vectors.py` established good patterns: `load_state_dict` via `hf_hub_download`, `--quick` flag, `save_intermediate()`, `make_json_serializable()`, and `classify_layer()`. Reusing these patterns (rather than reinventing them) reduces bugs and keeps the codebase consistent. When adding a new analysis script, check what the existing ones already solved.

## Session 2026-03-17 (Rank Ablation & Infrastructure)

### Issue 10: Login Node OOM Kills Full Analysis
**Problem**: The full `analyze_task_vectors.py` with 8 tasks was killed by the OOM killer on the login node. Loading 8 state dicts (~430MB each × 2 for finetuned+pretrained + task vectors) uses ~7GB, plus SVD computation overhead exceeds login node memory.
**Fix applied**: Created `slurm/launch_analysis.slurm` for running analysis scripts on compute nodes with 32GB RAM. Usage: `sbatch slurm/launch_analysis.slurm scripts/analyze_task_vectors.py`
**Recommendation for autoresearch.md**: Add note that analysis scripts with all 8 N8 tasks should be submitted via SLURM, not run on login node. Only `--quick` mode (3 tasks) is safe for login node.

### Issue 11: HuggingFace Rate Limiting on Parallel SLURM Jobs
**Problem**: Submitting 7 SLURM jobs simultaneously caused HF rate limiting (429 Too Many Requests) for dataset downloads. Job 37870644 (rank=32) failed because all 7 jobs were downloading datasets concurrently.
**Fix**: Resubmitted the failed job after the other jobs completed. Datasets are cached after first download, so subsequent runs won't hit this issue.
**Recommendation**: When submitting many parallel jobs, stagger them or ensure datasets are pre-cached. The first job in a batch will populate the cache for subsequent ones.

### Issue 12: Scalar Tensors Break Analysis Scripts
**Problem**: `model.logit_scale` is a 0-dimensional (scalar) tensor. The Fisher proxy analysis tried to access `tensor.shape[0]` which fails on scalars.
**Fix applied**: Added `if tensor.dim() >= 1` guard before accessing `tensor.shape[0]`, defaulting to 1 for scalars.
**Recommendation**: When iterating over model state dict, always handle 0D tensors (scalars) explicitly. ViT-B-32 has at least `model.logit_scale` as a scalar.
