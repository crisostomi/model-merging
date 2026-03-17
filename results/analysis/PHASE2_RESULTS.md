# Phase 2: Interference-Aware Merging Results

## Summary

The InterferenceAwareMerger combines per-layer-type SVD rank allocation with isotropic scaling, achieving **92.79% normalized accuracy** on N8/ViT-B-32/val — beating TSV (92.57%) by +0.22 pts.

## Full Results Table (N8/ViT-B-32/val)

| Method | Norm Acc (%) | Total SVs per layer | Notes |
|--------|-------------|--------------------:|-------|
| **Interference-aware + isotropic** | **92.79** | 32 MLP/attn, 16 proj, 8 emb | **Best result** |
| MP-edge adaptive | 92.63 | ~50-100 (adaptive) | Parameter-free rank |
| Rank ablation (k=64) | 92.58 | 64 uniform | |
| TSV baseline | 92.57 | ~96 uniform | cf=8 |
| Interference-aware (rank64) | 92.40 | 64 MLP/attn, 16 proj, 8 emb | |
| Iso-CTS baseline | 91.00 | | |
| Interference-aware (alpha=0.8) | 90.78 | 32 MLP/attn, 16 proj, 8 emb | Lower alpha hurts |
| Rank ablation (k=32) | 89.97 | 32 uniform | |
| Interference-aware (default) | 89.67 | 32 MLP/attn, 16 proj, 8 emb | No isotropic |

## Key Findings

### 1. Isotropic scaling is the critical ingredient
- rank32 without isotropic: 89.67%
- rank32 WITH isotropic: 92.79%
- The isotropic step (replacing all SVs with their mean after concatenation) provides a +3.12 pt boost
- This makes intuitive sense: isotropic scaling equalizes energy across directions, preventing dominant SVs from drowning out task-specific but numerically smaller components

### 2. Per-layer-type rank allocation enables efficient compression
- With isotropic scaling, rank32 per type (92.79%) EXCEEDS uniform rank64 (92.58%) and uniform rank96/TSV (92.57%)
- This means we can use 50-67% fewer SVs by allocating rank according to layer importance
- The key is giving less rank to layers that contribute less to the merged model (embeddings, projections)

### 3. Alpha reduction is counterproductive with SVD concatenation
- Reducing alpha from 1.0 to 0.8 for MLP/attention drops performance by 2 pts (92.79 → 90.78)
- With TSV-style concatenation + Procrustes orthogonalization, the task-specific directions are placed in orthogonal subspaces, so they DON'T conflict
- Full alpha is correct because the signal is already separated — reducing it just weakens ALL signals

### 4. MP-edge adaptive rank is principled and competitive
- Using the Marchenko-Pastur noise edge as a rank cutoff achieves 92.63% — better than TSV
- This is completely parameter-free: no rank_per_type or compression_factor needed
- The MP edge naturally gives more rank to layers with more signal (MLP layers with more SVs above noise)

## Next Steps

1. **Validate on N14, N20**: Does the advantage hold with more tasks?
2. **Combine MP-edge + isotropic**: Use MP edge for rank AND isotropic for SV scaling
3. **Test on ViT-B-16, ViT-L-14**: Generalize across architectures
4. **Investigate why isotropic + low rank works**: Is it because isotropic acts as regularization, preventing overfitting to dominant SVs?

## SLURM Job IDs
- 37873378: default (rank32, alpha1.0) → 89.67%
- 37873380: isotropic (rank32) → 92.79%
- 37873381: rank64 → 92.40%
- 37873383: alpha=0.8 → 90.78%
- 37873729: MP-edge → 92.63%
