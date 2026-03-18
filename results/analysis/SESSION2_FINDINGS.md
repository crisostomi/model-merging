# Session 2 Findings: Interference Analysis and Task Interactions

## 1. Subspace Analysis: Where Does Interference Live?

**Method**: Decomposed N8 task vectors into a common subspace (top 80% of singular values of the summed task vectors) and a task-specific subspace (residual). Full analysis via SLURM job 37899245 across all 8 tasks and all layers.

### Energy Distribution

~80% of task vector energy lives in the common subspace, varying by task:

| Task | Common Energy |
|------|---------------|
| SVHN | 89% |
| Cars | 82% |
| DTD | 73% |

### Energy by Layer Type

| Layer Type | Common Energy |
|------------|---------------|
| Projection | 94% |
| Embedding | 87% |
| Attention | 83% |
| MLP FC | 78% |

### Similarity Structure

- **Task-specific cosine similarities are NEGATIVE** (-0.3 to -0.5): task-specific residuals are anti-correlated.
- **Common subspace cosine similarity ~0.03**: nearly orthogonal even within the common subspace.
- **Leakage ~0**: the orthogonal decomposition is numerically clean.

### Interpretation

The "common subspace" at fraction=0.8 is not truly "common" -- it uses 80% of the available dimensions and captures most energy by sheer size. The anti-correlation in task-specific residuals is a mathematical necessity: if P_common captures most of the sum, then the sum of residuals is approximately zero, forcing residuals to be anti-correlated. SVHN having the highest common energy (89%) means SVHN's directions dominate the sum -- the "common" subspace is biased toward high-magnitude tasks.

### Implication for Iso-CTS

Iso-CTS's common/task-specific split at fraction=0.8 is not doing what we think. A more informative decomposition would use alignment between task vectors (positive cosine similarity) to identify truly shared directions, rather than top singular values of the sum.

---

## 2. Leave-One-Out Task Interaction Matrix

**Method**: 8 SLURM jobs (37899255--37899267), each merging 7 of 8 tasks with MP-edge + isotropic. Per-task accuracies compared to the all-8 baseline (job 37874516).

### Harm Scores (average accuracy delta on remaining tasks when a task is removed)

```
     EuroSAT: +0.0125  (MOST HARMFUL)
      SUN397: +0.0106
         DTD: +0.0089
    RESISC45: +0.0062
       GTSRB: +0.0052
        Cars: +0.0021
        SVHN: +0.0000  (NOT HARMFUL)
       MNIST: -0.0001  (SLIGHTLY HELPFUL)
```

A positive score means other tasks improve when that task is removed, i.e., the task is harmful to the merged model.

### Top Pairwise Interactions: Beneficial Removals

| Removing | Helps | Delta |
|----------|-------|-------|
| EuroSAT | SVHN | +3.18 pts |
| SUN397 | DTD | +2.72 pts |
| EuroSAT | RESISC45 | +2.50 pts |
| DTD | SVHN | +1.78 pts |
| RESISC45 | SVHN | +1.59 pts |

### Top Pairwise Interactions: Harmful Removals

| Removing | Hurts | Delta |
|----------|-------|-------|
| MNIST | SVHN | -1.12 pts |
| RESISC45 | Cars | -0.93 pts |
| GTSRB | SVHN | -0.88 pts |

### Key Insight: SVHN Is a Victim, Not a Perpetrator

SVHN's poor accuracy in the merged model (77.66%) is caused by other tasks interfering with it -- primarily EuroSAT (-3.18 pts), SUN397, DTD, and RESISC45. SVHN itself has near-zero average harm to others. MNIST helps SVHN (+1.12 pts), consistent with both being digit/number recognition tasks.

### Semantic Interpretation

- **EuroSAT** (satellite imagery) and **RESISC45** (remote sensing) occupy similar feature spaces and interfere with each other (+2.50 when EuroSAT removed) and with SVHN.
- **Scene-understanding tasks** (SUN397, DTD) dominate the merged model's feature directions, crowding out SVHN's number-recognition features.
- The **beneficial MNIST-to-SVHN interaction** confirms that semantically similar tasks help each other in the merged model.

---

## 3. Task Vector Magnitude Normalization

**Method**: Added `normalize_task_vectors=true` to InterferenceAwareMerger (SLURM job 37899268). This normalizes all task vectors per-layer to the geometric mean Frobenius norm before SVD.

### Result

**Hurts performance.** Average accuracy drops from 93.11% to 92.68% (-0.43 pts). SVHN gets worse: -1.51 pts to 76.15%.

### Conclusion

Task vector magnitude differences are informative, not artifacts. Normalizing destroys useful scale information. The interference between tasks is **directional** (about which directions in weight space they use), not about magnitude.

---

## SLURM Job Summary

| Job ID | Type | Config | Result |
|--------|------|--------|--------|
| 37899245 | Analysis | Subspace decomposition (full 8 tasks) | Complete |
| 37899255 | LOO | N8 no SUN397, MP-edge+iso | 93.43% avg |
| 37899257 | LOO | N8 no Cars, MP-edge+iso | 93.17% avg |
| 37899258 | LOO | N8 no RESISC45, MP-edge+iso | 93.53% avg |
| 37899260 | LOO | N8 no EuroSAT, MP-edge+iso | 93.80% avg |
| 37899261 | LOO | N8 no SVHN, MP-edge+iso | 95.32% avg (7 tasks) |
| 37899263 | LOO | N8 no GTSRB, MP-edge+iso | 93.60% avg |
| 37899265 | LOO | N8 no MNIST, MP-edge+iso | 92.67% avg |
| 37899267 | LOO | N8 no DTD, MP-edge+iso | 93.89% avg |
| 37899268 | Norm | N8 MP-edge+iso+normalize | 92.68% avg |
| 37901172 | Per-task alpha | N8 MP-edge+iso, mild LOO weights | 92.91% avg |
| 37901173 | Per-task alpha | N8 MP-edge+iso, strong LOO weights | 92.60% avg |

## 4. Per-Task Alpha (LOO-Guided Weights)

**Method**: Added `alpha_per_task` to InterferenceAwareMerger. Each task's SVs are scaled by its alpha before TSV concatenation. For 1D params, weighted averaging. Tested mild (0.94-1.0) and strong (0.875-1.0) reweighting based on LOO harm scores.

**Result**: Both **FAIL**.

| Config | AVG | SVHN | Delta vs baseline |
|--------|-----|------|-------------------|
| Baseline (equal weights) | 93.11% | 77.66% | — |
| Mild per-task alpha | 92.91% | 77.18% | -0.20 / -0.48 |
| Strong per-task alpha | 92.60% | 76.58% | -0.51 / -1.08 |

**Why it fails**: TSV concatenation + Procrustes already orthogonalizes task components. Reducing a task's alpha just weakens its signal without changing the Procrustes rotation. The interference operates through a mechanism that per-task SV scaling cannot address — likely through the interaction of merged weights during forward-pass computation (representation/activation space), not through weight-space geometry.

## 5. Key Insight: Geometry ≠ Interference

MNIST-SVHN has HIGHER cosine similarity than EuroSAT-SVHN in every block (Block 11: 0.660 vs 0.426 for weight matrices), yet MNIST helps SVHN (+1.12 pts) while EuroSAT hurts (-3.18 pts). Additionally:
- EuroSAT has one of the LOWEST task vector norms but is the MOST harmful
- Magnitude normalization HURTS (confirmed separately)
- Per-task alpha scaling HURTS
- Tasks with higher merged accuracy tend to be more harmful (r=0.56)

**Conclusion**: Weight-space geometric measures (cosine similarity, SV magnitude, task vector norms, subspace overlap) CANNOT predict or fix interference direction. Current geometric-only methods (TSV, Iso-CTS) have a fundamental ceiling. The path forward requires representation-level analysis or functional measures of interference.
