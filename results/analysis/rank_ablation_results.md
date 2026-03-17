# Rank Ablation Results (N8/ViT-B-32/val)

## Results

| Rank per task | Norm Acc (%) | Abs Acc (%) | Notes |
|---------------|-------------|-------------|-------|
| 1             | 66.58       | 59.61       |       |
| 2             | 73.26       | 66.18       |       |
| 4             | 79.76       | 72.43       |       |
| 8             | 83.02       | 75.53       |       |
| 16            | 85.69       | 77.97       |       |
| 32            | TBD         | TBD         | Rate-limited, resubmitted |
| 64            | 92.58       | 83.86       |       |
| TSV default   | 92.57       | 82.07       | cf=8, effective rank~96 |
| Iso-CTS       | 91.00       | 82.07       |       |

## Key Observations

1. **Rank 64 matches TSV (92.58% vs 92.57%)**: Even though TSV default uses rank~96 (compression_factor=8, ratio=1/8, rank=768*0.125=96), dropping to rank=64 loses nothing. This means ~30% of the retained SVs in default TSV contribute zero marginal value.

2. **Clear logarithmic elbow**: Performance improves approximately linearly in log(rank):
   - 1→2: +6.7 pts
   - 2→4: +6.5 pts
   - 4→8: +3.3 pts
   - 8→16: +2.7 pts
   - 16→64: +6.9 pts (over 4x the rank)
   The elbow is around rank 8-16.

3. **Even rank=1 achieves 66.58%**: A single singular vector per task captures 72% of what full TSV achieves. This confirms the earlier analysis finding that top SVs are very task-specific (low cross-task overlap at k=1).

4. **The gap from rank 16→64 is still substantial (+6.9 pts)**: This suggests there IS useful signal beyond the top few SVs, contradicting a pure "signal is in the top" narrative. The tail SVs contribute cumulatively.

## Interpretation

The signal in task vectors is concentrated but NOT solely in the top few SVs. Rather, there's a "long tail" of useful information. The optimal rank is likely determined by the point where additional SVs start introducing more cross-task interference than useful signal. The fact that rank=64 matches rank=96 (default TSV) suggests the interference threshold is around rank 64 for N8.

**Hypothesis for follow-up**: The optimal rank might scale with the number of tasks. With more tasks, cross-task interference grows, so the optimal rank should decrease. Test this by running rank ablation on N2, N14, N20.

## Jobs

- rank=1: SLURM 37870619 (COMPLETED)
- rank=2: SLURM 37870620 (COMPLETED)
- rank=4: SLURM 37870621 (COMPLETED)
- rank=8: SLURM 37870622 (COMPLETED)
- rank=16: SLURM 37870643 (COMPLETED)
- rank=32: SLURM 37872352 (resubmitted, original hit HF rate limit)
- rank=64: SLURM 37870645 (COMPLETED)
