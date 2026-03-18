"""
Interference-Aware Merger

A merger informed by Phase 1 analysis findings:
- Weight matrices have nearly orthogonal task vectors (cos_sim ~0.01)
  → Use SVD concatenation (like TSV) to preserve task-specific directions
- Non-weight params (LayerNorm, biases) have near-identical task vectors (cos_sim ~0.98)
  → Simple averaging works perfectly
- Different layer types benefit from different alpha scaling
- Adaptive rank allocation based on layer type

This merger combines the best aspects of TSV and Isotropic merging,
applying them selectively based on the structural properties of each layer.
"""

import copy
import logging

import torch
from tqdm import tqdm

from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    is_matrix,
    print_memory,
)

pylogger = logging.getLogger(__name__)


def classify_layer_type(layer_name: str) -> str:
    """Classify layer for rank/alpha allocation."""
    if "ln" in layer_name or "norm" in layer_name:
        return "layernorm"
    if "bias" in layer_name:
        return "bias"
    if "c_fc" in layer_name or "mlp" in layer_name:
        return "mlp"
    if "in_proj" in layer_name or "out_proj" in layer_name or "attn" in layer_name:
        return "attention"
    if "conv1" in layer_name:
        return "conv"
    if "proj" in layer_name:
        return "projection"
    if "embedding" in layer_name:
        return "embedding"
    return "other"


def compute_mp_edge(matrix: torch.Tensor) -> float:
    """Compute the Marchenko-Pastur upper edge for a matrix.

    The MP law predicts that eigenvalues of the sample covariance of a random
    matrix with i.i.d. entries of variance sigma^2 lie in
    [sigma^2*(1-sqrt(gamma))^2, sigma^2*(1+sqrt(gamma))^2] where gamma = m/n.

    Singular values above sqrt(lambda_plus * n) carry signal.
    """
    m, n = matrix.shape
    gamma = m / n if m >= n else n / m
    sigma2 = float(matrix.var())
    lambda_plus = sigma2 * (1 + gamma**0.5) ** 2
    # Convert eigenvalue threshold to singular value threshold
    sv_threshold = (lambda_plus * min(m, n)) ** 0.5
    return sv_threshold


@torch.no_grad()
def decompose_with_layer_ranks(
    task_dicts, rank_per_type: dict, default_rank: int, use_mp_edge: bool = False,
    mp_min_rank: int = 4, mp_max_rank: int = 128,
):
    """
    SVD decomposition with per-layer-type rank allocation.

    Args:
        task_dicts: {dataset: {layer_name: tensor}}
        rank_per_type: {layer_type: max_rank} e.g. {"mlp": 32, "attention": 16}
        default_rank: fallback rank for layer types not in rank_per_type
        use_mp_edge: if True, use Marchenko-Pastur edge to determine rank adaptively
        mp_min_rank: minimum rank when using MP edge
        mp_max_rank: maximum rank when using MP edge
    """
    svd_dict = {}

    for dataset, task_dict in tqdm(
        task_dicts.items(), desc="SVD decomposition (adaptive rank)"
    ):
        svd_dict[dataset] = {}

        for key, layer in task_dict.items():
            if is_matrix(layer):
                layer_type = classify_layer_type(key)
                U, S, V = torch.linalg.svd(layer.float(), full_matrices=False)

                if use_mp_edge:
                    # Adaptive rank: keep SVs above the MP noise edge
                    threshold = compute_mp_edge(layer.float())
                    k = int((S > threshold).sum().item())
                    k = max(mp_min_rank, min(mp_max_rank, k))
                else:
                    max_rank = rank_per_type.get(layer_type, default_rank)
                    k = min(max_rank, S.shape[0])

                svd_dict[dataset][key] = {
                    "u": U[:, :k].detach().cpu(),
                    "s": S[:k].detach().cpu(),
                    "v": V[:k, :].detach().cpu(),
                }
            else:
                svd_dict[dataset][key] = {"dim1": layer.detach().cpu()}

    return svd_dict


@torch.no_grad()
def aggregate_interference_aware(
    ref_state_dict,
    decomposed_task_vectors,
    alpha_per_type: dict,
    default_alpha: float = 1.0,
    use_isotropic: bool = False,
    device="cuda",
):
    """
    Aggregate decomposed task vectors with per-layer-type alpha scaling.

    For 2D layers: TSV-style concatenation + Procrustes, with optional isotropic scaling.
    For 1D layers: weighted averaging with layer-type-specific alpha.

    Args:
        ref_state_dict: pretrained model state dict (used as base)
        decomposed_task_vectors: SVD-decomposed task vectors
        alpha_per_type: {layer_type: alpha} for per-type scaling
        default_alpha: fallback alpha
        use_isotropic: if True, replace SVs with their mean after concatenation
        device: compute device
    """
    aggregated = ref_state_dict
    layer_names = list(aggregated.keys())
    datasets = list(decomposed_task_vectors.keys())

    for layer_name in tqdm(layer_names, desc="Aggregating (interference-aware)"):
        is_mat = aggregated[layer_name].dim() == 2
        layer_type = classify_layer_type(layer_name)
        alpha = alpha_per_type.get(layer_type, default_alpha)

        if "text_projection" in layer_name:
            continue

        if is_mat:
            # TSV-style concatenation
            offset = 0
            for i, dataset in enumerate(datasets):
                delta_svd = decomposed_task_vectors[dataset][layer_name]
                u, s, v = (
                    delta_svd["u"].to(device),
                    delta_svd["s"].to(device),
                    delta_svd["v"].to(device),
                )

                if i == 0:
                    total_rank = sum(
                        decomposed_task_vectors[d][layer_name]["s"].shape[0]
                        for d in datasets
                    )
                    sum_u = torch.zeros(u.shape[0], total_rank, device=device)
                    sum_s = torch.zeros(total_rank, device=device)
                    sum_v = torch.zeros(total_rank, v.shape[1], device=device)

                rank_i = s.shape[0]
                sum_u[:, offset : offset + rank_i] = u
                sum_s[offset : offset + rank_i] = s
                sum_v[offset : offset + rank_i, :] = v
                offset += rank_i

            # Procrustes orthogonalization
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

            if use_isotropic is True or use_isotropic == "mean":
                # Replace singular values with their mean (isotropic scaling)
                iso_factor = torch.mean(sum_s)
                merged = iso_factor * torch.linalg.multi_dot(
                    (u_u, v_u, u_v, v_v)
                )
            elif use_isotropic == "median":
                iso_factor = torch.median(sum_s)
                merged = iso_factor * torch.linalg.multi_dot(
                    (u_u, v_u, u_v, v_v)
                )
            elif use_isotropic == "geometric":
                # Geometric mean of SVs
                log_mean = torch.mean(torch.log(sum_s + 1e-10))
                iso_factor = torch.exp(log_mean)
                merged = iso_factor * torch.linalg.multi_dot(
                    (u_u, v_u, u_v, v_v)
                )
            elif use_isotropic == "topk":
                # Keep top half of SVs, zero the rest
                k = max(1, sum_s.shape[0] // 2)
                topk_s = sum_s.clone()
                topk_s[k:] = 0.0
                merged = torch.linalg.multi_dot(
                    (u_u, v_u, torch.diag(topk_s), u_v, v_v)
                )
            else:
                merged = torch.linalg.multi_dot(
                    (u_u, v_u, torch.diag(sum_s), u_v, v_v)
                )

            # Apply per-layer-type alpha
            aggregated[layer_name] = (alpha * merged).to(device)

        else:
            # 1D params: weighted average across tasks
            for i, dataset in enumerate(datasets):
                delta = decomposed_task_vectors[dataset][layer_name]["dim1"].to(device)
                if i == 0:
                    aggregated[layer_name] = delta.clone()
                else:
                    aggregated[layer_name] += (
                        delta - aggregated[layer_name]
                    ) / (i + 1)

            # Apply per-layer-type alpha
            aggregated[layer_name] = alpha * aggregated[layer_name]

    return aggregated


class InterferenceAwareMerger(TaskVectorBasedMerger):
    """
    Merger that applies different strategies based on layer-type interference patterns.

    Parameters:
        rank_per_type: dict mapping layer types to max SVD rank per task
            e.g. {"mlp": 32, "attention": 32, "projection": 16, "conv": 16, "embedding": 8}
        default_rank: fallback rank for unspecified layer types (default: 32)
        alpha_per_type: dict mapping layer types to scaling coefficients
            e.g. {"layernorm": 1.0, "bias": 1.0, "mlp": 0.8, "attention": 0.8}
        default_alpha: fallback alpha (default: 1.0)
        use_isotropic: if True, apply isotropic scaling to SVs (default: False)
    """

    def __init__(
        self,
        rank_per_type: dict = None,
        default_rank: int = 32,
        alpha_per_type: dict = None,
        default_alpha: float = 1.0,
        use_isotropic: bool = False,
        use_mp_edge: bool = False,
        mp_min_rank: int = 4,
        mp_max_rank: int = 128,
        normalize_task_vectors: bool = False,
    ):
        super().__init__()

        self.rank_per_type = rank_per_type or {}
        self.default_rank = default_rank
        self.alpha_per_type = alpha_per_type or {}
        self.default_alpha = default_alpha
        self.use_isotropic = use_isotropic
        self.use_mp_edge = use_mp_edge
        self.mp_min_rank = mp_min_rank
        self.mp_max_rank = mp_max_rank
        self.normalize_task_vectors = normalize_task_vectors

    def merge(self, base_model, finetuned_models):
        task_dicts = {}
        datasets = list(finetuned_models.keys())

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        if self.normalize_task_vectors:
            # Normalize each task vector per-layer to have the same Frobenius norm.
            # Use the geometric mean of norms as the target (preserves overall scale).
            for key in list(task_dicts[datasets[0]].keys()):
                norms = []
                for dataset in datasets:
                    norm = task_dicts[dataset][key].float().norm().item()
                    norms.append(norm)

                # Skip layers where all tasks have zero change
                if all(n < 1e-10 for n in norms):
                    continue

                # Target norm = geometric mean of non-zero norms
                nonzero_norms = [n for n in norms if n > 1e-10]
                if not nonzero_norms:
                    continue
                target_norm = float(torch.tensor(nonzero_norms).log().mean().exp())

                for i, dataset in enumerate(datasets):
                    if norms[i] > 1e-10:
                        task_dicts[dataset][key] = task_dicts[dataset][key] * (target_norm / norms[i])

            pylogger.info("Normalized task vectors to geometric mean norm per layer")

        pylogger.info(
            f"Interference-aware decomposition: rank_per_type={self.rank_per_type}, "
            f"default_rank={self.default_rank}"
        )
        svd_dict = decompose_with_layer_ranks(
            task_dicts, self.rank_per_type, self.default_rank,
            use_mp_edge=self.use_mp_edge,
            mp_min_rank=self.mp_min_rank,
            mp_max_rank=self.mp_max_rank,
        )

        pylogger.info(
            f"Aggregating: alpha_per_type={self.alpha_per_type}, "
            f"default_alpha={self.default_alpha}, isotropic={self.use_isotropic}"
        )
        multi_task_vector = aggregate_interference_aware(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            decomposed_task_vectors=svd_dict,
            alpha_per_type=self.alpha_per_type,
            default_alpha=self.default_alpha,
            use_isotropic=self.use_isotropic,
        )

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(multi_task_vector, merged_encoder)

        return merged_encoder
