from model_merging.permutations.permutation_spec import PermutationSpec
from model_merging.matching.utils import LayerIterationOrder
import torch
from model_merging.utils import timeit
from model_merging.matching.weight_matching import weight_matching


class Matcher:
    def __init__(self, name, permutation_spec: PermutationSpec):
        self.name = name
        self.permutation_spec = permutation_spec

    def __call__(self, *args, **kwargs):
        pass


class DummyMatcher(Matcher):
    def __init__(self, name, permutation_spec: PermutationSpec):
        super().__init__(name, permutation_spec)

    def __call__(self, fixed, permutee):
        fixed = fixed.state_dict()

        perm_sizes = {
            p: fixed[params_and_axes[0][0]].shape[params_and_axes[0][1]]
            for p, params_and_axes in self.permutation_spec.perm_to_layers_and_axes.items()
        }

        permutation_indices = {p: torch.arange(n) for p, n in perm_sizes.items()}

        return permutation_indices, None


class GitRebasinMatcher(Matcher):
    def __init__(
        self,
        name,
        permutation_spec: PermutationSpec,
        max_iter=100,
        layer_iteration_order: LayerIterationOrder = LayerIterationOrder.RANDOM,
    ):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.layer_iteration_order = layer_iteration_order

    @timeit
    def __call__(self, fixed, permutee):
        permutation_indices = weight_matching(
            ps=self.permutation_spec,
            fixed=fixed.state_dict(),
            permutee=permutee.state_dict(),
            max_iter=self.max_iter,
            layer_iteration_order=self.layer_iteration_order,
            verbose=True,
        )

        return permutation_indices, None
