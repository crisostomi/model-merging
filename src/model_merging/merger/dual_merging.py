
import copy
import logging
import math

import torch

from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    print_memory,
)
from model_merging.merging.structured import (
    get_svd_dict,
    isotropic_sum,
    avg_layers,
)

pylogger = logging.getLogger(__name__)

def scale_nested(scalar, data):
    """
    Recursively multiply a scalar with a tensor or nested tuple of tensors.
    """
    if isinstance(data, dict):
        for k in data:
            data[k]  =  data[k]*scalar
        return data
    elif isinstance(data, tuple):
        return tuple(scale_nested(scalar, x) for x in data)
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


def flatten_and_move_to_device(nested, device='cuda', clone=True):
    """
    Recursively flatten nested tuples of dictionaries into a single dictionary,
    move all tensors to the specified device, optionally clone them.
    """
    flat_dict = {}
    
    if isinstance(nested, dict):
        for k, v in nested.items():
            if clone:
                flat_dict[k] = v.clone().to(device)
            else:
                flat_dict[k] = v.to(device)
    elif isinstance(nested, tuple):
        for item in nested:
            flat_dict.update(flatten_and_move_to_device(item, device, clone))
    else:
        raise TypeError(f"Unexpected type: {type(nested)}")
    
    return flat_dict


class Module:
    def __init__(self, mass, sensitivity, dualize=None):
        self.mass = mass
        self.sensitivity = sensitivity
        self.dualize = dualize
    
    def set_mass(self, mass):
        self.mass = mass
        
    def set_dualize(self, dualize):
        self.dualize = dualize
        
    def get_mass(self):
        return self.mass
    
    def get_sensitivity(self):
        return self.sensitivity
        
    def get_dualitymap(self):
        return self.dualize


def create_linear_mod(g, name, mass):
    def linear_dualize():
        U, S, Vt = torch.linalg.svd(g, full_matrices=False)
        return {name: U @ Vt * math.sqrt(g.shape[0] / g.shape[1])}
    M = Module(mass, 1, linear_dualize)
    return M


def create_conv2d_mod(g, name, mass):
    def conv_dualize():
        matrix = g
        dout, din, k, _ = matrix.shape
        
        scaling_factor = (1.0 / k**2) * math.sqrt(dout / din)
        transformed = torch.zeros_like(matrix)
        
        for i in range(k):
            for j in range(k):
                slice_matrix = matrix[:, :, i, j]
                U, S, Vt = torch.linalg.svd(slice_matrix, full_matrices=False)
                reconstructed = U @ Vt
                transformed[:, :, i, j] = scaling_factor * reconstructed
        return{name: transformed}
    M = Module(mass, 1, conv_dualize)
    return M


def create_embedding_mod(g, name,mass):
    def embedding_dualize():
        rms_norm = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        return {name: g / rms_norm}
    M = Module(mass, 1, embedding_dualize)
    return M


def concatenate(M1, M2):
    M = Module(M1.get_mass() + M2.get_mass(), 
               M1.get_sensitivity() + M2.get_sensitivity())
    
    def concat_dualize():
        ratio1 = M1.get_mass() / M.get_mass()
        ratio2 = M2.get_mass() / M.get_mass()
        g1 = M1.get_dualitymap()()
        g2 = M2.get_dualitymap()()
        return (scale_nested(ratio1, g1), scale_nested(ratio2, g2))
    
    M.set_dualize(concat_dualize)
    return M


def compose(M2, M1):
    M = Module(M1.get_mass() + M2.get_mass(), 
               M1.get_sensitivity() * M2.get_sensitivity())
    
    def compose_dualize():
        sensitivity_factor = 1.0 / M2.get_sensitivity()
        ratio1 = M1.get_mass() / M.get_mass()
        ratio2 = M2.get_mass() / M.get_mass()
        g1 = M1.get_dualitymap()()
        g2 = M2.get_dualitymap()()
        return (scale_nested(sensitivity_factor * ratio1, g1),
                scale_nested(ratio2, g2))
    
    M.set_dualize(compose_dualize)
    return M


def build_clip_vit_network_module(layer_names, grads, masses):
    """
    Build a modular duality network for CLIP ViT.
    
    Architecture:
    - Visual encoder: conv1 → 12 transformer blocks → projection
    - Text encoder: token_embedding → 12 transformer blocks
    - Each transformer block: attn (in_proj → out_proj) and mlp (c_fc → c_proj)
    
    Args:
        layer_names: List of parameter names from the model
    
    Returns:
        module_map: Dictionary containing all modules
    """
    module_map = {}
    
    print("\n" + "="*80)
    print("="*80)
    
    # ========================================================================
    # Step 1: Create atomic modules for all layers
    # ========================================================================
    print("\n" + "="*80)
    print("Step 1: Creating Atomic Layer Modules")
    print("="*80)
    for name in layer_names:
        # Skip biases, layer norms, and non-trainable parameters
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue
        mass = masses[name]
        # Visual conv1
        if 'visual.conv1.weight' in name:
            module_map['visual_conv1'] = create_conv2d_mod(grads[name], name, mass)
            print(f"✓ visual_conv1: Conv2D module")
        
        # Visual projection
        elif 'visual.proj' in name and 'out_proj' not in name:
            module_map['visual_proj'] = create_linear_mod(grads[name], name,mass) 
            print(f"✓ visual_proj: Linear module")
        
        # Visual positional embedding
        elif 'visual.positional_embedding' in name:
            print(f"⊗ visual.positional_embedding: SKIPPED (parameter)") 
        
        # Text token embedding
        elif 'token_embedding.weight' in name:
            module_map['token_embedding'] = create_embedding_mod(grads[name],name,mass)
            print(f"✓ token_embedding: Embedding module")
        
        # Text positional embedding
        elif 'positional_embedding' in name and 'visual' not in name:
            print(f"⊗ model.positional_embedding: SKIPPED (parameter)")
        
        # Text projection
        elif 'text_projection' in name:
            module_map['text_projection'] = create_linear_mod(grads[name],name,mass)
            print(f"✓ text_projection: Linear module")
        
        # Visual transformer blocks
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            # Extract block number - handle both with and without 'model.' prefix
            # Example: 'model.visual.transformer.resblocks.0.attn.in_proj_weight'
            # or: 'visual.transformer.resblocks.0.attn.in_proj_weight'
            
            # Find 'resblocks' and get the next part
            parts = name.split('.')
            try:
                resblocks_idx = parts.index('resblocks')
                block_idx = int(parts[resblocks_idx + 1])
            except (ValueError, IndexError):
                print(f"⚠ Skipping malformed name: {name}")
                continue
            
            block_name = f"visual_block_{block_idx}"
            
            if 'attn.in_proj_weight' in name:
                key = f'{block_name}_attn_in'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name], name,mass)
                    print(f"✓ {key}: Linear module")
            elif 'attn.out_proj.weight' in name:
                key = f'{block_name}_attn_out'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name],name,mass)
                    print(f"✓ {key}: Linear module")
            elif 'mlp.c_fc.weight' in name:
                key = f'{block_name}_mlp_fc'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name],name,mass)
                    print(f"✓ {key}: Linear module")
            elif 'mlp.c_proj.weight' in name:
                key = f'{block_name}_mlp_proj'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name],name,mass)
                    print(f"✓ {key}: Linear module")
    
    # ========================================================================
    # Step 2: Build visual transformer blocks
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2: Building Visual Transformer Blocks")
    print("="*80)
    
    # Determine how many blocks we actually have
    block_indices = set()
    for key in module_map.keys():
        if key.startswith('visual_block_'):
            # Extract block index from key like 'visual_block_0_attn_in'
            parts = key.split('_')
            if len(parts) >= 3 and parts[2].isdigit():
                block_indices.add(int(parts[2]))
    
    num_blocks = len(block_indices)
    print(f"\nFound {num_blocks} transformer blocks")
    
    visual_blocks = []
    for i in sorted(block_indices):
        block_name = f"visual_block_{i}"
        
        # Check if all required components exist
        required_keys = [
            f'{block_name}_attn_in',
            f'{block_name}_attn_out',
            f'{block_name}_mlp_fc',
            f'{block_name}_mlp_proj'
        ]
        
        if not all(key in module_map for key in required_keys):
            print(f"⚠ Skipping incomplete block {i}")
            continue
        
        # HACK: why do we compose? Non Linearities?
        # Compose attention: out_proj ∘ in_proj 
        attn_block = compose(
            module_map[f'{block_name}_attn_out'],  # M2 (applied second)
            module_map[f'{block_name}_attn_in']    # M1 (applied first)
        )
        module_map[f'{block_name}_attn'] = attn_block
        
        # Compose MLP: c_proj ∘ c_fc
        mlp_block = compose(
            module_map[f'{block_name}_mlp_proj'],  # M2 (applied second)
            module_map[f'{block_name}_mlp_fc']     # M1 (applied first)
        )
        module_map[f'{block_name}_mlp'] = mlp_block
        
        # Compose full block: mlp ∘ attn (SEQUENTIAL, not parallel)
        full_block = compose(
            mlp_block,   # M2 (applied second)
            attn_block   # M1 (applied first)
        )
        module_map[f'{block_name}'] = full_block
        visual_blocks.append(full_block)
        
        print(f"✓ {block_name} = mlp ∘ attn  [Mass: {full_block.get_mass():.2f}]")
    
    # ========================================================================
    # Step 3: Compose all visual transformer blocks sequentially
    # ========================================================================
    print("\n" + "="*80)
    print("Step 3: Composing Visual Transformer Blocks Sequentially")
    print("="*80)
    
    if len(visual_blocks) == 0:
        print("⚠ ERROR: No visual blocks found!")
        return module_map
    
    # Compose blocks sequentially: block_N ∘ ... ∘ block_1 ∘ block_0
    visual_transformer = visual_blocks[0]
    for i in range(1, len(visual_blocks)):
        visual_transformer = compose(
            visual_blocks[i],      # M2 (later block, applied second)
            visual_transformer     # M1 (earlier blocks, applied first)
        )
        print(f"✓ Composed blocks 0-{i}  [Mass: {visual_transformer.get_mass():.2f}]")
    
    module_map['visual_transformer'] = visual_transformer
    print(f"\n✓ visual_transformer complete [Mass: {visual_transformer.get_mass():.2f}]")
    
    # ========================================================================
    # Step 4: Build visual encoder
    # ========================================================================
    print("\n" + "="*80)
    print("Step 4: Building Visual Encoder")
    print("="*80)
    
    if 'visual_conv1' not in module_map:
        print("⚠ ERROR: visual_conv1 not found!")
        return module_map
    
    # Visual encoder: visual_transformer ∘ conv1
    visual_backbone = compose(
        visual_transformer,           # M2 (applied second)
        module_map['visual_conv1']    # M1 (applied first)
    )
    module_map['visual_backbone'] = visual_backbone
    print(f"✓ visual_backbone = visual_transformer ∘ conv1")
    print(f"  Mass: {visual_backbone.get_mass():.2f}")
    
    # Add projection if it exists
    if 'visual_proj' in module_map:
        visual_encoder = compose(
            module_map['visual_proj'],  # M2 (applied second)
            visual_backbone             # M1 (applied first)
        )
        module_map['visual_encoder'] = visual_encoder
        print(f"✓ visual_encoder = visual_proj ∘ visual_backbone")
        print(f"  Mass: {visual_encoder.get_mass():.2f}")
    else:
        module_map['visual_encoder'] = visual_backbone
        print(f"⚠ No visual_proj found, using backbone as encoder")
    
    # ========================================================================
    # Step 5: Build complete network (just visual for now)
    # ========================================================================
    print("\n" + "="*80)
    print("Step 5: Final Network Module")
    print("="*80)
    
    # For simplicity, we'll treat the visual encoder as the main network
    module_map['network'] = module_map['visual_encoder']
    
    print(f"\n{'='*80}")
    print(f"✓ NETWORK = visual_encoder")
    print(f"  Total Mass:        {module_map['network'].get_mass():.2f}")
    print(f"  Total Sensitivity: {module_map['network'].get_sensitivity():.2f}")
    print(f"{'='*80}")
    
    return module_map


class DualMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, svd_path, svd_compress_factor,model_name, device=None, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.model_name = model_name

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        pylogger.info(f"DualMerger initialized on device: {self.device}")
        
    @torch.no_grad()
    def merge(self, base_model, finetuned_models):

        base_model = base_model.to(self.device)

        task_dicts = {}
        datasets = list(finetuned_models.keys())
        
        num_tasks = len(datasets) 

        for dataset in datasets:

            ft_state_dict = {
                k: v.to(self.device) for k, v in finetuned_models[dataset].items()
            }

            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), ft_state_dict
            )

            del finetuned_models[dataset] 
            del ft_state_dict 
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        ref_state_dict = {k: v.to(self.device) for k, v in base_model.state_dict().items()}

        multi_task_vector = avg_layers( # HACK returns avg task vector
            ref_state_dict=ref_state_dict,
            svd_dict=svd_dict,
        )
        
        list_layer = list(multi_task_vector.keys())
        masses = {key: 0.5 for key in multi_task_vector}  # TODO: masses can be configured and swept
        module_net = build_clip_vit_network_module(list_layer, copy.deepcopy(multi_task_vector), masses)
        module_vec = flatten_and_move_to_device(module_net['network'].get_dualitymap()())
        for key in module_vec:
            multi_task_vector[key] = module_vec[key]
        model_name = self.model_name

        if self.alpha is not None:
            coefficient = self.alpha
        elif (
            model_name in self.optimal_alphas
            and f"{num_tasks}" in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][f"{num_tasks}"]
        else:
            raise ValueError(
                f"No alpha provided and no optimal alpha found for model {model_name} with {num_tasks} tasks"
            )

        merged_encoder = copy.deepcopy(base_model)
        pylogger.info(f"Using alpha: {coefficient}")
        
        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder


class DualCommonTaskSpecificMerger(TaskVectorBasedMerger):
    def __init__(
        self,
        common_space_fraction,
        optimal_alphas,
        model_name,
        device,
        svd_path, 
        svd_compress_factor,
        *args, **kwargs
    ):
        super().__init__()

        self.common_space_fraction = common_space_fraction
        self.optimal_alphas = optimal_alphas
        self.model_name = model_name
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        
    @torch.no_grad()
    def merge(self, base_model, finetuned_models) -> ImageEncoder:
        multi_task_vector = {}
        datasets = list(finetuned_models.keys())
        
        task_dicts = {}
        list_layer = list(finetuned_models[datasets[0]].keys())
        masses = {key: 0.5 for key in finetuned_models[datasets[0]]}
        num_tasks = len(datasets)

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        pylogger.info("Computing SVD...")
        ref_task_dict = task_dicts[datasets[0]]
        for key in ref_task_dict:
            shape_ = ref_task_dict[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
            if not is_2d_matrix:
                pylogger.info(f"Combining by avg {key}...")

                for i, (dataset, task_dict) in enumerate(task_dicts.items()):
                    vec = task_dict[key].to(self.device)
                    if i == 0:
                        multi_task_vector[key] = vec.clone()
                    else:
                        multi_task_vector[key] += (vec - multi_task_vector[key]) / (
                            i + 1
                        )
                continue

            pylogger.info(f"Computing common space using sum for {key}...")
            combined_w = sum(
                [task_dict[key].to(self.device) for task_dict in task_dicts.values()]
            )

            ### Calculate the common space size (making sure that task specific space is equally divisible) ###
            common_space_index_s = int(min(shape_) * self.common_space_fraction)
            _task_specific_total_space_index_s = (
                round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
            )
            common_space_index_s = min(shape_) - _task_specific_total_space_index_s

            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]
            common_space_s = s[:common_space_index_s]
            common_space_v = v[:common_space_index_s, :]
            ###################################################################

            ### Calculate task specific space ###
            n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
            for i, task_dict in enumerate(task_dicts.values()):
                w = task_dict[key].to(self.device)

                # calculate the projection onto task specific space to remove the common space
                w_ts = w - common_space_u @ common_space_u.T @ w
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                if i == 0:
                    combined_space_u = torch.zeros_like(u_ts, device=self.device)
                    combined_space_s = torch.zeros_like(s_ts, device=self.device)
                    combined_space_v = torch.zeros_like(v_ts, device=self.device)

                combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    u_ts[:, :n_dims_per_task]
                )
                combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    s_ts[:n_dims_per_task]
                )
                combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = (
                    v_ts[:n_dims_per_task, :]
                )
            ###################################################################

            combined_space_u[
                :,
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
            ] = common_space_u
            combined_space_s[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s
            ] = common_space_s
            combined_space_v[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
                :,
            ] = common_space_v

            ### Orthogonalize combined_space_u and combined_space_v ###
            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u, full_matrices=False)
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v, full_matrices=False)
            )
            combined_space_u = u_combined_space_u @ v_combined_space_u
            combined_space_v = u_combined_space_v @ v_combined_space_v
            ###################################################################

            combined_space_s = (
                torch.ones_like(combined_space_s) * combined_space_s.mean()
            )

            multi_task_vector[key] = torch.linalg.multi_dot(
                (
                    combined_space_u,
                    torch.diag(combined_space_s),
                    combined_space_v,
                )
            )

        module_net = build_clip_vit_network_module(list_layer, copy.deepcopy(multi_task_vector), masses)
        module_vec = flatten_and_move_to_device(module_net['network'].get_dualitymap()(), device=str(self.device))
        for key in module_vec:
            multi_task_vector[key] = module_vec[key]

        model_name = self.model_name
        if (
            model_name in self.optimal_alphas
            and f"{num_tasks}" in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][f"{num_tasks}"]
        else:
            raise ValueError(
                f"No optimal alpha found for model {model_name} with {num_tasks} tasks"
            )

        pylogger.info(f"Using alpha: {coefficient}")
        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
 