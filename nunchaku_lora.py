"""
Nunchaku LoRA loader for QwenImage transformers.
Based on ComfyUI-QwenImageLoraLoader implementation.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Dict, Optional, Tuple
import re

from nunchaku.lora.flux.nunchaku_converter import (
    pack_lowrank_weight,
    unpack_lowrank_weight,
)

# Storage for original weights to enable reset
_original_weights: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}


def reset_nunchaku_lora(transformer, verbose: bool = True) -> bool:
    """
    Reset transformer to original weights (before LoRA was applied).
    
    Returns:
        True if reset was performed, False if no original weights stored.
    """
    model_id = id(transformer)
    if model_id not in _original_weights:
        if verbose:
            print("No original weights stored - nothing to reset")
        return False
    
    restored = 0
    for module_name, weights in _original_weights[model_id].items():
        module = _get_module_by_name(transformer, module_name)
        if module is not None:
            module.proj_down.data = weights['proj_down'].clone()
            module.proj_up.data = weights['proj_up'].clone()
            module.rank = weights['rank']
            restored += 1
    
    # Clear stored weights
    del _original_weights[model_id]
    
    if verbose:
        print(f"Reset {restored} modules to original weights")
    return True


def has_nunchaku_lora(transformer) -> bool:
    """Check if LoRA has been applied to this transformer."""
    return id(transformer) in _original_weights


def load_nunchaku_lora(
    transformer,
    lora_path: str,
    strength: float = 1.0,
    verbose: bool = True,
    force: bool = False,
) -> None:
    """
    Load and apply LoRA to Nunchaku transformer.
    
    This implementation matches ComfyUI-QwenImageLoraLoader's compose_loras_v2:
    - Applies scale = strength * (alpha / rank) to B matrices
    - Fuses Q/K/V into block-diagonal B matrix
    - Concatenates A/B to Nunchaku's proj_down/proj_up
    
    Args:
        transformer: NunchakuQwenImageTransformer2DModel instance
        lora_path: Path to LoRA safetensors file  
        strength: LoRA strength (default 1.0)
        verbose: Print progress
        force: If True, reset existing LoRA before applying new one
    """
    model_id = id(transformer)
    
    # Check if LoRA already applied
    if model_id in _original_weights:
        if force:
            if verbose:
                print("Resetting existing LoRA before applying new one...")
            reset_nunchaku_lora(transformer, verbose=False)
        else:
            raise RuntimeError(
                "LoRA already applied to this transformer! "
                "Use reset_nunchaku_lora(transformer) first, or pass force=True to auto-reset."
            )
    if verbose:
        print(f"Loading Nunchaku LoRA from {lora_path}...")
    
    lora_state_dict = load_file(lora_path)
    
    if verbose:
        print(f"  Loaded {len(lora_state_dict)} LoRA tensors")
    
    # Group weights by module using ComfyUI's key mapping
    aggregated = _aggregate_lora_weights(lora_state_dict, strength)
    
    if verbose:
        print(f"  Aggregated into {len(aggregated)} target modules")
    
    # Initialize storage for original weights
    _original_weights[model_id] = {}
    
    # Apply to model
    applied_count = 0
    for module_name, (A, B) in aggregated.items():
        module = _get_module_by_name(transformer, module_name)
        if module is None:
            if verbose:
                print(f"  [MISS] Module not found: {module_name}")
            continue
        
        try:
            _apply_lora_to_module(module, A, B, module_name, model_id)
            applied_count += 1
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {module_name}: {e}")
    
    if verbose:
        print(f"  Applied LoRA to {applied_count} modules")


def _aggregate_lora_weights(
    lora_state_dict: Dict[str, torch.Tensor],
    strength: float,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parse and aggregate LoRA weights, applying scaling as ComfyUI does.
    Returns dict of module_name -> (A_fused, B_scaled_fused)
    """
    # Group by base module
    lora_grouped = {}
    
    for key, tensor in lora_state_dict.items():
        parsed = _classify_key(key)
        if parsed is None:
            continue
        
        base_key, group, component, ab = parsed
        
        if base_key not in lora_grouped:
            lora_grouped[base_key] = {'group': group}
        
        weight_key = f"{component}_{ab}" if component else ab
        lora_grouped[base_key][weight_key] = tensor
    
    # Process groups into fused tensors
    result = {}
    
    for base_key, weights in lora_grouped.items():
        group = weights.get('group', 'regular')
        
        if group == 'qkv':
            # Fuse Q/K/V
            A, B = _fuse_qkv(weights, strength)
        elif group == 'add_qkv':
            # Fuse add_q/k/v
            A, B = _fuse_qkv(weights, strength)
        else:
            # Regular layer
            A, B = _process_regular(weights, strength)
        
        if A is not None and B is not None:
            result[base_key] = (A, B)
    
    return result


def _classify_key(key: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Classify a LoRA key and return (base_key, group, component, ab).
    Matches ComfyUI's _classify_and_map_key logic.
    """
    # Strip prefixes
    k = key
    for prefix in ['transformer.', 'diffusion_model.', 'lora_unet_']:
        if k.startswith(prefix):
            k = k[len(prefix):]
    
    # Detect A/B/alpha
    ab = None
    base = None
    
    # Match lora_down/lora_up or lora_A/lora_B
    lora_match = re.search(r'\.(?P<tag>lora(?:[._](?:A|B|down|up)))(?:\.[^.]+)*\.weight$', k)
    if lora_match:
        tag = lora_match.group('tag')
        base = k[:lora_match.start()]
        if 'down' in tag or 'A' in tag:
            ab = 'A'
        elif 'up' in tag or 'B' in tag:
            ab = 'B'
    else:
        # Check for alpha
        alpha_match = re.search(r'\.(?:alpha|lora_alpha)(?:\.[^.]+)*$', k)
        if alpha_match:
            ab = 'alpha'
            base = k[:alpha_match.start()]
    
    if base is None or ab is None:
        return None
    
    # Map to target module
    return _map_key_to_module(base, ab)


def _map_key_to_module(base: str, ab: str) -> Optional[Tuple[str, str, Optional[str], str]]:
    """Map a base key to (target_module, group, component, ab)."""
    
    # QKV patterns - decomposed Q/K/V -> fused to_qkv
    qkv_match = re.match(r'^transformer_blocks[._](\d+)[._]attn[._]to[._](q|k|v)$', base)
    if qkv_match:
        block = qkv_match.group(1)
        comp = qkv_match.group(2).upper()
        return (f'transformer_blocks.{block}.attn.to_qkv', 'qkv', comp, ab)
    
    # Add QKV patterns
    add_qkv_match = re.match(r'^transformer_blocks[._](\d+)[._]attn[._]add[._](q|k|v)[._]proj$', base)
    if add_qkv_match:
        block = add_qkv_match.group(1)
        comp = add_qkv_match.group(2).upper()
        return (f'transformer_blocks.{block}.attn.add_qkv_proj', 'add_qkv', comp, ab)
    
    # to_out patterns
    to_out_match = re.match(r'^transformer_blocks[._](\d+)[._]attn[._]to[._]out(?:[._]0)?$', base)
    if to_out_match:
        block = to_out_match.group(1)
        return (f'transformer_blocks.{block}.attn.to_out.0', 'regular', None, ab)
    
    # to_add_out patterns
    to_add_out_match = re.match(r'^transformer_blocks[._](\d+)[._]attn[._]to[._]add[._]out$', base)
    if to_add_out_match:
        block = to_add_out_match.group(1)
        return (f'transformer_blocks.{block}.attn.to_add_out', 'regular', None, ab)
    
    # img_mlp patterns: img_mlp.net.0.proj and img_mlp.net.2
    img_mlp_match = re.match(r'^transformer_blocks[._](\d+)[._]img_mlp[._]net[._](\d+)(?:[._]proj)?$', base)
    if img_mlp_match:
        block = img_mlp_match.group(1)
        layer = img_mlp_match.group(2)
        if layer == '0':
            return (f'transformer_blocks.{block}.img_mlp.net.0.proj', 'regular', None, ab)
        else:
            return (f'transformer_blocks.{block}.img_mlp.net.{layer}', 'regular', None, ab)
    
    # txt_mlp patterns: txt_mlp.net.0.proj and txt_mlp.net.2
    txt_mlp_match = re.match(r'^transformer_blocks[._](\d+)[._]txt_mlp[._]net[._](\d+)(?:[._]proj)?$', base)
    if txt_mlp_match:
        block = txt_mlp_match.group(1)
        layer = txt_mlp_match.group(2)
        if layer == '0':
            return (f'transformer_blocks.{block}.txt_mlp.net.0.proj', 'regular', None, ab)
        else:
            return (f'transformer_blocks.{block}.txt_mlp.net.{layer}', 'regular', None, ab)
    
    return None


def _fuse_qkv(
    weights: Dict[str, torch.Tensor],
    strength: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fuse Q/K/V LoRA weights with ComfyUI-style scaling."""
    
    required = ['Q_A', 'Q_B', 'K_A', 'K_B', 'V_A', 'V_B']
    if not all(k in weights for k in required):
        return None, None
    
    A_q, A_k, A_v = weights['Q_A'], weights['K_A'], weights['V_A']
    B_q, B_k, B_v = weights['Q_B'], weights['K_B'], weights['V_B']
    
    # Get alpha (usually same for all Q/K/V)
    alpha = weights.get('Q_alpha', weights.get('K_alpha', weights.get('V_alpha')))
    
    # Calculate scale: strength * (alpha / rank)
    rank = A_q.shape[0]
    scale = strength
    if alpha is not None:
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        scale *= (alpha_val / rank)
    
    # Scale B matrices
    B_q_scaled = B_q * scale
    B_k_scaled = B_k * scale
    B_v_scaled = B_v * scale
    
    # Fuse A: concatenate
    A_fused = torch.cat([A_q, A_k, A_v], dim=0)
    
    # Fuse B: block diagonal
    out_q, out_k, out_v = B_q.shape[0], B_k.shape[0], B_v.shape[0]
    r = B_q.shape[1]
    
    B_fused = torch.zeros(out_q + out_k + out_v, 3 * r, dtype=B_q.dtype, device=B_q.device)
    B_fused[:out_q, :r] = B_q_scaled
    B_fused[out_q:out_q + out_k, r:2*r] = B_k_scaled
    B_fused[out_q + out_k:, 2*r:] = B_v_scaled
    
    return A_fused, B_fused


def _process_regular(
    weights: Dict[str, torch.Tensor],
    strength: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Process regular (non-fused) LoRA weights."""
    
    if 'A' not in weights or 'B' not in weights:
        return None, None
    
    A = weights['A']
    B = weights['B']
    alpha = weights.get('alpha')
    
    # Calculate scale
    rank = A.shape[0]
    scale = strength
    if alpha is not None:
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        scale *= (alpha_val / rank)
    
    B_scaled = B * scale
    
    return A, B_scaled


def _get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get a module by dot-separated path."""
    if not name:
        return model
    
    module = model
    for part in name.split('.'):
        if not part:
            continue
        
        if hasattr(module, part):
            module = getattr(module, part)
        elif part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential, list, tuple)):
            try:
                module = module[int(part)]
            except (IndexError, TypeError):
                return None
        else:
            return None
    
    return module


def _apply_lora_to_module(
    module: nn.Module,
    A: torch.Tensor,
    B: torch.Tensor,
    module_name: str,
    model_id: int,
) -> None:
    """
    Apply LoRA to a Nunchaku module exactly as ComfyUI does.
    Stores original weights for later reset.
    """
    if not hasattr(module, 'proj_down') or not hasattr(module, 'proj_up'):
        raise ValueError(f"{module_name}: Not a Nunchaku LoRA-ready module")
    
    # Store original weights BEFORE modification (for reset)
    _original_weights[model_id][module_name] = {
        'proj_down': module.proj_down.data.clone(),
        'proj_up': module.proj_up.data.clone(),
        'rank': module.rank,
    }
    
    # Unpack existing weights
    pd = unpack_lowrank_weight(module.proj_down.data, down=True)
    pu = unpack_lowrank_weight(module.proj_up.data, down=False)
    
    base_rank = pd.shape[0] if pd.shape[1] == module.in_features else pd.shape[1]
    
    # Determine concatenation axis for proj_down
    if pd.shape[1] == module.in_features:  # [rank, in]
        new_proj_down = torch.cat([pd, A.to(pd.device, pd.dtype)], dim=0)
    else:  # [in, rank]
        new_proj_down = torch.cat([pd, A.T.to(pd.device, pd.dtype)], dim=1)
    
    # Concatenate B to proj_up
    new_proj_up = torch.cat([pu, B.to(pu.device, pu.dtype)], dim=1)
    
    # Pack and update
    module.proj_down.data = pack_lowrank_weight(new_proj_down, down=True)
    module.proj_up.data = pack_lowrank_weight(new_proj_up, down=False)
    module.rank = base_rank + A.shape[0]
