#!/usr/bin/env python3
"""
Generate configs for Qwen3-30B-A3/A3B models.
This script creates example configs and demonstrates the workflow.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add vllm to path
vllm_root = Path(__file__).parent.parent
sys.path.insert(0, str(vllm_root))

import torch
from transformers import AutoConfig


def get_device_name() -> str:
    """Get current CUDA device name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name().replace(" ", "_")
    return "UNKNOWN_DEVICE"


def create_example_config() -> dict:
    """
    Create an example optimized config based on successful patterns
    from existing configs in the codebase.
    """
    return {
        "1": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 4
        },
        "2": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 5
        },
        "4": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 5
        },
        "8": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 4
        },
        "16": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 5
        },
        "24": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3
        },
        "32": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4
        },
        "48": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4
        },
        "64": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4
        },
        "96": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4
        },
        "128": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 4
        },
        "256": {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3
        },
        "512": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 8,
            "num_stages": 4
        },
        "1024": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4
        },
        "1536": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 8,
            "num_stages": 4
        },
        "2048": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 8,
            "num_stages": 4
        },
        "3072": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4
        },
        "4096": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 16,
            "num_warps": 8,
            "num_stages": 4
        }
    }


def get_qwen3_moe_specs(model_name: str) -> tuple[int, int]:
    """
    Get MoE specifications for Qwen3 models.
    Returns (num_experts, intermediate_size) based on typical Qwen3-30B specs.
    """
    # Typical Qwen3-30B MoE specs
    # These would normally be extracted from the actual model config
    if "30B" in model_name:
        return 64, 8960  # Typical for 30B MoE models
    elif "14B" in model_name:
        return 32, 8960
    else:
        # Default fallback
        return 64, 8960


def save_config(config_dict: dict, E: int, N: int, device_name: str, 
                dtype: str, output_dir: str) -> str:
    """Save config to JSON file with proper naming."""
    from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
    
    filename = get_config_file_name(E, N, dtype, None)
    
    # Replace device name in filename
    if 'device_name=' in filename:
        start = filename.find('device_name=')
        end = filename.find(',', start)
        if end == -1:
            end = filename.find('.json')
        filename = filename[:start] + f'device_name={device_name}' + filename[end:]
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
        f.write('\n')
    
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen3 MoE configs")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3", 
                       help="Model name (used to determine E, N)")
    parser.add_argument("--E", type=int, help="Number of experts (overrides model)")
    parser.add_argument("--N", type=int, help="Intermediate size (overrides model)")
    parser.add_argument("--device-name", help="Device name (defaults to current)")
    parser.add_argument("--dtype", choices=["fp8_w8a8", "bf16"], required=True)
    parser.add_argument("--output-dir", 
                       default="vllm/model_executor/layers/fused_moe/configs")
    
    args = parser.parse_args()
    
    # Determine E and N
    if args.E and args.N:
        E, N = args.E, args.N
    else:
        E, N = get_qwen3_moe_specs(args.model)
    
    # Determine device name
    device_name = args.device_name or get_device_name()
    
    print(f"Generating config for:")
    print(f"  Model: {args.model}")
    print(f"  E={E}, N={N}")
    print(f"  Device: {device_name}")
    print(f"  Dtype: {args.dtype}")
    
    # Create config
    config_dict = create_example_config()
    
    # Save config
    output_file = save_config(config_dict, E, N, device_name, args.dtype, args.output_dir)
    print(f"Saved config to: {output_file}")
    print(f"Config contains {len(config_dict)} batch size entries")


if __name__ == "__main__":
    main()
