#!/usr/bin/env python3
"""
Utility to print MoE metadata (E, N values) for a given model.
This helps determine the exact values needed for config filenames.
"""
import argparse
import os
import sys
from pathlib import Path

# Add vllm to path
vllm_root = Path(__file__).parent.parent
sys.path.insert(0, str(vllm_root))

import torch
from transformers import AutoConfig


def get_moe_metadata(model_name_or_path: str) -> tuple[int, int]:
    """
    Extract MoE metadata (E=num_experts, N=intermediate_size) from model config.
    
    Args:
        model_name_or_path: HuggingFace model name or path to model
        
    Returns:
        Tuple of (num_experts, intermediate_size)
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Check if this is a MoE model
        if not hasattr(config, 'num_experts') or config.num_experts is None:
            raise ValueError(f"Model {model_name_or_path} is not a MoE model (no num_experts)")
            
        if not hasattr(config, 'intermediate_size'):
            raise ValueError(f"Model {model_name_or_path} has no intermediate_size")
            
        num_experts = config.num_experts
        intermediate_size = config.intermediate_size
        
        return num_experts, intermediate_size
        
    except Exception as e:
        print(f"Error loading model config: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Print MoE metadata for config filenames")
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--device", action="store_true", 
                       help="Also print current CUDA device name")
    
    args = parser.parse_args()
    
    # Get MoE metadata
    num_experts, intermediate_size = get_moe_metadata(args.model)
    
    print(f"E={num_experts} N={intermediate_size}")
    
    if args.device:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            print(f"device_name={device_name}")
        else:
            print("CUDA not available")


if __name__ == "__main__":
    main()
