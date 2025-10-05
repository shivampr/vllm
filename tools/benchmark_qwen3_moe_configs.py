#!/usr/bin/env python3
"""
Enhanced MoE benchmark script for Qwen3-30B-A3/A3B config tuning.
Supports focused grid search and CSV output for config optimization.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add vllm to path
vllm_root = Path(__file__).parent.parent
sys.path.insert(0, str(vllm_root))

import torch
from transformers import AutoConfig

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    fused_topk,
    get_config_file_name,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


class BenchmarkConfig:
    """Configuration for MoE kernel benchmarking."""
    
    def __init__(self, **kwargs):
        self.BLOCK_SIZE_M = kwargs.get('BLOCK_SIZE_M', 64)
        self.BLOCK_SIZE_N = kwargs.get('BLOCK_SIZE_N', 128)
        self.BLOCK_SIZE_K = kwargs.get('BLOCK_SIZE_K', 256)
        self.GROUP_SIZE_M = kwargs.get('GROUP_SIZE_M', 1)
        self.num_warps = kwargs.get('num_warps', 4)
        self.num_stages = kwargs.get('num_stages', 4)
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'BLOCK_SIZE_M': self.BLOCK_SIZE_M,
            'BLOCK_SIZE_N': self.BLOCK_SIZE_N,
            'BLOCK_SIZE_K': self.BLOCK_SIZE_K,
            'GROUP_SIZE_M': self.GROUP_SIZE_M,
            'num_warps': self.num_warps,
            'num_stages': self.num_stages,
        }


def get_qwen3_moe_metadata(model_name_or_path: str) -> Tuple[int, int, int]:
    """
    Extract MoE metadata from Qwen3 model config.
    
    Returns:
        Tuple of (num_experts, intermediate_size, hidden_size)
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        if not hasattr(config, 'num_experts') or config.num_experts is None:
            raise ValueError(f"Model {model_name_or_path} is not a MoE model")
            
        num_experts = config.num_experts
        intermediate_size = config.moe_intermediate_size
        hidden_size = config.hidden_size
        
        return num_experts, intermediate_size, hidden_size
        
    except Exception as e:
        print(f"Error loading model config: {e}")
        sys.exit(1)


def get_search_grid() -> List[BenchmarkConfig]:
    """
    Generate focused search grid for Triton configs.
    Based on successful configurations from existing configs.
    """
    # Focused grid as specified in requirements
    block_size_m_range = [16, 32, 64, 128]
    block_size_n_range = [32, 64, 128, 256]
    block_size_k_range = [64, 128, 256]
    group_size_m_range = [1, 8, 16]
    num_warps_range = [2, 4, 8]
    num_stages_range = [2, 3, 4]
    
    configs = []
    for block_m, block_n, block_k, group_m, warps, stages in product(
        block_size_m_range, block_size_n_range, block_size_k_range,
        group_size_m_range, num_warps_range, num_stages_range
    ):
        configs.append(BenchmarkConfig(
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=group_m,
            num_warps=warps,
            num_stages=stages,
        ))
    
    return configs


def benchmark_config(
    config: BenchmarkConfig,
    batch_size: int,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    use_fp8: bool,
    num_iters: int = 100,
) -> float:
    """
    Benchmark a single config and return average latency in microseconds.
    """
    from vllm.model_executor.layers.fused_moe import override_config
    
    # Prepare test data
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device='cuda')
    
    if use_fp8:
        w1 = torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float8_e4m3fn, device='cuda')
        w2 = torch.randn(num_experts, hidden_size, intermediate_size // 2, dtype=torch.float8_e4m3fn, device='cuda')
        w1_scale = torch.randn(num_experts, dtype=torch.float32, device='cuda')
        w2_scale = torch.randn(num_experts, dtype=torch.float32, device='cuda')
        a1_scale = torch.randn(1, dtype=torch.float32, device='cuda')
        a2_scale = torch.randn(1, dtype=torch.float32, device='cuda')
        
        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
        )
    else:
        w1 = torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype, device='cuda')
        w2 = torch.randn(num_experts, hidden_size, intermediate_size // 2, dtype=dtype, device='cuda')
        quant_config = None
    
    # Prepare gating data
    gating_output = torch.randn(batch_size, num_experts, dtype=torch.float32, device='cuda')
    
    def run_kernel():
        with override_config(config.to_dict()):
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                x, gating_output, top_k, renormalize=True
            )
            return fused_experts(
                x, w1, w2, topk_weights, topk_ids,
                inplace=True, quant_config=quant_config
            )
    
    # Warmup
    run_kernel()
    torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        run_kernel()
        end_event.record()
        end_event.synchronize()
        
        latencies.append(start_event.elapsed_time(end_event))  # ms
    
    return sum(latencies) / len(latencies) * 1000  # Convert to microseconds


def compute_throughput(
    batch_size: int,
    intermediate_size: int,
    hidden_size: int,
    top_k: int,
    latency_us: float,
) -> float:
    """
    Compute throughput in TFLOP/s.
    """
    # MoE forward pass: 2 * batch_size * top_k * hidden_size * intermediate_size
    # (gate_up_proj + down_proj)
    flops = 2 * batch_size * top_k * hidden_size * intermediate_size
    tflops = flops / (latency_us * 1e6) * 1e-12
    return tflops


def benchmark_batch_size(
    batch_size: int,
    num_experts: int,
    intermediate_size: int,
    hidden_size: int,
    top_k: int,
    dtype: torch.dtype,
    use_fp8: bool,
    search_configs: List[BenchmarkConfig],
) -> Tuple[BenchmarkConfig, float, float]:
    """
    Find best config for a given batch size.
    """
    print(f"Benchmarking batch size {batch_size}...")
    
    best_config = None
    best_latency = float('inf')
    best_throughput = 0.0
    
    for config in search_configs:
        try:
            latency = benchmark_config(
                config, batch_size, num_experts, intermediate_size,
                hidden_size, top_k, dtype, use_fp8
            )
            throughput = compute_throughput(
                batch_size, intermediate_size, hidden_size, top_k, latency
            )
            
            if latency < best_latency:
                best_latency = latency
                best_throughput = throughput
                best_config = config
                
        except Exception as e:
            # Skip invalid configs
            continue
    
    print(f"  Best latency: {best_latency:.2f} μs, throughput: {best_throughput:.3f} TFLOP/s")
    return best_config, best_latency, best_throughput


def save_csv_results(
    results: List[Dict[str, Any]],
    output_file: str,
) -> None:
    """
    Save benchmark results to CSV.
    """
    fieldnames = [
        'batch_size', 'latency_us', 'throughput_tflops',
        'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K',
        'GROUP_SIZE_M', 'num_warps', 'num_stages'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 MoE configs")
    parser.add_argument("--model", required=True, help="Qwen3 model name or path")
    parser.add_argument("--dtype", choices=["fp8_w8a8", "bf16"], default="bf16")
    parser.add_argument("--batches", type=str, 
                       default="1,2,4,8,16,24,32,48,64,96,128,256,512,1024,1536,2048,3072,4096",
                       help="Comma-separated batch sizes")
    parser.add_argument("--output", default="benchmarks/results/", 
                       help="Output directory")
    parser.add_argument("--num-iters", type=int, default=50,
                       help="Number of iterations per config")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batches.split(',')]
    
    # Get model metadata
    num_experts, intermediate_size, hidden_size = get_qwen3_moe_metadata(args.model)
    top_k = 8  # Typical for Qwen3 models
    
    print(f"Model: {args.model}")
    print(f"E={num_experts}, N={intermediate_size}, H={hidden_size}")
    print(f"Top-k: {top_k}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Dtype: {args.dtype}")
    
    # Set up dtype
    if args.dtype == "fp8_w8a8":
        dtype = torch.float16  # Input dtype for FP8
        use_fp8 = True
    else:  # bf16
        dtype = torch.bfloat16
        use_fp8 = False
    
    # Get search grid
    search_configs = get_search_grid()
    print(f"Searching over {len(search_configs)} configurations...")
    
    # Benchmark each batch size
    results = []
    start_time = time.time()
    
    for batch_size in batch_sizes:
        config, latency, throughput = benchmark_batch_size(
            batch_size, num_experts, intermediate_size, hidden_size,
            top_k, dtype, use_fp8, search_configs
        )
        
        result = {
            'batch_size': batch_size,
            'latency_us': latency,
            'throughput_tflops': throughput,
            **config.to_dict()
        }
        results.append(result)
    
    end_time = time.time()
    print(f"Benchmarking completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    dtype_str = args.dtype if args.dtype == "fp8_w8a8" else "bf16"
    
    csv_file = os.path.join(args.output, f"qwen3_moe_E{num_experts}_N{intermediate_size}_{device_name}_{dtype_str}.csv")
    save_csv_results(results, csv_file)
    print(f"Results saved to {csv_file}")
    
    # Print summary
    print("\nSummary:")
    print("Batch Size | Latency (μs) | Throughput (TFLOP/s)")
    print("-" * 50)
    for result in results:
        print(f"{result['batch_size']:10} | {result['latency_us']:11.2f} | {result['throughput_tflops']:15.3f}")


if __name__ == "__main__":
    main()
