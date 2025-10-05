#!/usr/bin/env python3
"""
JSON config emitter for fused MoE with plateau pruning.
Reads CSV results and emits optimized JSON configs with deduplication.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add vllm to path
vllm_root = Path(__file__).parent.parent
sys.path.insert(0, str(vllm_root))

from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name


def load_csv_results(csv_file: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from CSV file.
    """
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['batch_size', 'latency_us', 'throughput_tflops', 
                       'BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K',
                       'GROUP_SIZE_M', 'num_warps', 'num_stages']:
                if key in row:
                    row[key] = float(row[key]) if '.' in row[key] else int(row[key])
            results.append(row)
    return results


def prune_plateaus(results: List[Dict[str, Any]], 
                   min_improvement_pct: float = 1.0) -> List[Dict[str, Any]]:
    """
    Prune plateau configs - keep only turning points where config changes 
    or throughput improves significantly.
    
    Args:
        results: Sorted list of results by batch_size
        min_improvement_pct: Minimum throughput improvement to keep a config
    
    Returns:
        Pruned list of results
    """
    if not results:
        return results
    
    # Sort by batch size
    results = sorted(results, key=lambda x: x['batch_size'])
    
    pruned = [results[0]]  # Always keep the first one
    last_kept_config = results[0]
    last_kept_throughput = results[0]['throughput_tflops']
    
    for result in results[1:]:
        current_config = {
            'BLOCK_SIZE_M': result['BLOCK_SIZE_M'],
            'BLOCK_SIZE_N': result['BLOCK_SIZE_N'],
            'BLOCK_SIZE_K': result['BLOCK_SIZE_K'],
            'GROUP_SIZE_M': result['GROUP_SIZE_M'],
            'num_warps': result['num_warps'],
            'num_stages': result['num_stages'],
        }
        
        last_config = {
            'BLOCK_SIZE_M': last_kept_config['BLOCK_SIZE_M'],
            'BLOCK_SIZE_N': last_kept_config['BLOCK_SIZE_N'],
            'BLOCK_SIZE_K': last_kept_config['BLOCK_SIZE_K'],
            'GROUP_SIZE_M': last_kept_config['GROUP_SIZE_M'],
            'num_warps': last_kept_config['num_warps'],
            'num_stages': last_kept_config['num_stages'],
        }
        
        # Check if config changed
        config_changed = current_config != last_config
        
        # Check if throughput improved significantly
        throughput_improvement = ((result['throughput_tflops'] - last_kept_throughput) 
                                / last_kept_throughput * 100)
        throughput_improved = throughput_improvement >= min_improvement_pct
        
        # Keep if config changed or throughput improved
        if config_changed or throughput_improved:
            pruned.append(result)
            last_kept_config = result
            last_kept_throughput = result['throughput_tflops']
    
    return pruned


def create_config_dict(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Create config dictionary from results.
    """
    config_dict = {}
    
    for result in results:
        batch_size = int(result['batch_size'])
        config_dict[str(batch_size)] = {
            'BLOCK_SIZE_M': int(result['BLOCK_SIZE_M']),
            'BLOCK_SIZE_N': int(result['BLOCK_SIZE_N']),
            'BLOCK_SIZE_K': int(result['BLOCK_SIZE_K']),
            'GROUP_SIZE_M': int(result['GROUP_SIZE_M']),
            'num_warps': int(result['num_warps']),
            'num_stages': int(result['num_stages']),
        }
    
    return config_dict


def save_json_config(
    config_dict: Dict[str, Dict[str, int]],
    E: int,
    N: int,
    device_name: str,
    dtype: str,
    output_dir: str,
    block_shape: Optional[List[int]] = None,
) -> str:
    """
    Save config dictionary to JSON file with proper naming.
    """
    filename = get_config_file_name(E, N, dtype, block_shape)
    
    # Replace device name in filename with actual device name
    # The get_config_file_name function uses torch.cuda.get_device_name()
    # but we want to use the provided device_name
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
    parser = argparse.ArgumentParser(description="Emit fused MoE JSON configs from CSV")
    parser.add_argument("csv_file", help="CSV file with benchmark results")
    parser.add_argument("--E", type=int, required=True, help="Number of experts")
    parser.add_argument("--N", type=int, required=True, help="Intermediate size")
    parser.add_argument("--device-name", required=True, help="Device name (e.g., NVIDIA_H100)")
    parser.add_argument("--dtype", choices=["fp8_w8a8", "bf16"], required=True)
    parser.add_argument("--output-dir", default="vllm/model_executor/layers/fused_moe/configs",
                       help="Output directory for JSON config")
    parser.add_argument("--min-improvement", type=float, default=1.0,
                       help="Minimum throughput improvement % to keep config")
    parser.add_argument("--no-pruning", action="store_true",
                       help="Disable plateau pruning")
    
    args = parser.parse_args()
    
    # Load and process results
    print(f"Loading results from {args.csv_file}...")
    results = load_csv_results(args.csv_file)
    print(f"Loaded {len(results)} results")
    
    # Apply plateau pruning
    if not args.no_pruning:
        print(f"Applying plateau pruning (min improvement: {args.min_improvement}%)...")
        results = prune_plateaus(results, args.min_improvement)
        print(f"After pruning: {len(results)} results")
    
    # Create config dictionary
    config_dict = create_config_dict(results)
    
    # Save JSON config
    output_file = save_json_config(
        config_dict,
        args.E,
        args.N,
        args.device_name,
        args.dtype,
        args.output_dir,
    )
    
    print(f"Saved config to {output_file}")
    print(f"Config contains {len(config_dict)} batch size entries:")
    
    # Print summary table
    print("\nBatch Size | Throughput (TFLOP/s) | Config")
    print("-" * 70)
    for result in results:
        batch_size = int(result['batch_size'])
        throughput = result['throughput_tflops']
        config = config_dict[str(batch_size)]
        config_str = f"M{config['BLOCK_SIZE_M']}N{config['BLOCK_SIZE_N']}K{config['BLOCK_SIZE_K']}G{config['GROUP_SIZE_M']}W{config['num_warps']}S{config['num_stages']}"
        print(f"{batch_size:10} | {throughput:15.3f} | {config_str}")


if __name__ == "__main__":
    main()
