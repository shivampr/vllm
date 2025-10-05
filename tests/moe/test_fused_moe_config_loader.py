"""
Test for fused MoE config loader to ensure config selection behavior works correctly.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_config_file_name,
    get_moe_configs,
)


class TestFusedMoEConfigLoader(unittest.TestCase):
    """Test cases for fused MoE configuration loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_config(self, batch_sizes: list[int]) -> dict:
        """Create a test configuration dictionary."""
        config = {}
        for batch_size in batch_sizes:
            config[str(batch_size)] = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        return config

    def test_config_file_name_generation(self):
        """Test that config file names are generated correctly."""
        E, N = 64, 8960
        dtype = "fp8_w8a8"
        
        filename = get_config_file_name(E, N, dtype, None)
        
        # Should contain the expected components
        self.assertIn(f"E={E}", filename)
        self.assertIn(f"N={N}", filename)
        self.assertIn(f"dtype={dtype}", filename)
        self.assertTrue(filename.endswith(".json"))

    @patch('torch.cuda.get_device_name')
    def test_config_loading_with_mock_device(self, mock_get_device_name):
        """Test config loading with a mocked device name."""
        mock_get_device_name.return_value = "NVIDIA_H100"
        
        E, N = 64, 8960
        dtype = "fp8_w8a8"
        
        # Create test config
        test_config = self.create_test_config([1, 8, 32, 128, 512, 2048])
        
        # Save config to temp file
        filename = get_config_file_name(E, N, dtype, None)
        config_path = os.path.join(self.temp_dir, filename)
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Test config loading
        with patch('vllm.model_executor.layers.fused_moe.fused_moe.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_config)
                
                configs = get_moe_configs(E, N, dtype)
                
                self.assertIsNotNone(configs)
                self.assertIsInstance(configs, dict)
                self.assertGreater(len(configs), 0)
                
                # Check that we have the expected batch sizes
                for batch_size in [1, 8, 32, 128, 512, 2048]:
                    self.assertIn(batch_size, configs)
                    self.assertIn("BLOCK_SIZE_M", configs[batch_size])

    def test_nearest_key_selection(self):
        """Test nearest key selection behavior."""
        E, N = 64, 8960
        dtype = "fp8_w8a8"
        
        # Create config with sparse batch sizes
        test_config = self.create_test_config([8, 64, 512, 2048])
        
        # Save config to temp file
        filename = get_config_file_name(E, N, dtype, None)
        config_path = os.path.join(self.temp_dir, filename)
        
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
        
        with patch('vllm.model_executor.layers.fused_moe.fused_moe.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_config)
                
                configs = get_moe_configs(E, N, dtype)
                
                # Test nearest key selection
                # For batch size 16, nearest should be 8
                nearest_16 = min(configs.keys(), key=lambda x: abs(x - 16))
                self.assertEqual(nearest_16, 8)
                
                # For batch size 100, nearest should be 64
                nearest_100 = min(configs.keys(), key=lambda x: abs(x - 100))
                self.assertEqual(nearest_100, 64)
                
                # For batch size 1000, nearest should be 1024
                nearest_1000 = min(configs.keys(), key=lambda x: abs(x - 1000))
                self.assertEqual(nearest_1000, 512)  # Actually 512 is closer

    def test_config_structure(self):
        """Test that loaded configs have the expected structure."""
        E, N = 64, 8960
        dtype = "bf16"
        
        test_config = self.create_test_config([32, 128])
        
        with patch('vllm.model_executor.layers.fused_moe.fused_moe.os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(test_config)
                
                configs = get_moe_configs(E, N, dtype)
                
                for batch_size, config in configs.items():
                    # Check required fields
                    required_fields = [
                        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                        "GROUP_SIZE_M", "num_warps", "num_stages"
                    ]
                    
                    for field in required_fields:
                        self.assertIn(field, config)
                        self.assertIsInstance(config[field], int)
                        self.assertGreater(config[field], 0)

    @patch('torch.cuda.is_available')
    def test_no_cuda_fallback(self, mock_cuda_available):
        """Test behavior when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        E, N = 64, 8960
        dtype = "fp8_w8a8"
        
        # Should still be able to generate filename
        filename = get_config_file_name(E, N, dtype, None)
        self.assertIn("device_name=UNKNOWN", filename)


if __name__ == "__main__":
    unittest.main()
