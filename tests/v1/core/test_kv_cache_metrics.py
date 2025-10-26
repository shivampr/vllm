# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.core.kv_cache_metrics import BlockMetricsState, KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class TestBlockMetricsState:
    
    def test_initialization(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
            assert state.birth_time_ns == 1000000000
            assert state.last_access_ns == 1000000000
            assert len(state.access_history) == 0
            assert state.max_request_end_ns == 0
    
    def test_record_access(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        
        with patch('time.monotonic_ns', return_value=2000000000):
            state.record_access()
        
        assert state.last_access_ns == 2000000000
        assert list(state.access_history) == [2000000000]
    
    def test_ring_buffer_size_4(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        
        # Add 5 accesses
        for i in range(5):
            with patch('time.monotonic_ns', return_value=1000000000 + (i + 1) * 1000000000):
                state.record_access()
        
        # Only keeps last 4
        assert len(state.access_history) == 4
        assert list(state.access_history) == [2000000000, 3000000000, 4000000000, 5000000000]
    
    def test_lifetime_calculation(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        
        lifetime = state.get_lifetime_seconds(6500000000)
        assert abs(lifetime - 5.5) < 0.001
    
    def test_idle_time_calculation(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        state.last_access_ns = 2000000000
        
        idle = state.get_idle_time_seconds(5200000000)
        assert abs(idle - 3.2) < 0.001
    
    def test_reuse_gaps(self):
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        
        base = 1000000000
        for offset in [0, 1.5, 3.0, 5.5]:
            state.access_history.append(base + int(offset * 1e9))
        
        gaps = state.get_reuse_gaps_seconds()
        assert len(gaps) == 3
        assert abs(gaps[0] - 1.5) < 0.001
        assert abs(gaps[1] - 1.5) < 0.001
        assert abs(gaps[2] - 2.5) < 0.001
    
    def test_ring_wrap_gaps(self):
        """5 accesses in size-4 buffer → 3 gaps."""
        with patch('time.monotonic_ns', return_value=1000000000):
            state = BlockMetricsState()
        
        for i in range(5):
            state.access_history.append(1000000000 + i * 1000000000)
        
        assert len(state.get_reuse_gaps_seconds()) == 3


class TestKVCacheMetricsCollector:
    
    def test_disabled_does_nothing(self):
        collector = KVCacheMetricsCollector(enabled=False)
        assert not collector.enabled
        
        block = KVCacheBlock(block_id=0)
        collector.on_block_allocated(block)
        collector.on_block_accessed(block)
        collector.on_block_evicted(block)
        
        assert len(collector.block_metrics) == 0
    
    def test_sampling_validation(self):
        # Negative → disabled
        c = KVCacheMetricsCollector(enabled=True, sample_rate=-0.1)
        assert not c.enabled
        
        # Zero → disabled
        c = KVCacheMetricsCollector(enabled=True, sample_rate=0.0)
        assert not c.enabled
        
        # > 1.0 → clamped
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.5)
        assert c.sample_rate == 1.0
        
        # Valid
        c = KVCacheMetricsCollector(enabled=True, sample_rate=0.5)
        assert c.sample_rate == 0.5
    
    def test_sampling_rate(self):
        # 100%
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100
        
        # 0%
        c = KVCacheMetricsCollector(enabled=True, sample_rate=0.0)
        assert sum(1 for _ in range(100) if c.should_sample_block()) == 0
        
        # ~50%
        c = KVCacheMetricsCollector(enabled=True, sample_rate=0.5)
        samples = sum(1 for _ in range(1000) if c.should_sample_block())
        assert 400 < samples < 600
    
    def test_block_allocation(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        with patch('time.monotonic_ns', return_value=1000000000):
            for block in blocks:
                c.on_block_allocated(block)
        
        assert len(c.block_metrics) == 5
    
    def test_block_access(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        for i in range(3):
            with patch('time.monotonic_ns', return_value=1000000000 + (i + 1) * 1000000000):
                c.on_block_accessed(block)
        
        assert len(c.block_metrics[0].access_history) == 3
    
    def test_no_access_eviction(self):
        """No accesses → lifetime == idle."""
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        lifetime_hist = MagicMock()
        idle_hist = MagicMock()
        c.histogram_block_lifetime = lifetime_hist
        c.histogram_idle_before_evict = idle_hist
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        with patch('time.monotonic_ns', return_value=6000000000):
            c.on_block_evicted(block)
        
        lifetime = lifetime_hist.observe.call_args[0][0]
        idle = idle_hist.observe.call_args[0][0]
        assert abs(lifetime - 5.0) < 0.001
        assert abs(idle - 5.0) < 0.001
    
    def test_eviction_observes_metrics(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        c.histogram_idle_before_evict = MagicMock()
        c.histogram_reuse_gap = MagicMock()
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        with patch('time.monotonic_ns', return_value=2000000000):
            c.on_block_accessed(block)
        with patch('time.monotonic_ns', return_value=3000000000):
            c.on_block_accessed(block)
        
        with patch('time.monotonic_ns', return_value=4000000000):
            c.on_block_evicted(block)
        
        assert c.histogram_block_lifetime.observe.called
        assert c.histogram_idle_before_evict.observe.called
        assert c.histogram_reuse_gap.observe.called
        assert 0 not in c.block_metrics
    
    def test_prefix_residency(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_prefix_residency = MagicMock()
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        with patch('time.monotonic_ns', return_value=2000000000):
            c.on_request_prefill_complete("req1", {0})
        
        assert c.block_metrics[0].max_request_end_ns == 2000000000
        
        with patch('time.monotonic_ns', return_value=7000000000):
            c.on_block_evicted(block)
        
        residency = c.histogram_prefix_residency.observe.call_args[0][0]
        assert abs(residency - 5.0) < 0.001
    
    def test_multi_request_prefix(self):
        """Multiple requests → tracks max end time."""
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_prefix_residency = MagicMock()
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        with patch('time.monotonic_ns', return_value=2000000000):
            c.on_request_prefill_complete("req1", {0})
        
        with patch('time.monotonic_ns', return_value=5000000000):
            c.on_request_prefill_complete("req2", {0})
        
        # Should track max (5s, not 2s)
        assert c.block_metrics[0].max_request_end_ns == 5000000000
        
        with patch('time.monotonic_ns', return_value=10000000000):
            c.on_block_evicted(block)
        
        # Residency = 10s - 5s = 5s
        residency = c.histogram_prefix_residency.observe.call_args[0][0]
        assert abs(residency - 5.0) < 0.001
    
    def test_reset(self):
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        
        with patch('time.monotonic_ns', return_value=1000000000):
            for i in range(5):
                c.on_block_allocated(KVCacheBlock(block_id=i))
        
        assert len(c.block_metrics) == 5
        c.reset()
        assert len(c.block_metrics) == 0
        
        # Can allocate after reset
        with patch('time.monotonic_ns', return_value=2000000000):
            c.on_block_allocated(KVCacheBlock(block_id=10))
        assert 10 in c.block_metrics
    
    def test_thread_safety(self):
        """Concurrent alloc/access/evict should not crash."""
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        c.histogram_idle_before_evict = MagicMock()
        
        errors = []
        
        def alloc(start, count):
            try:
                for i in range(count):
                    c.on_block_allocated(KVCacheBlock(block_id=start + i))
            except Exception as e:
                errors.append(e)
        
        def access(start, count):
            try:
                for i in range(count):
                    c.on_block_accessed(KVCacheBlock(block_id=start + i))
            except Exception as e:
                errors.append(e)
        
        def evict(start, count):
            try:
                for i in range(count):
                    c.on_block_evicted(KVCacheBlock(block_id=start + i))
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=alloc, args=(0, 100)),
            threading.Thread(target=alloc, args=(100, 100)),
            threading.Thread(target=access, args=(0, 100)),
            threading.Thread(target=access, args=(100, 100)),
            threading.Thread(target=evict, args=(0, 50)),
            threading.Thread(target=evict, args=(100, 50)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_monotonic_time_jump(self):
        """Large time jumps should be handled gracefully."""
        c = KVCacheMetricsCollector(enabled=True, sample_rate=1.0)
        c.histogram_block_lifetime = MagicMock()
        
        block = KVCacheBlock(block_id=0)
        with patch('time.monotonic_ns', return_value=1000000000):
            c.on_block_allocated(block)
        
        # Jump 100 years
        huge_time = 1000000000 + int(100 * 365 * 24 * 3600 * 1e9)
        with patch('time.monotonic_ns', return_value=huge_time):
            c.on_block_evicted(block)
        
        assert c.histogram_block_lifetime.observe.called
        lifetime = c.histogram_block_lifetime.observe.call_args[0][0]
        assert lifetime > 0


@pytest.mark.skip(reason="Integration test - requires running server")
class TestKVCacheMetricsIntegration:
    
    def test_metrics_endpoint(self):
        """
        Start server with --kv-cache-metrics --kv-cache-metrics-sample=1.0
        Send requests, scrape /metrics, verify histograms present.
        """
        pass
