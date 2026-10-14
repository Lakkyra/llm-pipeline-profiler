"""
Roofline Collector & Telemetry Engine for llm-pipeline-profiler v2.0.

Provides high-precision cudaEvent timing, analytical and hardware counter dispatch,
and runtime operator classification.
"""

import time
import math
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Callable
import uuid

from ..schema import OperatorMetric, HardwareCeilings
from ..analyzers.roofline_analyzer import RooflineAnalyzer


class ProfileContext:
    """Mutable context passed into `profile_op` blocks allowing dynamic telemetry updates."""
    def __init__(self, op_name: str, category: str, flops: float = 0.0, bytes_transferred: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        self.op_name = op_name
        self.category = category
        self.flops = flops
        self.bytes_transferred = bytes_transferred
        self.metadata = metadata or {}


class RooflineCollector:
    """
    Operator-level telemetry collector for hardware Roofline modeling.
    Tracks high-resolution CUDA / CPU latencies, FLOPs, memory transfers,
    and classifies bottlenecks via RooflineAnalyzer.
    """

    def __init__(
        self,
        ceilings: Optional[HardwareCeilings] = None,
        warmup_runs: int = 3,
        measure_runs: int = 10,
        enabled: bool = True,
    ):
        self.ceilings = ceilings or HardwareCeilings()
        self.analyzer = RooflineAnalyzer(self.ceilings)
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs
        self.enabled = enabled
        self.operators: List[OperatorMetric] = []

        # Check PyTorch CUDA availability
        self._has_torch = False
        self._has_cuda = False
        try:
            import torch
            self._torch = torch
            self._has_torch = True
            self._has_cuda = torch.cuda.is_available()
        except ImportError:
            self._torch = None

    def set_ceilings(self, ceilings: HardwareCeilings) -> None:
        self.ceilings = ceilings
        self.analyzer.set_ceilings(ceilings)

    @contextmanager
    def profile_op(
        self,
        op_name: str,
        category: str = "compute_dense",
        stage_id: Optional[str] = None,
        flops: float = 0.0,
        bytes_transferred: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager to measure execution latency and capture hardware metrics for an operator.
        Uses CUDA Events on GPU or high-resolution perf_counter on CPU.
        """
        if not self.enabled:
            ctx = ProfileContext(op_name, category, flops, bytes_transferred, metadata)
            yield ctx
            return

        ctx = ProfileContext(op_name, category, flops, bytes_transferred, metadata)
        start_event = None
        end_event = None
        t0_ns = 0

        use_cuda_timing = self._has_cuda and self._torch is not None and self._torch.cuda.is_initialized()

        if use_cuda_timing:
            start_event = self._torch.cuda.Event(enable_timing=True)
            end_event = self._torch.cuda.Event(enable_timing=True)
            self._torch.cuda.synchronize()
            start_event.record()
        else:
            t0_ns = time.perf_counter_ns()

        try:
            yield ctx
        finally:
            if use_cuda_timing and start_event is not None and end_event is not None:
                end_event.record()
                self._torch.cuda.synchronize()
                duration_ms = float(start_event.elapsed_time(end_event))
            else:
                t1_ns = time.perf_counter_ns()
                duration_ms = max((t1_ns - t0_ns) / 1_000_000.0, 1e-4)

            op = OperatorMetric(
                id=str(uuid.uuid4()),
                stage_id=stage_id,
                op_name=ctx.op_name,
                category=ctx.category,
                duration_ms=duration_ms,
                flops=float(ctx.flops),
                bytes_transferred=float(ctx.bytes_transferred),
                operational_intensity=0.0,
                achieved_tflops=0.0,
                achieved_bandwidth_gb_s=0.0,
                bound_classification="compute_bound",
                recommendation="",
                metadata=ctx.metadata,
            )

            # Classify operator through analyzer
            classified_op = self.analyzer.classify_operator(op)
            self.operators.append(classified_op)

    def benchmark_callable(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        op_name: str = "custom_op",
        category: str = "compute_dense",
        stage_id: Optional[str] = None,
        flops: float = 0.0,
        bytes_transferred: float = 0.0,
        warmup_runs: Optional[int] = None,
        measure_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperatorMetric:
        """
        Executes warmup passes, runs multiple iterations, trims IQR outliers,
        and records a statistically robust OperatorMetric.
        """
        kwargs = kwargs or {}
        warmup = warmup_runs if warmup_runs is not None else self.warmup_runs
        runs = measure_runs if measure_runs is not None else self.measure_runs

        use_cuda = self._has_cuda and self._torch is not None and self._torch.cuda.is_initialized()

        # Warmup passes
        for _ in range(warmup):
            fn(*args, **kwargs)
        if use_cuda:
            self._torch.cuda.synchronize()

        durations_ms: List[float] = []

        if use_cuda:
            for _ in range(runs):
                s = self._torch.cuda.Event(enable_timing=True)
                e = self._torch.cuda.Event(enable_timing=True)
                s.record()
                fn(*args, **kwargs)
                e.record()
                e.synchronize()
                durations_ms.append(float(s.elapsed_time(e)))
        else:
            for _ in range(runs):
                t0 = time.perf_counter_ns()
                fn(*args, **kwargs)
                t1 = time.perf_counter_ns()
                durations_ms.append((t1 - t0) / 1_000_000.0)

        # Statistical IQR outlier rejection
        if len(durations_ms) >= 4:
            sorted_times = sorted(durations_ms)
            q1_idx = len(sorted_times) // 4
            q3_idx = (len(sorted_times) * 3) // 4
            q1 = sorted_times[q1_idx]
            q3 = sorted_times[q3_idx]
            iqr = q3 - q1
            filtered = [t for t in sorted_times if (q1 - 1.5 * iqr) <= t <= (q3 + 1.5 * iqr)]
            if not filtered:
                filtered = sorted_times
            duration_ms = sum(filtered) / len(filtered)
        else:
            duration_ms = sum(durations_ms) / max(len(durations_ms), 1)

        meta = dict(metadata or {})
        meta["runs_measured"] = len(durations_ms)
        meta["p50_latency_ms"] = sorted(durations_ms)[len(durations_ms) // 2]

        op = OperatorMetric(
            id=str(uuid.uuid4()),
            stage_id=stage_id,
            op_name=op_name,
            category=category,
            duration_ms=duration_ms,
            flops=float(flops),
            bytes_transferred=float(bytes_transferred),
            operational_intensity=0.0,
            achieved_tflops=0.0,
            achieved_bandwidth_gb_s=0.0,
            bound_classification="compute_bound",
            recommendation="",
            metadata=meta,
        )

        classified_op = self.analyzer.classify_operator(op)
        self.operators.append(classified_op)
        return classified_op

    def get_operators(self) -> List[OperatorMetric]:
        return list(self.operators)

    def clear(self) -> None:
        self.operators.clear()

    def get_summary(self) -> Dict[str, Any]:
        return self.analyzer.summarize(self.operators)
