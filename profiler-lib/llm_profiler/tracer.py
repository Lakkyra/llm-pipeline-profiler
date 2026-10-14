import time
import os
import uuid
import sys
import threading
from datetime import datetime, timezone
from contextlib import ContextDecorator
from typing import Optional, List, Dict, Any, Callable

from .schema import RunSchema, StageSchema, MetricSchema, OperatorMetric, HardwareCeilings
from .collectors.cpu_mem import CPUMemCollector
from .collectors.gpu import GPUCollector
from .collectors.torch_profiler import TorchProfilerCollector
from .collectors.roofline_collector import RooflineCollector
from .analyzers.roofline_analyzer import RooflineAnalyzer
from .sampler_bridge import SamplerBridge
from .exporter import Exporter


class Tracer:
    def __init__(
        self,
        run_name: str,
        model_name: Optional[str] = None,
        dashboard_url: Optional[str] = None,
        sample_interval_ms: int = 50,
        auto_instrument: bool = True,
        profile_torch: bool = False,
        device_id: int = 0,
        ceilings: Optional[HardwareCeilings] = None,
        profile_roofline: bool = True,
    ):
        self.run_name = run_name
        self.model_name = model_name
        self.dashboard_url = dashboard_url
        self.sample_interval_ms = sample_interval_ms
        self.auto_instrument = auto_instrument
        self.profile_torch = profile_torch
        self.device_id = device_id
        self.profile_roofline = profile_roofline

        self.run_id = str(uuid.uuid4())
        self.created_at = datetime.now().astimezone()

        self.cpu_mem_collector = CPUMemCollector()
        self.gpu_collector = GPUCollector()
        self.sampler_bridge = SamplerBridge(sample_interval_ms=self.sample_interval_ms)

        # Calibrate hardware ceilings
        if ceilings is not None:
            self.ceilings = ceilings
        else:
            self.ceilings = self.sampler_bridge.probe_hardware(self.device_id)

        # Initialize roofline operator telemetry collector
        self.roofline_collector = RooflineCollector(
            ceilings=self.ceilings,
            enabled=self.profile_roofline,
        )

        # Determine hardware info
        cpu_count = os.cpu_count() or 1
        gpu_info = self.gpu_collector.get_hardware_info()
        self.hardware_info = {
            "cpu_count": cpu_count,
            "gpu_type": self.ceilings.device_name if self.ceilings.device_name != "NVIDIA GPU" else gpu_info.get("gpu_type"),
            "gpu_vram_mb": gpu_info.get("gpu_vram_mb"),
            "peak_tflops": self.ceilings.peak_tflops,
            "peak_bandwidth_gb_s": self.ceilings.peak_bandwidth_gb_s,
            "ridge_point": self.ceilings.ridge_point,
            "precision": self.ceilings.precision,
            "is_cuda_calibrated": self.ceilings.is_cuda_calibrated,
        }

        self.stages: List[StageSchema] = []
        self.exporter = Exporter(dashboard_url=self.dashboard_url)
        self._active_stage_contexts = []
        self._background_threads = []

        # Export initial empty run
        self._export_current_run()

        if self.auto_instrument:
            from .auto_instrument import register_tracer
            register_tracer(self)

    def stop_auto_instrument(self):
        """
        Stop auto-instrumentation for this Tracer. If no other active
        auto-instrumented Tracers exist, original methods will be restored.
        """
        if self.auto_instrument:
            from .auto_instrument import unregister_tracer
            unregister_tracer(self)
            self.auto_instrument = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_auto_instrument()

    def _export_current_run(self):
        all_ops = self.roofline_collector.get_operators()
        run_data = RunSchema(
            id=self.run_id,
            name=self.run_name,
            created_at=self.created_at,
            model_name=self.model_name,
            hardware_info=self.hardware_info,
            hardware_ceilings=self.ceilings,
            stages=self.stages,
            operators=all_ops,
        )
        self.exporter.export(run_data)

    def export(self):
        """
        Export the current run statistics to local file and HTTP endpoint.
        """
        self.stop_auto_instrument()
        # Wait for background serialization threads to complete
        if hasattr(self, "_background_threads"):
            for t in self._background_threads:
                if t.is_alive():
                    t.join()
            self._background_threads.clear()
        self._export_current_run()

    def stage(self, name: str, profile_torch: bool = False):
        return StageContext(self, name, profile_torch)

    def profile_op(
        self,
        op_name: str,
        category: str = "compute_dense",
        flops: float = 0.0,
        bytes_transferred: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Profiles an individual operator with high precision, calculating operational
        intensity and bound classification against the Roofline Model.
        """
        stage_id = self._active_stage_contexts[-1].stage_id if self._active_stage_contexts else None
        return self.roofline_collector.profile_op(
            op_name=op_name,
            category=category,
            stage_id=stage_id,
            flops=flops,
            bytes_transferred=bytes_transferred,
            metadata=metadata,
        )

    def benchmark_op(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        op_name: str = "custom_op",
        category: str = "compute_dense",
        flops: float = 0.0,
        bytes_transferred: float = 0.0,
        warmup_runs: Optional[int] = None,
        measure_runs: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperatorMetric:
        """
        Benchmarks a callable with warmup, IQR outlier rejection, and automated
        roofline classification.
        """
        stage_id = self._active_stage_contexts[-1].stage_id if self._active_stage_contexts else None
        return self.roofline_collector.benchmark_callable(
            fn=fn,
            args=args,
            kwargs=kwargs,
            op_name=op_name,
            category=category,
            stage_id=stage_id,
            flops=flops,
            bytes_transferred=bytes_transferred,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
            metadata=metadata,
        )

    def get_roofline_summary(self) -> Dict[str, Any]:
        return self.roofline_collector.get_summary()

    def log_metric(self, key: str, value: float):
        """
        Log a custom metric to the current active stage context.
        """
        if self._active_stage_contexts:
            self._active_stage_contexts[-1].log_metric(key, value)


class StageContext(ContextDecorator):
    def __init__(self, tracer: Tracer, name: str, profile_torch: bool = False):
        self.tracer = tracer
        self.name = name
        self.profile_torch = profile_torch

        self.stage_id = None
        self.metrics = []
        self._sampling_active = False
        self._sampling_thread = None
        self.torch_profiler = None
        self.start_time = None
        self.start_perf = None
        self._start_op_idx = 0

        self.start_cpu = 0.0
        self.start_ram = 0.0
        self.start_gpu_util = 0.0
        self.start_gpu_mem = 0.0
        self.use_sampler_cpp = False

    def log_metric(self, key: str, value: float):
        self.metrics.append(MetricSchema(
            timestamp=datetime.now().astimezone(),
            key=key,
            value=float(value)
        ))

    def profile_op(
        self,
        op_name: str,
        category: str = "compute_dense",
        flops: float = 0.0,
        bytes_transferred: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Convenience hook to profile an operator under this stage context."""
        return self.tracer.roofline_collector.profile_op(
            op_name=op_name,
            category=category,
            stage_id=self.stage_id,
            flops=flops,
            bytes_transferred=bytes_transferred,
            metadata=metadata,
        )

    def __enter__(self):
        self.stage_id = str(uuid.uuid4())
        self.metrics = []
        self._start_op_idx = len(self.tracer.roofline_collector.operators)

        # Synchronize CUDA if available to get accurate GPU start time
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        self.start_time = datetime.now().astimezone()
        self.start_perf = time.perf_counter()

        # Take start snapshots
        self.start_cpu = self.tracer.cpu_mem_collector.get_cpu_percent()
        self.start_ram = self.tracer.cpu_mem_collector.get_ram_used_mb()
        self.start_gpu_util = self.tracer.gpu_collector.get_gpu_util_percent()
        self.start_gpu_mem = self.tracer.gpu_collector.get_gpu_mem_used_mb()

        # Try C++ sampler bridge
        self.use_sampler_cpp = self.tracer.sampler_bridge.connect()

        # PyTorch Profiler
        if self.profile_torch:
            self.torch_profiler = TorchProfilerCollector()
            self.torch_profiler.start()
        else:
            self.torch_profiler = None

        # Track active context in tracer
        self.tracer._active_stage_contexts.append(self)

        # Start background sampler thread
        self._sampling_active = True
        self._sampling_thread = threading.Thread(target=self._sample_loop)
        self._sampling_thread.daemon = True
        self._sampling_thread.start()

        return self

    def _sample_loop(self):
        interval = self.tracer.sample_interval_ms / 1000.0
        while self._sampling_active:
            loop_start = time.perf_counter()
            timestamp = datetime.now().astimezone()

            if self.use_sampler_cpp:
                # Read from socket
                samples = self.tracer.sampler_bridge.read_samples()
                for sample in samples:
                    ts_str = sample.get("timestamp")
                    ts = datetime.now().astimezone()
                    if ts_str:
                        try:
                            ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                            ts = ts.replace(tzinfo=timezone.utc).astimezone()
                        except Exception:
                            try:
                                ts = datetime.strptime(ts_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                                ts = ts.replace(tzinfo=timezone.utc).astimezone()
                            except Exception:
                                pass
                    for key in ["cpu_percent", "ram_used_mb", "gpu_util_percent", "gpu_mem_used_mb"]:
                        if key in sample:
                            self.metrics.append(MetricSchema(
                                timestamp=ts,
                                key=key,
                                value=float(sample[key])
                            ))
            else:
                # Fallback to pure Python sampling
                cpu = self.tracer.cpu_mem_collector.get_cpu_percent()
                ram = self.tracer.cpu_mem_collector.get_ram_used_mb()
                gpu_util = self.tracer.gpu_collector.get_gpu_util_percent()
                gpu_mem = self.tracer.gpu_collector.get_gpu_mem_used_mb()

                self.metrics.append(MetricSchema(timestamp=timestamp, key="cpu_percent", value=cpu))
                self.metrics.append(MetricSchema(timestamp=timestamp, key="ram_used_mb", value=ram))
                if self.tracer.gpu_collector.enabled:
                    self.metrics.append(MetricSchema(timestamp=timestamp, key="gpu_util_percent", value=gpu_util))
                    self.metrics.append(MetricSchema(timestamp=timestamp, key="gpu_mem_used_mb", value=gpu_mem))

            # Compute remaining time to sleep to keep interval
            elapsed = time.perf_counter() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop background sampling thread first, wrapping in try/finally to guarantee execution
        try:
            self._sampling_active = False
        finally:
            if self._sampling_thread:
                self._sampling_thread.join(timeout=1.0)
                self._sampling_thread = None

        # Clean active context reference
        if self in self.tracer._active_stage_contexts:
            self.tracer._active_stage_contexts.remove(self)

        # Synchronize CUDA if available to get accurate GPU end time
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        end_time = datetime.now().astimezone()
        end_perf = time.perf_counter()

        # Stop PyTorch profiler if active and get trace in background
        if self.torch_profiler and self.torch_profiler._active and self.torch_profiler.profiler is None:
            self.torch_profiler = None

        profiler_to_serialize = None
        trace_dir = None
        if self.torch_profiler and self.torch_profiler.profiler:
            try:
                # Stop recording synchronously on the main thread (instantaneous)
                profiler_to_serialize = self.torch_profiler.profiler
                profiler_to_serialize.stop()
                self.torch_profiler._active = False
                trace_dir = self.torch_profiler.trace_dir
            except Exception as e:
                print(f"[llm-profiler] Error stopping torch profiler: {e}")
            self.torch_profiler = None

        duration_ms = (end_perf - self.start_perf) * 1000.0

        # Take end snapshots
        end_cpu = self.tracer.cpu_mem_collector.get_cpu_percent()
        end_ram = self.tracer.cpu_mem_collector.get_ram_used_mb()
        end_gpu_util = self.tracer.gpu_collector.get_gpu_util_percent()
        end_gpu_mem = self.tracer.gpu_collector.get_gpu_mem_used_mb()

        # Close sampler bridge socket
        self.tracer.sampler_bridge.close()

        # Compute summaries
        cpu_samples = [m.value for m in self.metrics if m.key == "cpu_percent"]
        ram_samples = [m.value for m in self.metrics if m.key == "ram_used_mb"]
        gpu_util_samples = [m.value for m in self.metrics if m.key == "gpu_util_percent"]
        gpu_mem_samples = [m.value for m in self.metrics if m.key == "gpu_mem_used_mb"]

        cpu_percent = sum(cpu_samples) / len(cpu_samples) if cpu_samples else (self.start_cpu + end_cpu) / 2.0
        ram_used_mb = max(ram_samples) if ram_samples else max(self.start_ram, end_ram)
        gpu_util_percent = sum(gpu_util_samples) / len(gpu_util_samples) if gpu_util_samples else (self.start_gpu_util + end_gpu_util) / 2.0
        gpu_mem_used_mb = max(gpu_mem_samples) if gpu_mem_samples else max(self.start_gpu_mem, end_gpu_mem)

        # Collect stage operators
        stage_operators = self.tracer.roofline_collector.operators[self._start_op_idx:]

        stage = StageSchema(
            id=self.stage_id,
            name=self.name,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            gpu_util_percent=gpu_util_percent,
            gpu_mem_used_mb=gpu_mem_used_mb,
            metrics=self.metrics,
            operators=list(stage_operators),
            trace_ref=None,
        )

        self.tracer.stages.append(stage)

        # Start the background serialization thread if a profiler was active
        if profiler_to_serialize and trace_dir:
            def serialize_trace(stage_object):
                import tempfile
                try:
                    fd, path = tempfile.mkstemp(suffix=".json", dir=trace_dir)
                    os.close(fd)
                    profiler_to_serialize.export_chrome_trace(path)
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            trace_content = f.read()
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        # Update the stage with the serialized trace in-memory
                        stage_object.trace_ref = trace_content
                except Exception as e:
                    print(f"[llm-profiler] Background trace export failed: {e}")

            t = threading.Thread(target=serialize_trace, args=(stage,))
            t.daemon = True
            t.start()
            if hasattr(self.tracer, "_background_threads"):
                self.tracer._background_threads.append(t)

        # Propagate exception if one occurred
        return False
