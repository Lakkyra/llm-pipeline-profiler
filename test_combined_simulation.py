"""
test_combined_simulation.py
Mock simulator script that runs without any heavy libraries or hardware dependencies.
Mocks the PyTorch profiler's stop call to return a sample trace.
Simulates stages, memory metrics, CPU, token throughput (tps), and v2.0 Roofline operators,
exporting them to the dashboard.
"""
from llm_profiler import Tracer
from llm_profiler.collectors.torch_profiler import TorchProfilerCollector
import time
import random

# Mock the PyTorch profiler's stop method to return a valid Chrome trace string
mock_trace_json = """{
  "traceEvents": [
    {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "pid": 1234, "tid": 1, "ts": 1000, "dur": 600},
    {"ph": "X", "cat": "cpu_op", "name": "aten::add", "pid": 1234, "tid": 1, "ts": 1100, "dur": 200},
    {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "pid": 1234, "tid": 1, "ts": 1200, "dur": 100},
    {"ph": "X", "cat": "cpu_op", "name": "aten::softmax", "pid": 1234, "tid": 1, "ts": 1400, "dur": 150},
    
    {"ph": "X", "cat": "runtime", "name": "cudaLaunchKernel", "pid": 1234, "tid": 1, "ts": 1050, "dur": 50},
    
    {"ph": "X", "cat": "cuda_op", "name": "void sgemm_kernel", "pid": 1234, "tid": 5, "ts": 1060, "dur": 520},
    {"ph": "X", "cat": "cuda_op", "name": "void elementwise_add_kernel", "pid": 1234, "tid": 5, "ts": 1190, "dur": 180}
  ]
}"""

# Override the collector's stop method so it works without PyTorch installed
TorchProfilerCollector._active = True
TorchProfilerCollector.stop = lambda self: mock_trace_json

print("Initializing tracer with mock trace injector and hardware roofline calibration...")
tracer = Tracer(
    run_name="simulated-all-visuals-run",
    model_name="gpt2-simulated",
    dashboard_url="http://localhost:3000"
)

# Run a stage with profile_torch=True to trigger the mocked flamegraph trace
with tracer.stage("prefill", profile_torch=True) as stage:
    print("Executing simulated prefill stage with operator profiling...")
    
    # Profile a GEMM operator
    with stage.profile_op(
        op_name="qkv_projection_gemm",
        category="compute_dense",
        flops=2.0 * 32 * 4096 * 4096,
        bytes_transferred=(32*4096 + 4096*4096 + 32*4096) * 2.0,
    ) as ctx:
        time.sleep(0.02)

    # Profile attention
    with stage.profile_op(
        op_name="fused_multihead_attention",
        category="attention",
        flops=4.0 * 2 * 32 * (512**2) * 128,
        bytes_transferred=4.0 * 2 * 32 * 512 * 128 * 2.0,
    ) as ctx:
        time.sleep(0.015)

    # Profile RMSNorm
    with stage.profile_op(
        op_name="rmsnorm_layer",
        category="norm",
        flops=5.0 * 4096 * 32,
        bytes_transferred=2.0 * 4096 * 32 * 2.0,
    ) as ctx:
        time.sleep(0.005)

with tracer.stage("generate", profile_torch=False):
    print("Executing simulated generation loop...")
    
    # Simulate logging tokens per second over a generation loop
    for i in range(10):
        time.sleep(0.05)
        # Log a changing tokens/sec metric for the throughput chart
        current_tps = 32.5 + random.uniform(-4.0, 4.0)
        tracer.log_metric("tps", current_tps)
        tracer.log_metric("batch_size", 1.0)

print("Exporting run data to dashboard...")
tracer.export()

print(f"\nCeilings Calibrated:")
print(f"  Device: {tracer.ceilings.device_name}")
print(f"  Peak TFLOPS: {tracer.ceilings.peak_tflops}")
print(f"  Peak Bandwidth: {tracer.ceilings.peak_bandwidth_gb_s} GB/s")
print(f"  Ridge Point: {tracer.ceilings.ridge_point:.2f} FLOPs/Byte")

roofline_summary = tracer.get_roofline_summary()
print(f"\nRoofline Operator Telemetry:")
print(f"  Total Profiled Operators: {roofline_summary['total_operators']}")
print(f"  Compute-bound: {roofline_summary['counts']['compute_bound']}")
print(f"  Memory-bound : {roofline_summary['counts']['memory_bound']}")
print(f"  Latency-bound: {roofline_summary['counts']['latency_limited']}")

print("\nSuccess! Open the dashboard, click on this run, and you will see:")
print("1. Roofline Model (v2.0) Tab with interactive log-log chart and DevTech diagnostics.")
print("2. The Throughput Chart showing tokens/sec.")
print("3. The Stages Profile Details with a 'Trace Flamegraph' button showing CPU/GPU operators.")
