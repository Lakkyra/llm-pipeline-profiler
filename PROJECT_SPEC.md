# LLM Pipeline Profiler — Project Spec (v2.0)

## 1. What this project is

A profiling and hardware bottleneck diagnostic system for LLM inference/training pipelines run on Kaggle notebooks or locally. It has four main parts:

1. **profiler-lib** — a Python package imported inside the Kaggle notebook or script that
   instruments pipeline stages, tracks CPU/RAM/GPU VRAM metrics, captures op-level PyTorch GPU traces, and provides high-precision `cudaEvent` operator-level Roofline telemetry.
2. **sampler-cpp** — a high-performance C++/CUDA hardware prober and system sampler (NVML + `/proc` + CUDA microbenchmarks)
   exposed to Python via pybind11, calibrated to measure empirical compute saturation ($P_{\text{peak}}$ TFLOPS) and DRAM memory bandwidth ($B_{\text{peak}}$ GB/s).
3. **dashboard** — a Next.js 16 app (PostgreSQL + Prisma) that ingests run data (via
   file upload, live HTTP POST, or SSE stream) and visualizes it: interactive Log-Log Roofline Model (v2.0), stage timelines, memory curves, throughput charts, and op-level flamegraphs.
4. **chrome-extension** — an optional thin add-on that scrapes Kaggle's own visible resource meters.

---

## 2. Full file structure

```text
llm-pipeline-profiler/
│
├── profiler-lib/
│   ├── pyproject.toml
│   ├── README.md
│   └── llm_profiler/
│       ├── __init__.py
│       ├── tracer.py
│       ├── schema.py
│       ├── exporter.py
│       ├── sampler_bridge.py
│       ├── collectors/
│       │   ├── roofline_collector.py      # [v2.0] cudaEvent timer & FLOP / Byte tracker
│       │   ├── cpu_mem.py
│       │   ├── gpu.py
│       │   └── torch_profiler.py
│       └── analyzers/
│           └── roofline_analyzer.py       # [v2.0] Williams Roofline model & DevTech classifier
│
├── sampler-cpp/
│   ├── CMakeLists.txt                     # Updated with CUDA compiler (nvcc) support
│   ├── src/
│   │   ├── hardware_prober.cu / .cuh      # [v2.0] Peak FMA TFLOPS & DRAM bandwidth microbenchmarks
│   │   ├── nvml_reader.cpp / .h
│   │   ├── proc_reader.cpp / .h
│   │   ├── ipc_server.cpp / .h
│   │   └── main.cpp
│   └── bindings/
│       └── pybind_wrapper.cpp             # Exposes probe_hardware() to Python
│
├── dashboard/
│   ├── package.json
│   ├── prisma/
│   │   └── schema.prisma                  # [v2.0] Extended with OperatorMetric and ceilings
│   └── src/
│       ├── app/
│       │   ├── api/
│       │   │   ├── runs/route.ts
│       │   │   ├── runs/[id]/route.ts
│       │   │   ├── runs/[id]/roofline/    # [v2.0] Serves (I, P) operator telemetry
│       │   │   └── stream/route.ts
│       │   └── dashboard/
│       │       └── [runId]/page.tsx       # [v2.0] Tabbed UI: Roofline (v2.0), Timelines, Traces
│       └── components/
│           ├── RooflineChart.tsx          # [v2.0] Interactive Log-Log Roofline chart
│           ├── Timeline.tsx
│           ├── Flamegraph.tsx
│           ├── MemoryChart.tsx
│           └── ThroughputChart.tsx
│
├── examples/
│   ├── kaggle_notebook_example.ipynb
│   └── roofline_operator_benchmark.py     # [v2.0] Core transformer operator benchmark suite
│
├── test_hardware_prober.py                # [v2.0] Hardware discovery unit tests
├── test_roofline_analyzer.py              # [v2.0] Mathematical formulation & classification tests
├── test_combined_simulation.py
└── docker-compose.yml
```

---

## 3. Data Contract: Roofline Extensions

### `OperatorMetric`
- `id`: string (UUID)
- `stage_id`: string (optional)
- `op_name`: string (e.g. `gemm_qkv`, `flash_attention`)
- `category`: string (`compute_dense`, `attention`, `norm`, `elementwise`)
- `duration_ms`: float (sub-microsecond execution time)
- `flops`: float (algorithmic floating-point operations)
- `bytes_transferred`: float (DRAM bytes transferred)
- `operational_intensity`: float ($I = \frac{\text{FLOPs}}{\text{Byte}}$)
- `achieved_tflops`: float ($P_{\text{achieved}} = \frac{\text{FLOPs}}{\text{Duration}} \times 10^{-12}$)
- `achieved_bandwidth_gb_s`: float ($B_{\text{achieved}} = \frac{\text{Bytes}}{\text{Duration}} \times 10^{-9}$)
- `bound_classification`: string (`compute_bound`, `memory_bound`, `latency_limited`)
- `recommendation`: string (Actionable DevTech optimization suggestion)

### `HardwareCeilings`
- `device_name`: string (e.g. `NVIDIA A100-SXM4-80GB`)
- `peak_tflops`: float (empirical or spec ceiling $P_{\text{peak}}$)
- `peak_bandwidth_gb_s`: float (empirical or spec bandwidth $B_{\text{peak}}$)
- `ridge_point`: float ($R = \frac{P_{\text{peak}} \times 10^3}{B_{\text{peak}}}$)
- `precision`: string (`FP32`, `FP16`)
- `is_cuda_calibrated`: boolean

---

## 4. Architectural Formulations & DevTech Heuristics

1. **Ridge Point:**
   $$R = \frac{P_{\text{peak}} \times 10^3}{B_{\text{peak}}} \quad [\text{FLOPs/Byte}]$$
2. **Theoretical Roofline Ceiling:**
   $$P_{\text{roof}}(I) = \min(P_{\text{peak}}, I \times B_{\text{peak}} \times 10^{-3}) \quad [\text{TFLOPS}]$$
3. **Bottleneck Classifications:**
   - **Latency-Limited:** $P_{\text{achieved}} < 0.30 \times P_{\text{roof}}(I)$
     *Action:* Check low warp occupancy, tail effects from small grid launches, or host launch overhead.
   - **Memory-Bound:** $I < R$ and $P_{\text{achieved}} \ge 0.30 \times P_{\text{roof}}(I)$
     *Action:* Kernel saturates DRAM bandwidth. Apply operator fusion (e.g., Fuse Bias+GELU into GEMM), use shared memory tiling, or replace standard Attention with FlashAttention.
   - **Compute-Bound:** $I \ge R$ and $P_{\text{achieved}} \ge 0.30 \times P_{\text{roof}}(I)$
     *Action:* Kernel saturates compute pipelines. Leverage FP16/BF16 Tensor Cores, unroll innermost loops, or investigate FP8/INT4 quantization.
