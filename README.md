# LLM Pipeline Profiler v2.0

A full-stack performance profiling and hardware bottleneck diagnostic engine for LLM inference and training pipelines — model loading, tokenization, autoregressive generation, and post-processing — executed in remote environments like Kaggle notebooks, or locally.

Built to answer two fundamental systems questions:
1. **Pipeline & System Level:** Where is latency and memory going across stages, and how do two runs compare after an optimization?
2. **GPU Hardware & Kernel Level (v2.0):** Is each transformer operator (GEMM, Attention, Norm, Activation) bounded by **DRAM Memory Bandwidth**, **SM Compute Saturation**, or **Launch Latency**, and what is the exact DevTech action to fix it?

---

## What's New in v2.0: Institutional-Grade GPU Roofline Profiling

v2.0 introduces deep GPU architecture calibration and kernel telemetry modeled after the **Williams Roofline Model**:

* **Empirical Peak Hardware Calibration (`sampler-cpp`):**
  * Dedicated CUDA microbenchmarks (`hardware_prober.cu`) executing unrolled 16-FMA loops to saturate FP32 SM tensor pipelines ($P_{\text{peak}}$ TFLOPS).
  * 128-bit streaming vector memory kernels (`float4`) measuring sustained DRAM bus throughput ($B_{\text{peak}}$ GB/s).
  * Automatic calculation of the architectural **Ridge / Balance Point**:
    $$R = \frac{P_{\text{peak}} \times 10^3}{B_{\text{peak}}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$
  * Built-in fallback database with verified NVIDIA whitepaper specs (H100, A100, V100, T4, RTX 4090, RTX 3090, L4).

* **Operator Telemetry & Operational Intensity Collector (`profiler-lib`):**
  * Sub-microsecond execution timing using native `cudaEventRecord` pairs on dedicated CUDA streams with statistical IQR outlier trimming.
  * Algorithmic FLOP counters:
    * **GEMM ($M \times K \times N$):** $\text{FLOPs} = 2MKN$
    * **Attention ($B, H, S, D$):** $\text{FLOPs} = 4BH S^2 D$
    * **LayerNorm / RMSNorm ($N$):** $\text{FLOPs} \approx 5N$
  * DRAM Byte transfers tracking minimum required data movement across memory hierarchy, quantifying operational intensity $I = \frac{\text{FLOPs}}{\text{Byte}}$.

* **Automated DevTech Diagnostic Engine & Interactive Log-Log Roofline (`dashboard`):**
  * Interactive SVG log-log canvas showing sloped DRAM bandwidth ceiling, flat compute ceiling, ridge point marker, and 30% under-utilization boundary.
  * Automated bottleneck classification and DevTech recommendations:
    * **Memory-Bound ($I < R$):** Saturating memory bus. *Action: Apply kernel fusion (e.g., Fuse Bias+GELU into preceding GEMM), use shared memory tiling, or replace standard Attention with FlashAttention.*
    * **Compute-Bound ($I \ge R$):** Saturating SM compute. *Action: Leverage FP16/BF16/FP8 Tensor Cores, ensure matrix dimensions are multiples of 8 or 16.*
    * **Latency-Limited ($< 30\%$ of Roofline):** Under-utilizing SMs. *Action: Check low warp occupancy, tail effects from small grid launches, or host-launch serialization.*

---

## System Architecture

```text
llm-pipeline-profiler/
│
├── sampler-cpp/                           # High-Performance C++/CUDA Prober & Sampler
│   ├── CMakeLists.txt                     # nvcc CUDA compiler support + fallback
│   ├── src/
│   │   ├── hardware_prober.cu / .cuh      # Peak FMA TFLOPS & DRAM bandwidth microbenchmarks
│   │   ├── nvml_reader.cpp / .h           # Device clock and memory bandwidth reader
│   │   ├── proc_reader.cpp / .h           # Host CPU & RAM reader
│   │   └── ipc_server.cpp / .h            # UDS telemetry streaming
│   └── bindings/
│       └── pybind_wrapper.cpp             # Exposes probe_hardware() to Python
│
├── profiler-lib/                          # Python Instrumentation Package
│   ├── pyproject.toml
│   └── llm_profiler/
│       ├── tracer.py                      # Stage orchestration & operator profiling hooks
│       ├── schema.py                      # OperatorMetric & HardwareCeilings data contracts
│       ├── exporter.py                    # Dual export: JSON file + HTTP POST
│       ├── collectors/
│       │   ├── roofline_collector.py      # cudaEvent timer & FLOP / Byte tracker
│       │   ├── cpu_mem.py
│       │   ├── gpu.py
│       │   └── torch_profiler.py
│       └── analyzers/
│           └── roofline_analyzer.py       # Ridge point calculator & DevTech classifier
│
├── dashboard/                             # Next.js 16 + Prisma Observability Platform
│   ├── prisma/
│   │   └── schema.prisma                  # OperatorMetric & hardware ceiling storage
│   └── src/
│       ├── app/
│       │   ├── api/runs/[id]/roofline/    # Endpoint serving operator (I, P) coordinates
│       │   └── dashboard/[runId]/page.tsx # Tabbed UI: Roofline (v2.0), Timelines, Flamegraph
│       └── components/
│           ├── RooflineChart.tsx          # Interactive Log-Log Roofline chart with DevTech diagnostics
│           ├── Timeline.tsx
│           ├── Flamegraph.tsx
│           ├── MemoryChart.tsx
│           └── ThroughputChart.tsx
│
└── examples/
    ├── kaggle_notebook_example.ipynb      # End-to-end LLM pipeline run
    └── roofline_operator_benchmark.py     # Standalone operator benchmark suite
```

---

## Quickstart

### 1. Start Database & Dashboard

```bash
docker compose up db -d
cd dashboard
npm install
npx prisma db push
npm run dev
```

Dashboard runs locally at `http://localhost:3000`.

### 2. Install Profiler Library

```bash
pip install -e ./profiler-lib
```

### 3. Run the GPU Roofline Benchmark

Run the standalone operator benchmark suite for GEMM, Multi-Head Attention (Naive vs. FlashAttention), RMSNorm, and SwiGLU:

```bash
python examples/roofline_operator_benchmark.py --dashboard-url http://localhost:3000
```

---

## Usage in Python / PyTorch Code

### 1. Profiling Operators Against the Roofline Model

```python
from llm_profiler import Tracer
from llm_profiler.analyzers.roofline_analyzer import RooflineAnalyzer

# Initialize Tracer (auto-probes GPU hardware ceilings)
tracer = Tracer(
    run_name="llama3-inference-run",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    dashboard_url="http://localhost:3000",
)

# Profile a stage with individual operator metrics
with tracer.stage("prefill_stage") as stage:
    m, k, n = 32, 4096, 4096
    gemm_est = RooflineAnalyzer.estimate_gemm(m, k, n, dtype_bytes=2)

    # High-precision CUDA event operator profiling
    with stage.profile_op(
        op_name="qkv_projection_gemm",
        category="compute_dense",
        flops=gemm_est["flops"],
        bytes_transferred=gemm_est["bytes_transferred"],
    ):
        output = model.qkv_proj(hidden_states)

    # Benchmark Attention (evaluates Naive vs. FlashAttention)
    attn_est = RooflineAnalyzer.estimate_attention(
        batch_size=2, num_heads=32, seq_len=512, head_dim=128, fused=True
    )
    with stage.profile_op(
        op_name="flash_attention_sdpa",
        category="attention",
        flops=attn_est["flops"],
        bytes_transferred=attn_est["bytes_transferred"],
    ):
        attn_out = flash_attn_func(q, k, v)

# Export all metrics, stage breakdowns, and roofline scatter points
tracer.export()
```

### 2. Auto-Instrumentation (Zero-Code HuggingFace Tracking)

```python
from llm_profiler import Tracer
from transformers import AutoModelForCausalLM, AutoTokenizer

tracer = Tracer(run_name="llama3-auto", dashboard_url="http://localhost:3000")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

tracer.export()
```

---

## Test & Verification Suites

Execute tests covering all layers without requiring external dependencies:

```bash
# Hardware Prober & Ridge Point tests
python test_hardware_prober.py

# Williams Roofline model & DevTech classifier unit tests
python test_roofline_analyzer.py

# End-to-end simulated run with stages, operators, and dashboard export
python test_combined_simulation.py

# C++ system sampler socket bindings
python test_cpp_sampler.py
```

---

## Tech Stack

* **Hardware & Profiling Engine:** C++17, CUDA (`nvcc`), pybind11, PyTorch (`cudaEvent`, `torch.profiler`), NVML
* **Python Telemetry Package:** Python 3.8+, Pydantic v2, psutil
* **Dashboard & Visualizer:** Next.js 16 (App Router), React 19, TypeScript, Recharts, Tailwind CSS v4, Lucide Icons
* **Data Storage & ORM:** PostgreSQL, Prisma ORM, Docker Compose

---

## License

MIT
