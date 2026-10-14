

# Product Requirement Document (PRD): LLM Pipeline & Operator Roofline Profiler (v2.0)

## 1. Document Overview & Objective

* **Project Name:** LLM Pipeline & Deep Learning Operator Roofline Profiler
* **Base Repository:** `llm-pipeline-profiler` (Extending existing `sampler-cpp`, `profiler-lib`, and `dashboard`)
* **Objective:** Upgrade the existing end-to-end LLM pipeline profiler into an institutional-grade **GPU bottleneck diagnostic engine**. In addition to macro-level stage timelines, CPU/RAM/VRAM metrics, and PyTorch operator traces, the system will discover hardware peak limits (TFLOPS & GB/s), compute exact operational intensity ($\text{FLOPs}/\text{Byte}$), plot empirical log-log **Roofline Models**, and automatically classify operator execution bounds (compute-bound vs. memory-bandwidth-bound vs. latency-limited).
* **Target Audience & Role Alignment:** NVIDIA Compute Developer Technology (DevTech), Compute Architecture (Deep Learning), and High-Performance AI Systems Engineering.

---

## 2. Target Scope & Workloads

### 2.1 Hardware Architecture Target
* Any NVIDIA GPU (Turing, Ampere, Ada Lovelace, Hopper, Blackwell; e.g., T4/V100/A100/H100/RTX 30xx/40xx).
* Operates seamlessly across local environments and remote cloud kernels (Kaggle, Colab, dedicated instances).

### 2.2 Deep Learning Operator Suite (Transformer & CNN Workloads)
The telemetry engine evaluates individual sub-operators captured during forward/backward or inference passes:
1. **Elementwise & Activations:** GELU, SiLU, ReLU, SwiGLU.
2. **Reduction & Normalization:** LayerNorm, RMSNorm, Softmax.
3. **Compute-Dense Primitives:** Dense Linear / GEMM ($M, N, K$), Batched MatMul (BMM), Convolution ($N, C, H, W, K, R, S$).
4. **Attention Mechanisms:**
   * Standard PyTorch Scaled Dot-Product Attention (Naive Multi-Head Attention — memory-bound due to materializing $S \times S$ score matrices).
   * Fused/Tiled Attention (FlashAttention / SDPA — compute-bound through SRAM tiling and online softmax).

---

## 3. Technical Requirements & Engineering Architecture

The upgrade integrates directly into the three existing sub-systems of `llm-pipeline-profiler`:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE (v2.0)                                │
│                                                                                        │
│   [sampler-cpp] (C++ / CUDA / NVML)                                                    │
│     ├── nvml_reader (device specs & live clocks)                                       │
│     ├── hardware_prober.cu (microbenchmarks: Peak FMA TFLOPS & Peak DRAM GB/s)         │
│     └── ipc_server (streams calibrated hardware limits to Python tracer via UDS)       │
│                                      │                                                 │
│                                      ▼                                                 │
│   [profiler-lib] (Python / PyTorch / CUDA Streams)                                     │
│     ├── tracer.py (stage boundaries + torch.cuda.synchronize)                          │
│     ├── collectors/roofline_collector.py (operator timings, FLOPs, and byte counters) │
│     ├── analyzers/roofline_analyzer.py (calculates I = FLOPs/B & classifies bounds)    │
│     └── exporter.py (packages unified Run schema + Roofline payload to JSON/HTTP)     │
│                                      │                                                 │
│                                      ▼                                                 │
│   [dashboard] (Next.js / PostgreSQL / Prisma)                                          │
│     ├── Stage Timelines & Throughput Curves (Existing)                                 │
│     ├── Op-Level CUDA Flamegraphs (Existing)                                           │
│     └── [NEW] Interactive Roofline View (Log-log chart, operator scatter, diagnostics) │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Module 1: Hardware Peak Discovery Engine (`sampler-cpp`)

Extends the existing C++ engine to calibrate theoretical and empirical hardware ceilings.

#### 1. Compute Ceiling ($P_{\text{peak}}$):
* Executes a dedicated CUDA microbenchmark (`hardware_prober.cu`) issuing continuous, unrolled fused multiply-add (FMA) instructions to saturate SM FP32 cores, and optionally half-precision (FP16/BF16) tensor pipelines.
* Calculates empirical peak compute:
  $$P_{\text{peak}} = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Elapsed Time (seconds)}} \times 10^{-12} \quad [\text{TFLOPS}]$$

#### 2. Memory Bandwidth Ceiling ($B_{\text{peak}}$):
* Executes a vectorized memory microbenchmark issuing continuous 128-bit streaming reads/writes (`float4`) across DRAM to measure sustained bus saturation.
* Calculates empirical peak bandwidth:
  $$B_{\text{peak}} = \frac{\text{Bytes Read} + \text{Bytes Written}}{\text{Elapsed Time (seconds)}} \times 10^{-9} \quad [\text{GB/s}]$$

#### 3. Ridge Point / Balance Point Calculation:
* Automatically calculates the architectural knee/ridge point:
  $$\text{Ridge Point} = \frac{P_{\text{peak}} \text{ (GFLOPS)}}{B_{\text{peak}} \text{ (GB/s)}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$
  *(Workloads with arithmetic intensity below this value are fundamentally memory-bandwidth bound; workloads above it are compute-bound).*

---

### Module 2: Operator Telemetry & Operational Intensity Collector (`profiler-lib`)

Integrated into the existing `llm_profiler` Python package as `llm_profiler/collectors/roofline_collector.py`.

#### 1. High-Precision Timing:
* Uses native `cudaEventRecord` pairs on dedicated CUDA streams with warmup passes and statistical outlier rejection (interquartile range trimming over 100 runs) to eliminate host-launch overhead.

#### 2. Algorithmic FLOP Counter:
Analytical formula dispatch based on tensor shapes:
* **Linear / GEMM ($M \times K \times N$):**
  $$\text{FLOPs} = 2 \times M \times N \times K$$
* **Multi-Head Attention ($B, H, S, D$):**
  $$\text{FLOPs}_{\text{Attention}} = 4 \times B \times H \times S^2 \times D \quad (\text{QK}^T \text{ and } \text{Score} \times V)$$
* **LayerNorm / RMSNorm ($N$ elements):**
  $$\text{FLOPs}_{\text{LayerNorm}} \approx 5 \times N \quad (\text{mean, variance, subtract, divide, scale/shift})$$

#### 3. DRAM Byte Counter:
* Tracks exact memory traffic using either **CUPTI (CUDA Profiling Tools Interface)** counters (`dram__bytes.sum`), PyTorch allocator peak memory stats, or analytical minimum transfers:
  $$\text{Bytes}_{\text{Transferred}} = \text{Bytes}_{\text{Read}} + \text{Bytes}_{\text{Written}}$$

#### 4. Operational Intensity ($I$):
* Evaluated per operator instance:
  $$I = \frac{\text{Algorithmic FLOPs}}{\text{Total DRAM Bytes Transferred}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$

---

### Module 3: Diagnostic Engine & Visualizer (`dashboard` + `profiler-lib`)

Extends the Next.js frontend with a dedicated **Roofline Model Tab** alongside Timelines and Flamegraphs.

#### 1. Interactive Roofline Visualization (Log-Log Scale):
* **Ceiling Curves:**
  * Diagonal ceiling: Performance limited by memory bandwidth ($P = I \times B_{\text{peak}}$).
  * Horizontal ceiling: Performance limited by compute capability ($P = P_{\text{peak}}$).
* **Plotted Data Points:** Each profiled operator plotted as a point $(I, P_{\text{achieved}})$ under varying batch sizes ($B \in [1, 8, 32, 128]$) and sequence lengths.

#### 2. Automated DevTech Diagnostic Engine:
The profiler automatically evaluates each operator's position and renders actionable optimization recommendations:

| Observed Region | Performance Characteristic | Diagnostic Output & DevTech Recommendation |
| :--- | :--- | :--- |
| **$I < \text{Ridge Point}$ and near sloped ceiling** | **Memory-Bandwidth Bound** | *"Operator saturates DRAM bandwidth. Arithmetic efficiency is bounded by memory bus. **Action:** Apply operator fusion (e.g., Fuse Bias+GELU into preceding GEMM), use shared memory tiling, or replace standard Attention with FlashAttention."* |
| **$I \ge \text{Ridge Point}$ and near flat ceiling** | **Compute Bound** | *"Operator saturates compute pipelines. Arithmetic intensity is optimal. **Action:** Leverage FP16/BF16 Tensor Cores, unroll innermost loops, or investigate quantization (FP8/INT4)."* |
| **Far below both ceilings ($< 30\%$ of roofline)** | **Latency / Under-Utilized** | *"Operator is latency-bound or severely under-utilizing SMs. **Action:** Check low warp occupancy, tail effects from small grid launches, excessive warp divergence, or host-launch serialization overhead."* |

---

## 4. Technical Constraints & Error Budgets

1. **Profiling Overhead Budget:** Profiling hooks and telemetry capture must not increase end-to-end execution latency by more than **$12\%$** during dynamic runs.
2. **Hardware Calibration Accuracy:** Microbenchmark-derived $P_{\text{peak}}$ and $B_{\text{peak}}$ must align within **$\le 5\%$** of published NVIDIA Whitepaper specifications for the target device (under maximum boost clock).
3. **Data Contract Consistency:** All newly captured Roofline data must seamlessly extend the existing JSON schema (`schema.py` and Prisma schema) without breaking backwards compatibility for standard timeline-only runs.

---

## 5. Deliverables & Updated Repository Layout

```text
llm-pipeline-profiler/
│
├── sampler-cpp/                           # High-Performance C++ Sampler Layer
│   ├── CMakeLists.txt                     # Updated with CUDA compiler (nvcc) support
│   ├── src/
│   │   ├── main.cpp
│   │   ├── nvml_reader.cpp / .h           # Device clock and memory bandwidth reader
│   │   ├── proc_reader.cpp / .h
│   │   ├── ipc_server.cpp / .h
│   │   └── hardware_prober.cu / .cuh      # [NEW] Peak FMA TFLOPS & DRAM bandwidth kernels
│   └── bindings/
│       └── pybind_wrapper.cpp             # Exposes probe_hardware() to Python
│
├── profiler-lib/                          # Python Instrumentation Package
│   ├── pyproject.toml
│   └── llm_profiler/
│       ├── __init__.py
│       ├── tracer.py                      # Updated to coordinate operator stage profiling
│       ├── schema.py                      # Extended with OperatorMetric and RooflineRun models
│       ├── exporter.py
│       ├── collectors/
│       │   ├── cpu_mem.py
│       │   ├── gpu.py
│       │   ├── torch_profiler.py
│       │   └── roofline_collector.py      # [NEW] cudaEvent timer, FLOP & Byte tracker
│       └── analyzers/
│           └── roofline_analyzer.py       # [NEW] Ridge point calculation & bottleneck classifier
│
├── dashboard/                             # Next.js 15 + Prisma Observability Platform
│   ├── prisma/
│   │   └── schema.prisma                  # [NEW] Added OperatorMetric and RooflineData models
│   └── src/
│       ├── app/
│       │   ├── api/runs/[id]/roofline/    # [NEW] Endpoint serving operator (I, P) coordinates
│       │   └── dashboard/[runId]/page.tsx # Updated with tab navigation
│       └── components/
│           ├── Timeline.tsx
│           ├── Flamegraph.tsx
│           ├── MemoryChart.tsx
│           └── RooflineChart.tsx          # [NEW] Interactive Log-Log Roofline chart with D3 / Recharts
│
├── examples/
│   ├── kaggle_notebook_example.ipynb      # End-to-end LLM pipeline run
│   └── roofline_operator_benchmark.py     # [NEW] Standalone benchmark script for Attention & GEMM
│
├── tests/
│   ├── test_cpp_sampler.py
│   ├── test_flamegraph.py
│   ├── test_hardware_prober.py            # [NEW] Validates peak TFLOPS & GB/s accuracy
│   └── test_roofline_analyzer.py          # [NEW] Tests FLOP counting and classification accuracy
│
├── docker-compose.yml
└── README.md
```

---

## 6. Resume Bullet Points Resulting from this Upgrade

By updating `llm-pipeline-profiler` with this PRD, your resume bullet points for this project become directly tailored to NVIDIA DevTech:

* *“Engineered a full-stack GPU performance profiler in C++ and Python, benchmarking arithmetic intensity ($\text{FLOPs}/\text{Byte}$) and memory traffic across core transformer operators against the theoretical **Roofline Model**.”*
* *“Developed CUDA microbenchmarks in C++ utilizing unrolled FMA pipelines and 128-bit vectorized memory loads (`float4`) to calibrate peak FP32 TFLOPS and DRAM bandwidth within 5% of device whitepaper limits.”*
* *“Diagnosed memory-bandwidth bottlenecks in un-fused multi-head attention versus FlashAttention mechanisms, automatically surfacing kernel fusion and shared memory tiling recommendations via a Next.js observability dashboard.”*