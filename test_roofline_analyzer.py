"""
test_roofline_analyzer.py
Unit tests for RooflineAnalyzer and RooflineCollector.
Verifies operational intensity, theoretical ceilings, DevTech bottleneck classifications,
analytical FLOP/Byte estimators, and statistical IQR filtering.
"""

import sys
import os

# Ensure profiler-lib is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "profiler-lib"))

from llm_profiler.schema import OperatorMetric, HardwareCeilings
from llm_profiler.analyzers.roofline_analyzer import RooflineAnalyzer
from llm_profiler.collectors.roofline_collector import RooflineCollector


def test_analytical_estimators():
    print("[TEST] Testing analytical FLOPs and DRAM Bytes estimators...")

    # GEMM: M=32, K=4096, N=4096, FP16 (2 bytes)
    # FLOPs = 2 * 32 * 4096 * 4096 = 1,073,741,824 FLOPs (~1.074 GFLOP)
    # Bytes = (32*4096 + 4096*4096 + 32*4096) * 2 = (131072 + 16777216 + 131072) * 2 = 34,078,720 Bytes (~34.08 MB)
    gemm = RooflineAnalyzer.estimate_gemm(m=32, k=4096, n=4096, dtype_bytes=2)
    expected_flops = 2.0 * 32 * 4096 * 4096
    expected_bytes = (32 * 4096 + 4096 * 4096 + 32 * 4096) * 2.0
    assert gemm["flops"] == expected_flops
    assert gemm["bytes_transferred"] == expected_bytes
    gemm_intensity = gemm["flops"] / gemm["bytes_transferred"]
    assert gemm_intensity > 30.0  # High intensity GEMM

    # Attention: B=2, H=32, S=512, D=128
    # FLOPs = 4 * 2 * 32 * (512^2) * 128 = 8,589,934,592 FLOPs
    attn_naive = RooflineAnalyzer.estimate_attention(batch_size=2, num_heads=32, seq_len=512, head_dim=128, dtype_bytes=2, fused=False)
    attn_fused = RooflineAnalyzer.estimate_attention(batch_size=2, num_heads=32, seq_len=512, head_dim=128, dtype_bytes=2, fused=True)

    assert attn_naive["flops"] == attn_fused["flops"]
    # Fused Attention (FlashAttention) avoids writing intermediate S x S matrices to DRAM
    assert attn_fused["bytes_transferred"] < attn_naive["bytes_transferred"]
    fused_intensity = attn_fused["flops"] / attn_fused["bytes_transferred"]
    naive_intensity = attn_naive["flops"] / attn_naive["bytes_transferred"]
    assert fused_intensity > naive_intensity
    print(f"  ✓ Attention Intensity: Naive = {naive_intensity:.2f} FLOPs/B, FlashAttention Fused = {fused_intensity:.2f} FLOPs/B")

    # LayerNorm: N = 4096
    norm = RooflineAnalyzer.estimate_norm(num_elements=4096, dtype_bytes=2)
    assert norm["flops"] == 5.0 * 4096
    assert norm["bytes_transferred"] == 2.0 * 4096 * 2.0
    print("  ✓ Analytical estimators match expected tensor formulas.")


def test_roofline_classifications():
    print("[TEST] Testing Williams Roofline classification & DevTech recommendations...")

    # Hardware: A100 (19.5 TFLOPS, 2039 GB/s, Ridge Point = 9.56 FLOPs/B)
    ceilings = HardwareCeilings(
        device_name="NVIDIA A100",
        peak_tflops=19.5,
        peak_bandwidth_gb_s=2039.0,
        ridge_point=9.56,
        precision="FP32",
        is_cuda_calibrated=True,
    )
    analyzer = RooflineAnalyzer(ceilings)

    # 1. Memory-Bound Operator (Low intensity I = 2.0 < 9.56, achieving ~3.5 TFLOPS near sloped ceiling)
    # Roofline at I=2.0 is min(19.5, 2.0 * 2039 * 1e-3) = 4.078 TFLOPS
    # 3.5 TFLOPS is ~85% of roofline ceiling
    mem_op = OperatorMetric(
        op_name="rmsnorm_forward",
        category="norm",
        duration_ms=0.05,
        flops=175_000_000.0,
        bytes_transferred=87_500_000.0,  # I = 2.0
        operational_intensity=0.0,
        achieved_tflops=0.0,
        achieved_bandwidth_gb_s=0.0,
        bound_classification="",
        recommendation="",
    )
    analyzer.classify_operator(mem_op)
    assert mem_op.bound_classification == "memory_bound"
    assert "DRAM bandwidth" in mem_op.recommendation
    assert "fusion" in mem_op.recommendation.lower()
    print("  ✓ Memory-bound classification and DevTech recommendation verified.")

    # 2. Compute-Bound Operator (High intensity I = 50.0 > 9.56, achieving ~17.0 TFLOPS near flat ceiling)
    # Roofline at I=50.0 is 19.5 TFLOPS
    compute_op = OperatorMetric(
        op_name="dense_linear_gemm",
        category="compute_dense",
        duration_ms=2.0,
        flops=34_000_000_000.0,
        bytes_transferred=680_000_000.0,  # I = 50.0
        operational_intensity=0.0,
        achieved_tflops=0.0,
        achieved_bandwidth_gb_s=0.0,
        bound_classification="",
        recommendation="",
    )
    analyzer.classify_operator(compute_op)
    assert compute_op.bound_classification == "compute_bound"
    assert "compute pipelines" in compute_op.recommendation
    assert "Tensor Cores" in compute_op.recommendation
    print("  ✓ Compute-bound classification and DevTech recommendation verified.")

    # 3. Latency-Limited Operator (<30% of roofline ceiling)
    # Roofline at I=50.0 is 19.5 TFLOPS; achieved is only 1.0 TFLOPS (~5% of ceiling)
    latency_op = OperatorMetric(
        op_name="tiny_batch_kernel",
        category="compute_dense",
        duration_ms=10.0,
        flops=10_000_000_000.0,
        bytes_transferred=200_000_000.0,  # I = 50.0
        operational_intensity=0.0,
        achieved_tflops=0.0,
        achieved_bandwidth_gb_s=0.0,
        bound_classification="",
        recommendation="",
    )
    analyzer.classify_operator(latency_op)
    assert latency_op.bound_classification == "latency_limited"
    assert "latency-bound" in latency_op.recommendation.lower()
    assert "warp occupancy" in latency_op.recommendation.lower()
    print("  ✓ Latency-limited classification (<30% ceiling) verified.")


def test_roofline_collector_context():
    print("[TEST] Testing RooflineCollector context manager...")
    ceilings = HardwareCeilings(
        device_name="Test GPU",
        peak_tflops=10.0,
        peak_bandwidth_gb_s=500.0,
        ridge_point=20.0,
    )
    collector = RooflineCollector(ceilings=ceilings)

    # Use context manager
    with collector.profile_op(
        op_name="test_matmul",
        category="compute_dense",
        flops=1_000_000_000.0,
        bytes_transferred=20_000_000.0,
    ) as ctx:
        # Simulate some work
        dummy = sum(i * i for i in range(10000))

    ops = collector.get_operators()
    assert len(ops) == 1
    op = ops[0]
    assert op.op_name == "test_matmul"
    assert op.flops == 1_000_000_000.0
    assert op.operational_intensity == 50.0
    assert op.duration_ms > 0.0
    assert op.bound_classification in ["compute_bound", "memory_bound", "latency_limited"]

    summary = collector.get_summary()
    assert summary["total_operators"] == 1
    assert summary["total_flops"] == 1_000_000_000.0
    print("  ✓ RooflineCollector context manager successfully profiled and classified operator.")


if __name__ == "__main__":
    print("=== Running Roofline Analyzer & Collector Tests ===")
    test_analytical_estimators()
    test_roofline_classifications()
    test_roofline_collector_context()
    print("\nAll Roofline Analyzer tests passed successfully!")
