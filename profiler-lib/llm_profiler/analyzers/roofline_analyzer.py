"""
Roofline Analyzer & Diagnostic Engine for llm-pipeline-profiler v2.0.

Provides architectural ceiling modeling, operational intensity calculations,
and automated DevTech optimization recommendations.
"""

from typing import List, Optional, Dict, Any, Tuple
from ..schema import OperatorMetric, HardwareCeilings


class RooflineAnalyzer:
    """
    Analyzes operator execution metrics against theoretical and empirical hardware ceilings.
    Implements Williams et al. Roofline Model with DevTech classification heuristics.
    """

    def __init__(self, ceilings: Optional[HardwareCeilings] = None):
        self.ceilings = ceilings or HardwareCeilings()

    def set_ceilings(self, ceilings: HardwareCeilings) -> None:
        self.ceilings = ceilings

    @staticmethod
    def calculate_ridge_point(peak_tflops: float, peak_bandwidth_gb_s: float) -> float:
        """
        Ridge Point R = (Peak GFLOPS) / (Peak GB/s) = (Peak TFLOPS * 1000) / Peak GB/s [FLOPs/Byte].
        """
        if peak_bandwidth_gb_s <= 0.0:
            return 0.0
        return (peak_tflops * 1000.0) / peak_bandwidth_gb_s

    def roofline_ceiling(self, operational_intensity: float) -> float:
        """
        Calculates theoretical performance ceiling P_roof(I) in TFLOPS at given intensity I:
        P_roof(I) = min(P_peak, I * B_peak * 1e-3)
        """
        if not self.ceilings or self.ceilings.peak_tflops <= 0.0:
            return 0.0
        memory_ceiling_tflops = operational_intensity * self.ceilings.peak_bandwidth_gb_s * 1e-3
        return min(self.ceilings.peak_tflops, memory_ceiling_tflops)

    def classify_operator(self, op: OperatorMetric) -> OperatorMetric:
        """
        Evaluates an operator metric, calculates operational intensity, achieved throughput,
        bound classification, and DevTech recommendation.
        """
        # Duration in seconds
        elapsed_s = max(op.duration_ms / 1000.0, 1e-9)

        # Operational Intensity I = FLOPs / max(bytes, 1)
        bytes_transferred = max(float(op.bytes_transferred), 1.0)
        flops = max(float(op.flops), 0.0)
        intensity = flops / bytes_transferred
        op.operational_intensity = intensity

        # Achieved TFLOPS = (FLOPs / elapsed_s) * 1e-12 = FLOPs / (duration_ms * 1e9)
        achieved_tflops = (flops / elapsed_s) / 1e12
        op.achieved_tflops = achieved_tflops

        # Achieved Bandwidth (GB/s) = (bytes / elapsed_s) * 1e-9
        achieved_bandwidth = (bytes_transferred / elapsed_s) / 1e9
        op.achieved_bandwidth_gb_s = achieved_bandwidth

        # Bound classification and recommendations
        if self.ceilings and self.ceilings.peak_tflops > 0.0 and self.ceilings.peak_bandwidth_gb_s > 0.0:
            roof_tflops = self.roofline_ceiling(intensity)
            ridge = self.ceilings.ridge_point
            if ridge <= 0.0:
                ridge = self.calculate_ridge_point(self.ceilings.peak_tflops, self.ceilings.peak_bandwidth_gb_s)

            ratio = achieved_tflops / max(roof_tflops, 1e-6)

            if ratio < 0.30:
                op.bound_classification = "latency_limited"
                op.recommendation = (
                    "Operator is latency-bound or severely under-utilizing SMs (<30% of roofline ceiling). "
                    "Action: Check low warp occupancy, tail effects from small grid launches, excessive warp "
                    "divergence, or host-launch serialization overhead."
                )
            elif intensity < ridge:
                op.bound_classification = "memory_bound"
                op.recommendation = (
                    "Operator saturates DRAM bandwidth. Arithmetic efficiency is bounded by memory bus. "
                    "Action: Apply operator fusion (e.g., Fuse Bias+GELU into preceding GEMM), use shared memory "
                    "tiling, or replace standard Attention with FlashAttention."
                )
            else:
                op.bound_classification = "compute_bound"
                op.recommendation = (
                    "Operator saturates compute pipelines. Arithmetic intensity is optimal. "
                    "Action: Leverage FP16/BF16 Tensor Cores, unroll innermost loops, or investigate "
                    "quantization (FP8/INT4)."
                )
        else:
            # Fallback heuristic when no ceilings calibrated
            if intensity >= 15.0:
                op.bound_classification = "compute_bound"
                op.recommendation = "Arithmetic intensity is high. Ensure Tensor Cores and optimal matrix alignment (multiples of 8/16) are utilized."
            else:
                op.bound_classification = "memory_bound"
                op.recommendation = "Low arithmetic intensity indicates memory bandwidth pressure. Consider kernel fusion or activation caching."

        return op

    def analyze_operators(self, operators: List[OperatorMetric]) -> Tuple[List[OperatorMetric], Dict[str, Any]]:
        """
        Classifies all operators and computes aggregate roofline telemetry statistics.
        """
        classified = [self.classify_operator(op) for op in operators]
        summary = self.summarize(classified)
        return classified, summary

    def summarize(self, operators: List[OperatorMetric]) -> Dict[str, Any]:
        """
        Computes aggregate diagnostic summary over a list of classified operators.
        """
        total_flops = sum(op.flops for op in operators)
        total_bytes = sum(op.bytes_transferred for op in operators)
        total_duration_ms = sum(op.duration_ms for op in operators)
        elapsed_s = max(total_duration_ms / 1000.0, 1e-9)

        overall_intensity = total_flops / max(total_bytes, 1.0)
        overall_tflops = (total_flops / elapsed_s) / 1e12 if total_flops > 0 else 0.0

        memory_bound_count = sum(1 for op in operators if op.bound_classification == "memory_bound")
        compute_bound_count = sum(1 for op in operators if op.bound_classification == "compute_bound")
        latency_limited_count = sum(1 for op in operators if op.bound_classification == "latency_limited")

        # Top bottlenecks by execution time
        bottlenecks = sorted(operators, key=lambda x: x.duration_ms, reverse=True)[:5]

        return {
            "total_operators": len(operators),
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "total_duration_ms": total_duration_ms,
            "overall_intensity": overall_intensity,
            "overall_tflops": overall_tflops,
            "counts": {
                "memory_bound": memory_bound_count,
                "compute_bound": compute_bound_count,
                "latency_limited": latency_limited_count,
            },
            "bottlenecks": [
                {
                    "op_name": b.op_name,
                    "category": b.category,
                    "duration_ms": b.duration_ms,
                    "bound": b.bound_classification,
                    "achieved_tflops": b.achieved_tflops,
                    "operational_intensity": b.operational_intensity,
                }
                for b in bottlenecks
            ],
        }

    # =========================================================================
    # Analytical FLOP and Byte Estimators
    # =========================================================================

    @staticmethod
    def estimate_gemm(m: int, k: int, n: int, dtype_bytes: int = 2) -> Dict[str, float]:
        """
        Analytical FLOPs and DRAM Bytes for Linear / GEMM (M x K) x (K x N).
        FLOPs = 2 * M * K * N
        DRAM Bytes = (M*K + K*N + M*N) * dtype_bytes
        """
        flops = 2.0 * float(m) * float(k) * float(n)
        bytes_transferred = float((m * k + k * n + m * n) * dtype_bytes)
        return {"flops": flops, "bytes_transferred": bytes_transferred}

    @staticmethod
    def estimate_attention(
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype_bytes: int = 2,
        fused: bool = False,
    ) -> Dict[str, float]:
        """
        Analytical FLOPs and DRAM Bytes for Multi-Head Attention.
        FLOPs = 4 * B * H * S^2 * D  (QK^T is 2BH(S^2)D and Score*V is 2BH(S^2)D).
        DRAM Bytes:
          - Fused (FlashAttention): Reads Q, K, V (3*B*H*S*D) and writes O (B*H*S*D) = 4*B*H*S*D * dtype
          - Naive: Also materializes S and P (2 * B*H*S^2) = (4*B*H*S*D + 2*B*H*S^2) * dtype
        """
        b = float(batch_size)
        h = float(num_heads)
        s = float(seq_len)
        d = float(head_dim)

        flops = 4.0 * b * h * (s ** 2) * d

        if fused:
            bytes_transferred = 4.0 * b * h * s * d * float(dtype_bytes)
        else:
            bytes_transferred = (4.0 * b * h * s * d + 2.0 * b * h * (s ** 2)) * float(dtype_bytes)

        return {"flops": flops, "bytes_transferred": bytes_transferred}

    @staticmethod
    def estimate_norm(num_elements: int, dtype_bytes: int = 2) -> Dict[str, float]:
        """
        Analytical FLOPs and DRAM Bytes for LayerNorm / RMSNorm.
        FLOPs = 5 * N (mean, variance, subtract, divide, scale/shift)
        DRAM Bytes = 2 * N * dtype_bytes (Read input, write normalized output)
        """
        n = float(num_elements)
        flops = 5.0 * n
        bytes_transferred = 2.0 * n * float(dtype_bytes)
        return {"flops": flops, "bytes_transferred": bytes_transferred}

    @staticmethod
    def estimate_elementwise(
        num_elements: int,
        flops_per_element: float = 1.0,
        inputs: int = 1,
        outputs: int = 1,
        dtype_bytes: int = 2,
    ) -> Dict[str, float]:
        """
        Analytical FLOPs and DRAM Bytes for Elementwise operations (e.g., GELU, SiLU, Residual Add).
        """
        n = float(num_elements)
        flops = n * float(flops_per_element)
        bytes_transferred = n * float(inputs + outputs) * float(dtype_bytes)
        return {"flops": flops, "bytes_transferred": bytes_transferred}
