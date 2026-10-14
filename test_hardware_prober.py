"""
test_hardware_prober.py
Unit tests for the Hardware Peak Discovery Engine and GPU specs lookup.
Verifies hardware ceilings calibration, Ridge Point calculations, and whitepaper fallbacks.
"""

import sys
import os

# Ensure profiler-lib is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "profiler-lib"))

from llm_profiler.schema import HardwareCeilings
from llm_profiler.sampler_bridge import SamplerBridge
from llm_profiler.analyzers.roofline_analyzer import RooflineAnalyzer


def test_hardware_ceilings_schema():
    print("[TEST] Testing HardwareCeilings schema serialization...")
    ceilings = HardwareCeilings(
        device_name="NVIDIA A100-SXM4-80GB",
        peak_tflops=19.5,
        peak_bandwidth_gb_s=2039.0,
        ridge_point=9.56,
        precision="FP32",
        is_cuda_calibrated=True,
    )
    assert ceilings.device_name == "NVIDIA A100-SXM4-80GB"
    assert ceilings.peak_tflops == 19.5
    assert ceilings.peak_bandwidth_gb_s == 2039.0
    assert abs(ceilings.ridge_point - 9.56) < 0.01
    assert ceilings.is_cuda_calibrated is True
    print("  ✓ HardwareCeilings schema valid.")


def test_ridge_point_calculation():
    print("[TEST] Testing architectural Ridge Point formulas...")
    # A100 FP32: 19.5 TFLOPS, 2039 GB/s -> R = (19.5 * 1000) / 2039 = 9.56 FLOPs/Byte
    ridge_a100 = RooflineAnalyzer.calculate_ridge_point(19.5, 2039.0)
    assert abs(ridge_a100 - 9.563) < 0.05, f"Unexpected ridge point: {ridge_a100}"

    # H100 SXM5 FP32: 67 TFLOPS, 3350 GB/s -> R = (67 * 1000) / 3350 = 20.0 FLOPs/Byte
    ridge_h100 = RooflineAnalyzer.calculate_ridge_point(67.0, 3350.0)
    assert abs(ridge_h100 - 20.0) < 0.05, f"Unexpected ridge point: {ridge_h100}"

    # Zero bandwidth edge case handling
    ridge_zero = RooflineAnalyzer.calculate_ridge_point(10.0, 0.0)
    assert ridge_zero == 0.0
    print("  ✓ Ridge point calculation verified.")


def test_hardware_specs_lookup_database():
    print("[TEST] Testing whitepaper fallback specs database across NVIDIA architectures...")
    bridge = SamplerBridge()

    # Test NVIDIA H100
    h100 = bridge.lookup_hardware_specs("NVIDIA H100 PCIe")
    assert h100.peak_tflops == 51.0
    assert h100.peak_bandwidth_gb_s == 2000.0
    assert abs(h100.ridge_point - 25.5) < 0.1

    # Test NVIDIA A100
    a100 = bridge.lookup_hardware_specs("NVIDIA A100-PCIE-40GB")
    assert a100.peak_tflops == 19.5
    assert a100.peak_bandwidth_gb_s == 1555.0

    # Test NVIDIA RTX 4090
    rtx4090 = bridge.lookup_hardware_specs("NVIDIA GeForce RTX 4090")
    assert rtx4090.peak_tflops == 82.58
    assert rtx4090.peak_bandwidth_gb_s == 1008.0

    # Test NVIDIA T4
    t4 = bridge.lookup_hardware_specs("Tesla T4")
    assert t4.peak_tflops == 8.1
    assert t4.peak_bandwidth_gb_s == 320.0

    # Test Unknown fallback
    generic = bridge.lookup_hardware_specs("Unknown Accelerator XYZ")
    assert generic.peak_tflops > 0.0
    assert generic.peak_bandwidth_gb_s > 0.0
    print("  ✓ Hardware specs database accurately matches published NVIDIA whitepapers.")


def test_probe_hardware_fallback():
    print("[TEST] Testing probe_hardware() graceful fallback in non-CUDA environments...")
    bridge = SamplerBridge()
    probed = bridge.probe_hardware(device_id=0)
    assert isinstance(probed, HardwareCeilings)
    assert probed.peak_tflops > 0.0
    assert probed.peak_bandwidth_gb_s > 0.0
    assert probed.ridge_point > 0.0
    print(f"  ✓ probe_hardware() returned: {probed.device_name} ({probed.peak_tflops} TFLOPS, {probed.peak_bandwidth_gb_s} GB/s)")


if __name__ == "__main__":
    print("=== Running Hardware Prober & Ceilings Tests ===")
    test_hardware_ceilings_schema()
    test_ridge_point_calculation()
    test_hardware_specs_lookup_database()
    test_probe_hardware_fallback()
    print("\nAll Hardware Prober tests passed successfully!")
