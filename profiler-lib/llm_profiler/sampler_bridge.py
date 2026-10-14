import socket
import json
import os
import time
import tempfile
from typing import List, Dict, Any

try:
    import llm_profiler_sampler
    HAS_CPP_SAMPLER = True
except ImportError:
    HAS_CPP_SAMPLER = False

class SamplerBridge:
    def __init__(self, socket_path: str = None, sample_interval_ms: int = 50):
        if socket_path is None:
            socket_path = os.path.join(tempfile.gettempdir(), "llm_profiler_sampler.sock")
        self.socket_path = socket_path
        self.sample_interval_ms = sample_interval_ms
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        """
        Attempts to connect to the Unix domain socket after starting C++ sampler if available.
        Returns True if successful, False otherwise.
        """
        if not hasattr(socket, "AF_UNIX"):
            self.connected = False
            return False
            
        if HAS_CPP_SAMPLER:
            try:
                llm_profiler_sampler.start_sampler(self.socket_path, self.sample_interval_ms)
                # Give background thread time to bind socket
                time.sleep(0.05)
            except Exception as e:
                print(f"[llm-profiler] Failed to start C++ sampler extension: {e}")
                self.connected = False
                return False

        if not os.path.exists(self.socket_path):
            self.connected = False
            return False
            
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(self.socket_path)
            self.socket.settimeout(0.01)  # Set low timeout to prevent blocking the tracer
            self.connected = True
            return True
        except Exception:
            if HAS_CPP_SAMPLER:
                try:
                    llm_profiler_sampler.stop_sampler()
                except Exception:
                    pass
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
            return False

    def read_samples(self) -> List[Dict[str, Any]]:
        """
        Reads any pending samples from the socket.
        """
        if not self.connected or not self.socket:
            return []
            
        samples = []
        try:
            data = self.socket.recv(4096)
            if data:
                lines = data.decode('utf-8').split('\n')
                for line in lines:
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except socket.timeout:
            pass
        except Exception:
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except Exception:
                    pass
                self.socket = None
        return samples

    def close(self):
        """
        Closes the socket connection.
        """
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            
        self.connected = False

        if HAS_CPP_SAMPLER:
            try:
                llm_profiler_sampler.stop_sampler()
            except Exception:
                pass

    @staticmethod
    def probe_hardware(device_id: int = 0) -> Dict[str, Any]:
        """
        Calls the C++ microbenchmark engine to calibrate peak TFLOPS and DRAM GB/s,
        falling back to built-in device whitepaper specifications if C++ extension is unavailable.
        """
        if HAS_CPP_SAMPLER and hasattr(llm_profiler_sampler, "probe_hardware"):
            try:
                return llm_profiler_sampler.probe_hardware(device_id)
            except Exception as e:
                print(f"[llm-profiler] C++ hardware probe error: {e}")

        # Fallback Python database
        return SamplerBridge.lookup_hardware_specs()

    @staticmethod
    def lookup_hardware_specs(gpu_name: str = "") -> Dict[str, Any]:
        if HAS_CPP_SAMPLER and hasattr(llm_profiler_sampler, "lookup_hardware_specs"):
            try:
                return llm_profiler_sampler.lookup_hardware_specs(gpu_name)
            except Exception:
                pass

        lower_name = (gpu_name or "").lower()
        
        # Detect via PyTorch if name not given
        if not gpu_name:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    lower_name = gpu_name.lower()
            except ImportError:
                pass

        if "h100" in lower_name:
            tflops, bw = 67.0, 3350.0
            device_name = gpu_name or "NVIDIA H100 SXM5"
        elif "a100" in lower_name:
            tflops, bw = 19.5, 1555.0
            device_name = gpu_name or "NVIDIA A100 SXM4"
        elif "v100" in lower_name:
            tflops, bw = 15.7, 900.0
            device_name = gpu_name or "NVIDIA V100 SXM2"
        elif "t4" in lower_name:
            tflops, bw = 8.1, 320.0
            device_name = gpu_name or "NVIDIA Tesla T4"
        elif "4090" in lower_name:
            tflops, bw = 82.6, 1008.0
            device_name = gpu_name or "NVIDIA GeForce RTX 4090"
        elif "3090" in lower_name:
            tflops, bw = 35.6, 936.0
            device_name = gpu_name or "NVIDIA GeForce RTX 3090"
        elif "3080" in lower_name:
            tflops, bw = 29.8, 760.0
            device_name = gpu_name or "NVIDIA GeForce RTX 3080"
        elif "a10g" in lower_name or "a10" in lower_name:
            tflops, bw = 31.2, 600.0
            device_name = gpu_name or "NVIDIA A10G"
        elif "l4" in lower_name:
            tflops, bw = 30.3, 300.0
            device_name = gpu_name or "NVIDIA L4"
        else:
            tflops, bw = 10.0, 400.0
            device_name = gpu_name or "Generic NVIDIA GPU"

        ridge = (tflops * 1000.0) / bw
        return {
            "device_name": device_name,
            "peak_tflops": tflops,
            "peak_bandwidth_gb_s": bw,
            "ridge_point": ridge,
            "precision": "FP32",
            "is_cuda_calibrated": False,
            "status_message": f"Resolved from hardware specifications database ({device_name})"
        }

