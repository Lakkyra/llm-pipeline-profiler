import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class MetricSchema(BaseModel):
    timestamp: datetime
    key: str
    value: float

class OperatorMetric(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage_id: Optional[str] = None
    op_name: str
    category: str = "compute_dense"  # "compute_dense", "attention", "norm", "elementwise"
    duration_ms: float
    flops: float
    bytes_transferred: float
    operational_intensity: float  # FLOPs/Byte (I)
    achieved_tflops: float        # Achieved compute throughput
    achieved_bandwidth_gb_s: float  # Achieved memory bandwidth
    bound_classification: str     # "memory_bound", "compute_bound", "latency_limited"
    recommendation: str           # Actionable DevTech optimization suggestion
    metadata: Optional[Dict[str, Any]] = None

class HardwareCeilings(BaseModel):
    device_name: str = "NVIDIA GPU"
    peak_tflops: float = 0.0
    peak_bandwidth_gb_s: float = 0.0
    ridge_point: float = 0.0  # FLOPs/Byte (P_peak / B_peak)
    precision: str = "FP32"
    is_cuda_calibrated: bool = False

class StageSchema(BaseModel):
    id: str
    name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    cpu_percent: float
    ram_used_mb: float
    gpu_util_percent: float
    gpu_mem_used_mb: float
    metrics: List[MetricSchema] = Field(default_factory=list)
    operators: List[OperatorMetric] = Field(default_factory=list)
    trace_ref: Optional[str] = None

class RunSchema(BaseModel):
    id: str
    name: str
    created_at: datetime
    model_name: Optional[str] = None
    hardware_info: Optional[Dict[str, Any]] = None
    hardware_ceilings: Optional[HardwareCeilings] = None
    stages: List[StageSchema] = Field(default_factory=list)
    operators: List[OperatorMetric] = Field(default_factory=list)

    def to_json(self) -> str:
        if hasattr(self, "model_dump_json"):
            return self.model_dump_json()
        return self.json()

    def to_dict(self) -> dict:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
