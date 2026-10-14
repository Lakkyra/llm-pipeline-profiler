export interface Metric {
  timestamp: string; // ISO 8601 string
  key: string;
  value: number;
}

export type BoundClassification = "memory_bound" | "compute_bound" | "latency_limited";

export interface OperatorMetric {
  id: string;
  run_id?: string;
  stage_id?: string | null;
  op_name: string;
  category: "compute_dense" | "attention" | "norm" | "elementwise" | string;
  duration_ms: number;
  flops: number;
  bytes_transferred: number;
  operational_intensity: number; // FLOPs/Byte (I)
  achieved_tflops: number;
  achieved_bandwidth_gb_s: number;
  bound_classification: BoundClassification;
  recommendation: string;
  metadata?: Record<string, any> | null;
  created_at?: string;
}

export interface HardwareCeilings {
  device_name: string;
  peak_tflops: number;
  peak_bandwidth_gb_s: number;
  ridge_point: number;
  precision?: string;
  is_cuda_calibrated?: boolean;
}

export interface Stage {
  id: string;
  name: string;
  start_time: string; // ISO 8601 string
  end_time: string; // ISO 8601 string
  duration_ms: number;
  cpu_percent: number;
  ram_used_mb: number;
  gpu_util_percent: number;
  gpu_mem_used_mb: number;
  metrics: Metric[];
  operators?: OperatorMetric[];
  trace_ref?: string | null;
  has_trace?: boolean;
}

export interface Run {
  id: string;
  name: string;
  created_at: string; // ISO 8601 string
  model_name?: string | null;
  hardware_info?: {
    gpu_type?: string | null;
    gpu_vram_mb?: number | null;
    cpu_count?: number | null;
  } | null;
  peak_tflops?: number | null;
  peak_bandwidth_gb_s?: number | null;
  ridge_point?: number | null;
  device_name?: string | null;
  hardware_ceilings?: HardwareCeilings | null;
  stages: Stage[];
  operators?: OperatorMetric[];
}

export interface RooflineDataResponse {
  run_id: string;
  run_name: string;
  device_name: string;
  peak_tflops: number;
  peak_bandwidth_gb_s: number;
  ridge_point: number;
  operators: OperatorMetric[];
  summary: {
    total_operators: number;
    compute_bound_count: number;
    memory_bound_count: number;
    latency_limited_count: number;
    avg_operational_intensity: number;
    avg_achieved_tflops: number;
  };
}
