import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { RooflineDataResponse, OperatorMetric } from "@/types/run";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;

    const run = await db.run.findUnique({
      where: { id },
      include: {
        operators: {
          orderBy: { operational_intensity: "asc" },
        },
        stages: {
          include: {
            operators: true,
          },
        },
      },
    });

    if (!run) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }

    // Consolidate operators from run-level and stage-level
    const operatorsMap = new Map<string, OperatorMetric>();
    for (const op of run.operators) {
      operatorsMap.set(op.id, op as OperatorMetric);
    }
    for (const st of run.stages) {
      for (const op of st.operators) {
        if (!operatorsMap.has(op.id)) {
          operatorsMap.set(op.id, op as OperatorMetric);
        }
      }
    }
    const operators = Array.from(operatorsMap.values()).sort(
      (a, b) => a.operational_intensity - b.operational_intensity
    );

    // Default hardware ceilings if none were calibrated
    const peak_tflops = run.peak_tflops || 15.0; // Default TFLOPS (e.g., T4/V100 scale)
    const peak_bandwidth_gb_s = run.peak_bandwidth_gb_s || 600.0; // Default GB/s
    const ridge_point = run.ridge_point || (peak_tflops * 1000.0) / peak_bandwidth_gb_s;
    const device_name = run.device_name || (run.hardware_info as any)?.gpu_type || "NVIDIA GPU";

    // Compute summary metrics
    const compute_bound_count = operators.filter((o) => o.bound_classification === "compute_bound").length;
    const memory_bound_count = operators.filter((o) => o.bound_classification === "memory_bound").length;
    const latency_limited_count = operators.filter((o) => o.bound_classification === "latency_limited").length;

    const total_ops = operators.length;
    const avg_operational_intensity = total_ops > 0
      ? operators.reduce((acc, o) => acc + o.operational_intensity, 0) / total_ops
      : 0;
    const avg_achieved_tflops = total_ops > 0
      ? operators.reduce((acc, o) => acc + o.achieved_tflops, 0) / total_ops
      : 0;

    const response: RooflineDataResponse = {
      run_id: run.id,
      run_name: run.name,
      device_name,
      peak_tflops,
      peak_bandwidth_gb_s,
      ridge_point,
      operators,
      summary: {
        total_operators: total_ops,
        compute_bound_count,
        memory_bound_count,
        latency_limited_count,
        avg_operational_intensity,
        avg_achieved_tflops,
      },
    };

    return NextResponse.json(response);
  } catch (error: any) {
    const { id } = await params;
    console.error(`Error in GET /api/runs/${id}/roofline:`, error);
    return NextResponse.json({ error: error.message || "Failed to fetch roofline data" }, { status: 500 });
  }
}
