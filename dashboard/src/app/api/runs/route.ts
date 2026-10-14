import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { sseManager } from "@/lib/sse";
import { Prisma } from "@prisma/client";

export async function POST(req: NextRequest) {
  try {
    const payload = await req.json();
    
    if (!payload.id || !payload.name) {
      return NextResponse.json({ error: "Missing required fields: id, name" }, { status: 400 });
    }

    // Extract hardware ceilings if provided (nested or top-level)
    const ceilings = payload.hardware_ceilings || {};
    const peak_tflops = payload.peak_tflops ?? ceilings.peak_tflops ?? null;
    const peak_bandwidth_gb_s = payload.peak_bandwidth_gb_s ?? ceilings.peak_bandwidth_gb_s ?? null;
    const ridge_point = payload.ridge_point ?? ceilings.ridge_point ?? null;
    const device_name = payload.device_name ?? ceilings.device_name ?? payload.hardware_info?.gpu_type ?? null;

    // Execute in a transaction to ensure atomicity
    const run = await db.$transaction(async (tx: Prisma.TransactionClient) => {
      // 1. Upsert Run
      await tx.run.upsert({
        where: { id: payload.id },
        create: {
          id: payload.id,
          name: payload.name,
          created_at: payload.created_at ? new Date(payload.created_at) : new Date(),
          model_name: payload.model_name,
          hardware_info: payload.hardware_info || {},
          peak_tflops: peak_tflops ? Number(peak_tflops) : null,
          peak_bandwidth_gb_s: peak_bandwidth_gb_s ? Number(peak_bandwidth_gb_s) : null,
          ridge_point: ridge_point ? Number(ridge_point) : null,
          device_name: device_name ? String(device_name) : null,
        },
        update: {
          name: payload.name,
          model_name: payload.model_name,
          hardware_info: payload.hardware_info || {},
          peak_tflops: peak_tflops ? Number(peak_tflops) : null,
          peak_bandwidth_gb_s: peak_bandwidth_gb_s ? Number(peak_bandwidth_gb_s) : null,
          ridge_point: ridge_point ? Number(ridge_point) : null,
          device_name: device_name ? String(device_name) : null,
        },
      });

      // 2. Clear old stages and cascade metrics/operators
      await tx.stage.deleteMany({
        where: { run_id: payload.id },
      });
      await tx.operatorMetric.deleteMany({
        where: { run_id: payload.id },
      });

      // 3. Create stages and metrics
      for (const stage of payload.stages || []) {
        const createdStage = await tx.stage.create({
          data: {
            id: stage.id,
            run_id: payload.id,
            name: stage.name,
            start_time: new Date(stage.start_time),
            end_time: new Date(stage.end_time),
            duration_ms: stage.duration_ms,
            cpu_percent: stage.cpu_percent,
            ram_used_mb: stage.ram_used_mb,
            gpu_util_percent: stage.gpu_util_percent,
            gpu_mem_used_mb: stage.gpu_mem_used_mb,
            trace_ref: stage.trace_ref,
            metrics: {
              create: (stage.metrics || []).map((m: any) => ({
                timestamp: new Date(m.timestamp),
                key: m.key,
                value: m.value,
              })),
            },
          },
        });

        // If stage contains operator metrics
        if (Array.isArray(stage.operators) && stage.operators.length > 0) {
          for (const op of stage.operators) {
            await tx.operatorMetric.create({
              data: {
                id: op.id,
                run_id: payload.id,
                stage_id: createdStage.id,
                op_name: op.op_name,
                category: op.category || "compute_dense",
                duration_ms: Number(op.duration_ms),
                flops: Number(op.flops),
                bytes_transferred: Number(op.bytes_transferred),
                operational_intensity: Number(op.operational_intensity),
                achieved_tflops: Number(op.achieved_tflops),
                achieved_bandwidth_gb_s: Number(op.achieved_bandwidth_gb_s),
                bound_classification: op.bound_classification,
                recommendation: op.recommendation || "",
                metadata: op.metadata || {},
              },
            });
          }
        }
      }

      // 4. Create top-level run operators (e.g., from dedicated roofline benchmarks)
      if (Array.isArray(payload.operators) && payload.operators.length > 0) {
        for (const op of payload.operators) {
          await tx.operatorMetric.create({
            data: {
              id: op.id,
              run_id: payload.id,
              stage_id: op.stage_id || null,
              op_name: op.op_name,
              category: op.category || "compute_dense",
              duration_ms: Number(op.duration_ms),
              flops: Number(op.flops),
              bytes_transferred: Number(op.bytes_transferred),
              operational_intensity: Number(op.operational_intensity),
              achieved_tflops: Number(op.achieved_tflops),
              achieved_bandwidth_gb_s: Number(op.achieved_bandwidth_gb_s),
              bound_classification: op.bound_classification,
              recommendation: op.recommendation || "",
              metadata: op.metadata || {},
            },
          });
        }
      }

      // Fetch the full run data with stages, metrics, and operators for the broadcast
      return tx.run.findUnique({
        where: { id: payload.id },
        include: {
          stages: {
            orderBy: { start_time: "asc" },
            include: {
              metrics: {
                orderBy: { timestamp: "asc" },
              },
              operators: true,
            },
          },
          operators: true,
        },
      });
    });

    // Broadcast the update
    if (run) {
      sseManager.broadcastRunUpdate(run.id, run);
    }

    return NextResponse.json({ success: true, run });
  } catch (error: any) {
    console.error("Error in POST /api/runs:", error);
    return NextResponse.json({ error: error.message || "Failed to process run payload" }, { status: 500 });
  }
}

export async function GET() {
  try {
    const runs = await db.run.findMany({
      orderBy: { created_at: "desc" },
      include: {
        stages: {
          select: {
            id: true,
            name: true,
            duration_ms: true,
          },
        },
        _count: {
          select: {
            operators: true,
          },
        },
      },
    });
    return NextResponse.json(runs);
  } catch (error: any) {
    console.error("Error in GET /api/runs:", error);
    return NextResponse.json({ error: error.message || "Failed to fetch runs" }, { status: 500 });
  }
}
