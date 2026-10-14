import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    
    const run = await db.run.findUnique({
      where: { id },
      include: {
        stages: {
          orderBy: { start_time: "asc" },
          select: {
            id: true,
            run_id: true,
            name: true,
            start_time: true,
            end_time: true,
            duration_ms: true,
            cpu_percent: true,
            ram_used_mb: true,
            gpu_util_percent: true,
            gpu_mem_used_mb: true,
            metrics: {
              orderBy: { timestamp: "asc" },
            },
            operators: true,
          },
        },
        operators: {
          orderBy: { operational_intensity: "asc" },
        },
      },
    });

    if (!run) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }

    // Find which stages actually have trace data to set the has_trace flag
    const stagesWithTrace = await db.stage.findMany({
      where: { run_id: id, NOT: { trace_ref: null } },
      select: { id: true },
    });
    const traceIdsSet = new Set(stagesWithTrace.map((s) => s.id));

    // Combine stage-level operators and run-level operators if any
    const allOperators = [...run.operators];
    for (const st of run.stages) {
      if (Array.isArray(st.operators)) {
        for (const op of st.operators) {
          if (!allOperators.some((existing) => existing.id === op.id)) {
            allOperators.push(op);
          }
        }
      }
    }

    const runWithTraceFlag = {
      ...run,
      operators: allOperators,
      stages: run.stages.map((stage) => ({
        ...stage,
        has_trace: traceIdsSet.has(stage.id),
      })),
    };

    return NextResponse.json(runWithTraceFlag);
  } catch (error: any) {
    const { id } = await params;
    console.error(`Error in GET /api/runs/${id}:`, error);
    return NextResponse.json({ error: error.message || "Failed to fetch run" }, { status: 500 });
  }
}
