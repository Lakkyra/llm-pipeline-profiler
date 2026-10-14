"use client";

import React, { useState, useMemo, useEffect } from "react";
import {
  Cpu,
  Database,
  Zap,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Search,
  Filter,
  ArrowUpDown,
  Sparkles,
  Info,
  Maximize2,
  RefreshCw,
} from "lucide-react";
import { OperatorMetric, HardwareCeilings, RooflineDataResponse } from "@/types/run";

interface RooflineChartProps {
  runId: string;
  initialCeilings?: HardwareCeilings;
  initialOperators?: OperatorMetric[];
}

export default function RooflineChart({
  runId,
  initialCeilings,
  initialOperators = [],
}: RooflineChartProps) {
  const [data, setData] = useState<RooflineDataResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(!initialCeilings);
  const [error, setError] = useState<string | null>(null);

  // Filtering & Search
  const [searchQuery, setSearchQuery] = useState("");
  const [boundFilter, setBoundFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [hoveredOp, setHoveredOp] = useState<OperatorMetric | null>(null);
  const [selectedOp, setSelectedOp] = useState<OperatorMetric | null>(null);

  const fetchRoofline = async () => {
    try {
      setLoading(true);
      const res = await fetch(`/api/runs/${runId}/roofline`);
      if (!res.ok) throw new Error("Failed to load Roofline telemetry");
      const json: RooflineDataResponse = await res.json();
      setData(json);
    } catch (err: any) {
      setError(err.message || "Failed to load Roofline data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRoofline();
  }, [runId]);

  const ceilings = data?.ceilings || initialCeilings || {
    device_name: "NVIDIA GPU",
    peak_tflops: 19.5,
    peak_bandwidth_gb_s: 900.0,
    ridge_point: 21.67,
    precision: "FP32",
    is_cuda_calibrated: false,
  };

  const operators = useMemo(() => {
    if (data?.operators && data.operators.length > 0) return data.operators;
    return initialOperators;
  }, [data, initialOperators]);

  // Derived Hardware Metrics
  const peakTflops = ceilings.peak_tflops > 0 ? ceilings.peak_tflops : 19.5;
  const peakBandwidth = ceilings.peak_bandwidth_gb_s > 0 ? ceilings.peak_bandwidth_gb_s : 900.0;
  const ridgePoint = ceilings.ridge_point > 0 ? ceilings.ridge_point : (peakTflops * 1000.0) / peakBandwidth;

  // Filtered operators for chart and table
  const filteredOperators = useMemo(() => {
    return operators.filter((op) => {
      const matchesSearch =
        op.op_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        op.category.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesBound = boundFilter === "all" || op.bound_classification === boundFilter;
      const matchesCategory = categoryFilter === "all" || op.category === categoryFilter;
      return matchesSearch && matchesBound && matchesCategory;
    });
  }, [operators, searchQuery, boundFilter, categoryFilter]);

  // Summary Counts
  const computeBoundCount = operators.filter((o) => o.bound_classification === "compute_bound").length;
  const memoryBoundCount = operators.filter((o) => o.bound_classification === "memory_bound").length;
  const latencyLimitedCount = operators.filter((o) => o.bound_classification === "latency_limited").length;

  // ---------------------------------------------------------------------------
  // Log-Log SVG Coordinate System Transformation
  // ---------------------------------------------------------------------------
  // Domain (Operational Intensity I in FLOPs/Byte): 10^-2 to 10^4
  // Range (Achieved TFLOPS P): 10^-3 to 10^3
  const minLogX = -2;
  const maxLogX = 4;
  const minLogY = -3;
  const maxLogY = 3;

  const width = 850;
  const height = 480;
  const margin = { top: 40, right: 40, bottom: 60, left: 75 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const log10 = (v: number) => Math.log10(Math.max(v, 1e-9));

  const scaleX = (intensity: number) => {
    const val = Math.max(Math.min(log10(intensity), maxLogX), minLogX);
    return margin.left + ((val - minLogX) / (maxLogX - minLogX)) * innerWidth;
  };

  const scaleY = (tflops: number) => {
    const val = Math.max(Math.min(log10(tflops), maxLogY), minLogY);
    return margin.top + innerHeight - ((val - minLogY) / (maxLogY - minLogY)) * innerHeight;
  };

  // Ceiling Path (Memory bandwidth slope -> Compute flat ceiling)
  // At I = ridgePoint, P = peakTflops
  const ridgeX = scaleX(ridgePoint);
  const peakY = scaleY(peakTflops);
  const startX = scaleX(Math.pow(10, minLogX));
  const startY = scaleY(Math.pow(10, minLogX) * peakBandwidth * 1e-3);
  const endX = scaleX(Math.pow(10, maxLogX));

  // 30% Latency Ceiling Line
  const latencyPeakY = scaleY(peakTflops * 0.3);
  const latencyStartY = scaleY(Math.pow(10, minLogX) * peakBandwidth * 1e-3 * 0.3);

  // Grid tick marks
  const xTicks = [-2, -1, 0, 1, 2, 3, 4];
  const yTicks = [-3, -2, -1, 0, 1, 2, 3];

  const getBoundColor = (bound: string) => {
    switch (bound) {
      case "compute_bound":
        return "#10b981"; // Emerald
      case "memory_bound":
        return "#06b6d4"; // Cyan
      case "latency_limited":
        return "#f43f5e"; // Rose
      default:
        return "#818cf8"; // Indigo
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Hardware Ceiling KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Peak Compute */}
        <div className="glass-card p-4 relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Peak Compute Ceiling ($P_{`{peak}`}$)
            </span>
            <div className="h-8 w-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400">
              <Zap className="h-4 w-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-100 flex items-baseline gap-1.5">
            {peakTflops.toFixed(1)}
            <span className="text-xs font-semibold text-emerald-400">TFLOPS</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1 flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
            {ceilings.precision} precision &bull; {ceilings.is_cuda_calibrated ? "CUDA FMA Probed" : "Device Profile"}
          </p>
        </div>

        {/* Peak Memory Bandwidth */}
        <div className="glass-card p-4 relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Peak Bandwidth ($B_{`{peak}`}$)
            </span>
            <div className="h-8 w-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400">
              <Database className="h-4 w-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-100 flex items-baseline gap-1.5">
            {peakBandwidth.toFixed(1)}
            <span className="text-xs font-semibold text-cyan-400">GB/s</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1 flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-400"></span>
            {ceilings.device_name}
          </p>
        </div>

        {/* Ridge Point */}
        <div className="glass-card p-4 relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Ridge Point ($R$)
            </span>
            <div className="h-8 w-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400">
              <Sparkles className="h-4 w-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-100 flex items-baseline gap-1.5">
            {ridgePoint.toFixed(2)}
            <span className="text-xs font-semibold text-indigo-400">FLOPs/Byte</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Memory bound if $I &lt; {ridgePoint.toFixed(1)}$; compute bound above
          </p>
        </div>

        {/* Bottleneck Distribution */}
        <div className="glass-card p-4 relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Bottleneck Breakdown
            </span>
            <div className="h-8 w-8 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-400">
              <Cpu className="h-4 w-4" />
            </div>
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {computeBoundCount} Compute
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              {memoryBoundCount} Memory
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
              {latencyLimitedCount} Latency
            </span>
          </div>
          <p className="text-[11px] text-slate-500 mt-2">
            Total {operators.length} operators benchmarked
          </p>
        </div>
      </div>

      {/* Main Roofline Log-Log Visualization Card */}
      <div className="glass-card p-6 relative">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-indigo-400" />
              Hierarchical Roofline Model &amp; Kernel Saturation
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Log-log operational intensity ($I = \text{FLOPs}/\text{Byte}$) vs. achieved compute throughput ($P_{\text{achieved}}$).
            </p>
          </div>

          {/* Quick Legend / Badges */}
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
              Compute-Bound
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="w-3 h-3 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400/50"></span>
              Memory-Bound
            </span>
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="w-3 h-3 rounded-full bg-rose-500 shadow-sm shadow-rose-500/50"></span>
              Latency-Limited (&lt;30%)
            </span>
            <button
              onClick={fetchRoofline}
              className="p-1.5 rounded-lg border border-slate-700 hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
              title="Refresh telemetry"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* SVG Canvas */}
        <div className="w-full overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-auto select-none bg-slate-950/60 rounded-xl border border-slate-800/80 shadow-inner"
            style={{ minWidth: "680px" }}
          >
            <defs>
              <linearGradient id="rooflineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9" />
                <stop offset="50%" stopColor="#818cf8" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.9" />
              </linearGradient>

              <pattern id="gridPattern" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(51, 65, 85, 0.15)" strokeWidth="1" />
              </pattern>
            </defs>

            {/* Background Grid */}
            <rect x={margin.left} y={margin.top} width={innerWidth} height={innerHeight} fill="url(#gridPattern)" />

            {/* Vertical Grid Lines & X-Axis Labels */}
            {xTicks.map((pow) => {
              const x = scaleX(Math.pow(10, pow));
              return (
                <g key={`xtick-${pow}`}>
                  <line
                    x1={x}
                    y1={margin.top}
                    x2={x}
                    y2={margin.top + innerHeight}
                    stroke="rgba(100, 116, 139, 0.2)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={x}
                    y={margin.top + innerHeight + 20}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    10{pow !== 0 && <tspan dy="-5" fontSize="8">{pow}</tspan>}
                  </text>
                </g>
              );
            })}

            {/* Horizontal Grid Lines & Y-Axis Labels */}
            {yTicks.map((pow) => {
              const y = scaleY(Math.pow(10, pow));
              return (
                <g key={`ytick-${pow}`}>
                  <line
                    x1={margin.left}
                    y1={y}
                    x2={margin.left + innerWidth}
                    y2={y}
                    stroke="rgba(100, 116, 139, 0.2)"
                    strokeDasharray="3 3"
                  />
                  <text
                    x={margin.left - 12}
                    y={y + 4}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    10{pow !== 0 && <tspan dy="-5" fontSize="8">{pow}</tspan>}
                  </text>
                </g>
              );
            })}

            {/* Axis Titles */}
            <text
              x={margin.left + innerWidth / 2}
              y={height - 15}
              textAnchor="middle"
              fill="#cbd5e1"
              fontSize="12"
              fontWeight="bold"
            >
              Operational Intensity $I$ (FLOPs / DRAM Byte)
            </text>

            <text
              x={- (margin.top + innerHeight / 2)}
              y="22"
              transform="rotate(-90)"
              textAnchor="middle"
              fill="#cbd5e1"
              fontSize="12"
              fontWeight="bold"
            >
              Performance $P$ (TFLOPS)
            </text>

            {/* Ridge Point Vertical Dashed Marker */}
            <line
              x1={ridgeX}
              y1={margin.top}
              x2={ridgeX}
              y2={margin.top + innerHeight}
              stroke="#818cf8"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
            <text
              x={ridgeX + 6}
              y={margin.top + 25}
              fill="#a5b4fc"
              fontSize="10"
              fontWeight="bold"
              fontFamily="monospace"
            >
              Ridge: {ridgePoint.toFixed(1)} FLOPs/B
            </text>

            {/* 30% Latency Ceiling Boundary */}
            <path
              d={`M ${startX} ${latencyStartY} L ${ridgeX} ${latencyPeakY} L ${endX} ${latencyPeakY}`}
              fill="none"
              stroke="#f43f5e"
              strokeWidth="1.2"
              strokeDasharray="5 5"
              opacity="0.65"
            />
            <text
              x={endX - 8}
              y={latencyPeakY - 6}
              textAnchor="end"
              fill="#f43f5e"
              fontSize="9"
              fontFamily="monospace"
              opacity="0.8"
            >
              30% Under-Utilization Boundary
            </text>

            {/* Theoretical Roofline Ceiling Path */}
            <path
              d={`M ${startX} ${startY} L ${ridgeX} ${peakY} L ${endX} ${peakY}`}
              fill="none"
              stroke="url(#rooflineGradient)"
              strokeWidth="3.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Ceiling Text Annotations */}
            <text
              x={startX + 30}
              y={startY - 16}
              fill="#38bdf8"
              fontSize="10"
              fontWeight="bold"
              transform={`rotate(-28, ${startX + 30}, ${startY - 16})`}
            >
              Memory Bandwidth Ceiling ({peakBandwidth} GB/s)
            </text>

            <text
              x={endX - 10}
              y={peakY - 10}
              textAnchor="end"
              fill="#34d399"
              fontSize="10"
              fontWeight="bold"
            >
              Peak Compute Ceiling ({peakTflops} TFLOPS)
            </text>

            {/* Operator Scatter Points */}
            {filteredOperators.map((op) => {
              const cx = scaleX(op.operational_intensity);
              const cy = scaleY(op.achieved_tflops);
              const isHovered = hoveredOp?.id === op.id;
              const isSelected = selectedOp?.id === op.id;
              const color = getBoundColor(op.bound_classification);

              return (
                <g
                  key={op.id}
                  className="cursor-pointer transition-transform"
                  onMouseEnter={() => setHoveredOp(op)}
                  onMouseLeave={() => setHoveredOp(null)}
                  onClick={() => setSelectedOp(op)}
                >
                  {/* Outer pulse when selected or hovered */}
                  {(isHovered || isSelected) && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r="14"
                      fill={color}
                      opacity="0.25"
                      className="animate-ping"
                    />
                  )}

                  {/* Main operator dot */}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered || isSelected ? "7" : "5"}
                    fill={color}
                    stroke="#0f172a"
                    strokeWidth="2"
                    className="transition-all duration-200"
                  />

                  {/* Tiny label on hover */}
                  {isHovered && (
                    <text
                      x={cx}
                      y={cy - 12}
                      textAnchor="middle"
                      fill="#f8fafc"
                      fontSize="10"
                      fontWeight="bold"
                      className="pointer-events-none drop-shadow"
                    >
                      {op.op_name} ({op.achieved_tflops.toFixed(2)} TF)
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Live Hover / Selected Operator Tooltip Card */}
        {(hoveredOp || selectedOp) && (
          <div className="mt-4 p-4 rounded-xl bg-slate-900/90 border border-slate-700/80 shadow-2xl backdrop-blur-md">
            {(() => {
              const active = hoveredOp || selectedOp!;
              const color = getBoundColor(active.bound_classification);
              return (
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span
                        className="inline-block w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: color }}
                      ></span>
                      <h4 className="text-sm font-bold text-slate-100">{active.op_name}</h4>
                      <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {active.category}
                      </span>
                      <span
                        className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded"
                        style={{
                          backgroundColor: `${color}1a`,
                          color: color,
                          border: `1px solid ${color}40`,
                        }}
                      >
                        {active.bound_classification.replace("_", " ")}
                      </span>
                    </div>

                    <p className="text-xs text-slate-400">
                      Intensity: <span className="text-indigo-300 font-mono font-semibold">{active.operational_intensity.toFixed(2)} FLOPs/Byte</span> &bull; 
                      Throughput: <span className="text-emerald-300 font-mono font-semibold">{active.achieved_tflops.toFixed(3)} TFLOPS</span> &bull; 
                      DRAM Bandwidth: <span className="text-cyan-300 font-mono font-semibold">{active.achieved_bandwidth_gb_s.toFixed(1)} GB/s</span> &bull; 
                      Duration: <span className="text-amber-300 font-mono font-semibold">{active.duration_ms.toFixed(2)} ms</span>
                    </p>
                  </div>

                  {/* DevTech Recommendation Action Box */}
                  <div className="md:max-w-md p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-xs">
                    <div className="flex items-center gap-1.5 text-indigo-400 font-bold mb-1">
                      <Info className="h-3.5 w-3.5" />
                      DevTech Diagnostic Advice:
                    </div>
                    <p className="text-slate-300 leading-relaxed text-[11px]">
                      {active.recommendation}
                    </p>
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Operator Telemetry & DevTech Diagnostics Table */}
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-md font-bold text-slate-100">Profiled Operator Telemetry &amp; Bottlenecks</h3>
            <p className="text-xs text-slate-400">
              Detailed FLOP counts, memory transfers, and actionable optimization heuristics.
            </p>
          </div>

          {/* Filters & Search */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                placeholder="Search operator..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
              />
            </div>

            {/* Bound Filter */}
            <select
              value={boundFilter}
              onChange={(e) => setBoundFilter(e.target.value)}
              className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-lg text-slate-300 focus:outline-none focus:border-indigo-500"
            >
              <option value="all">All Bounds</option>
              <option value="compute_bound">Compute-Bound</option>
              <option value="memory_bound">Memory-Bound</option>
              <option value="latency_limited">Latency-Limited</option>
            </select>

            {/* Category Filter */}
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="px-3 py-1.5 text-xs bg-slate-900 border border-slate-800 rounded-lg text-slate-300 focus:outline-none focus:border-indigo-500"
            >
              <option value="all">All Categories</option>
              <option value="compute_dense">Compute Dense (GEMM)</option>
              <option value="attention">Attention</option>
              <option value="norm">Normalization</option>
              <option value="elementwise">Elementwise / Activation</option>
            </select>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/40">
                <th className="py-3 px-3 font-semibold">Operator</th>
                <th className="py-3 px-3 font-semibold">Category</th>
                <th className="py-3 px-3 font-semibold">Duration</th>
                <th className="py-3 px-3 font-semibold">Intensity ($I$)</th>
                <th className="py-3 px-3 font-semibold">Achieved TF</th>
                <th className="py-3 px-3 font-semibold">Bandwidth</th>
                <th className="py-3 px-3 font-semibold">Bottleneck</th>
                <th className="py-3 px-3 font-semibold">DevTech Advice</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredOperators.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-500">
                    No operator metrics recorded matching criteria.
                  </td>
                </tr>
              ) : (
                filteredOperators.map((op) => {
                  const color = getBoundColor(op.bound_classification);
                  const isSelected = selectedOp?.id === op.id;

                  return (
                    <tr
                      key={op.id}
                      className={`hover:bg-slate-800/30 transition-colors cursor-pointer ${
                        isSelected ? "bg-indigo-950/30 border-l-2 border-indigo-500" : ""
                      }`}
                      onClick={() => setSelectedOp(op)}
                      onMouseEnter={() => setHoveredOp(op)}
                      onMouseLeave={() => setHoveredOp(null)}
                    >
                      <td className="py-3 px-3 font-medium text-slate-200">
                        {op.op_name}
                      </td>
                      <td className="py-3 px-3">
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400">
                          {op.category}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-slate-300 font-mono">
                        {op.duration_ms.toFixed(2)} ms
                      </td>
                      <td className="py-3 px-3 text-indigo-300 font-mono font-semibold">
                        {op.operational_intensity.toFixed(2)} FLOPs/B
                      </td>
                      <td className="py-3 px-3 text-emerald-300 font-mono font-semibold">
                        {op.achieved_tflops.toFixed(3)} TF
                      </td>
                      <td className="py-3 px-3 text-cyan-300 font-mono">
                        {op.achieved_bandwidth_gb_s.toFixed(1)} GB/s
                      </td>
                      <td className="py-3 px-3">
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider inline-block"
                          style={{
                            backgroundColor: `${color}18`,
                            color: color,
                            border: `1px solid ${color}35`,
                          }}
                        >
                          {op.bound_classification.replace("_", " ")}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-slate-400 max-w-xs truncate text-[11px]" title={op.recommendation}>
                        {op.recommendation}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
