import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SessionData } from "@/lib/types";
import { Empty } from "./shared";
export type ChartSeries = { label: string; color: string; data: SessionData };
export default function ConfidenceChart({
  series,
  height = 255,
}: {
  series: ChartSeries[];
  height?: number;
}) {
  const [showNull, setShowNull] = useState(true);
  const times = [...new Set(series.flatMap((s) => s.data.time_bins))].sort(
    (a, b) => a - b,
  );
  if (!times.length)
    return (
      <Empty title="No decoding curve yet">
        Run decoding to see confidence over time.
      </Empty>
    );
  const data = times.map((time) => {
    const row: Record<string, unknown> = { time };
    series.forEach((s, n) => {
      const i = s.data.time_bins.indexOf(time);
      row[`observed${n}`] = i < 0 ? null : s.data.observed[i];
      row[`null${n}`] = i < 0 ? null : s.data.null_mean[i];
      const lo = s.data.null_low[i],
        hi = s.data.null_high[i];
      row[`range${n}`] = i < 0 || lo == null || hi == null ? null : [lo, hi];
    });
    return row;
  });
  return (
    <div className="confidence-chart">
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
        <label>
          <input
            type="checkbox"
            checked={showNull}
            onChange={(e) => setShowNull(e.target.checked)}
          />{" "}
          Null mean & 95% range
        </label>
      </div>
      <div style={{ height }} aria-label="Decoding confidence by time">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 18, right: 15, bottom: 8, left: -18 }}
          >
            <CartesianGrid
              vertical={false}
              stroke="#eaeaf2"
              strokeDasharray="3 4"
            />
            <XAxis
              dataKey="time"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fill: "#8a91a6", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}`}
              minTickGap={30}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fill: "#8a91a6", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="chart-tooltip">
                    <strong>{label} ms</strong>
                    {series.map((s, n) => (
                      <div key={n}>
                        <i style={{ background: s.color }} />
                        {s.label}
                        <b>
                          {typeof payload[0].payload[`observed${n}`] ===
                          "number"
                            ? payload[0].payload[`observed${n}`].toFixed(3)
                            : "—"}
                        </b>
                      </div>
                    ))}
                  </div>
                ) : null
              }
            />
            {series.map((s, n) => (
              <Area
                key={`band${n}`}
                dataKey={`range${n}`}
                hide={!showNull}
                stroke="none"
                fill={s.color}
                fillOpacity={0.08}
                type="linear"
                isAnimationActive={false}
              />
            ))}
            {series.map((s, n) => (
              <Line
                key={`null${n}`}
                dataKey={`null${n}`}
                hide={!showNull}
                stroke={s.color}
                strokeOpacity={0.4}
                strokeDasharray="4 4"
                strokeWidth={1.5}
                dot={false}
                type="linear"
                isAnimationActive={false}
              />
            ))}
            {series.map((s, n) => (
              <Line
                key={n}
                dataKey={`observed${n}`}
                stroke={s.color}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: "white" }}
                type="linear"
                isAnimationActive={false}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-axis">
        <span>P(preferred cue)</span>
        <span>Time from cue onset (ms)</span>
        <span>Solid: observed · dashed: null</span>
      </div>
    </div>
  );
}
