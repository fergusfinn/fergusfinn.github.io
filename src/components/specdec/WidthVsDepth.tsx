import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Two ways a single verify forward gets N token-positions: a BATCH of N
// sequences (one decode token each, "width") or one sequence speculating a RUN
// of N ("depth"). Both pay ~the same MoE weight read; the run activates fewer
// distinct experts, because consecutive tokens route locally. The batch curve is
// the real-popularity null (it draws real routings), so the batch-to-CC gap is
// popularity skew and the batch-to-run gap is the locality. Coupon-collector is
// the old independent-uniform idealisation, here only a dashed reference ceiling.
// Per-layer mean distinct experts (of 256). Qwen3.6-35B-A3B, SPEED-Bench
// qualitative, ~11k generated positions.

const N = [1, 2, 3, 4, 6, 8, 12, 16]
const DEPTH = [8.0, 13.27, 17.81, 21.85, 29.07, 35.42, 46.18, 55.05] // run-of-N
const WIDTH = [8.0, 15.36, 22.15, 28.5, 39.94, 50.08, 67.3, 81.54] // batch-of-N (null)
const CC = [7.89, 15.54, 22.95, 30.14, 43.85, 56.72, 80.18, 100.88] // uniform coupon-collector

const pts = (a: number[]) => a.map((y, i) => ({ x: N[i], y }))

export default function WidthVsDepth() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const depthC = `hsl(220, 70%, ${theme.isDark ? 65 : 45}%)` // run = the hero
    const widthC = `hsl(28, 80%, ${theme.isDark ? 62 : 48}%)` // batch = the null

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }
    const lineDS = (
      label: string,
      data: { x: number; y: number }[],
      color: string,
      opts: Record<string, unknown> = {}
    ) => ({
      label,
      data,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0.25,
      ...opts,
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          lineDS('depth: run of N', pts(DEPTH), depthC, { borderWidth: 2.5 }),
          lineDS('width: batch of N', pts(WIDTH), widthC),
          lineDS('coupon-collector', pts(CC), theme.mutedForeground, {
            borderWidth: 1.5,
            borderDash: [5, 4],
          }),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        layout: { padding: { top: 10, right: 70 } },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              boxWidth: 14,
              boxHeight: 2,
              color: theme.foreground,
              font: { family: theme.fontFamily },
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `N = ${items[0].parsed.x} positions` : ''),
              label: (ctx) =>
                `  ${ctx.dataset.label}: ${(ctx.parsed.y as number).toFixed(1)} experts/layer`,
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 1,
            max: 16,
            title: axisTitle('N (token-positions in one verify forward)'),
            grid: { color: theme.grid },
            afterBuildTicks: (axis) => {
              axis.ticks = [1, 2, 4, 8, 12, 16].map((v) => ({ value: v }))
            },
            ticks: { ...tickStyle, callback: (v) => `${v}` },
          },
          y: {
            type: 'linear',
            min: 0,
            max: 108,
            title: axisTitle('distinct experts / layer (of 256)'),
            grid: { color: theme.grid },
            ticks: { ...tickStyle, stepSize: 20 },
          },
        },
      },
    })


    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  return (
    <div class="my-6" style="position: relative; height: 420px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
