import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Per-depth survival (fraction of rounds whose draft token at depth k commits)
// for each head, with a single-α geometric α^k overlaid (α = the first-token
// acceptance). On a log y-axis the geometric is a straight line, so the measured
// curve's shape IS its deviation from geometric: both heads decay faster than
// geometric early (the curve dips below its line) and DFlash's deep tail is
// heavier than geometric (it rises back above). A single α is a rough fit, not a
// good one. Qwen3.6-35B-A3B, SPEED-Bench, temp 0.6.

const MTP_SURV = [0.813, 0.618, 0.48, 0.389, 0.319, 0.256, 0.215, 0.178]
const DF_SURV = [
  0.789, 0.578, 0.429, 0.331, 0.253, 0.206, 0.172, 0.136, 0.115, 0.09, 0.074, 0.065, 0.052, 0.045,
  0.04,
]
const geo = (surv: number[]) => surv.map((_, i) => Math.pow(surv[0], i + 1))
const pts = (surv: number[]) => surv.map((y, i) => ({ x: i + 1, y }))

export default function AcceptanceCurve() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const mtp = `hsl(220, 70%, ${theme.isDark ? 65 : 45}%)`
    const df = `hsl(28, 80%, ${theme.isDark ? 62 : 48}%)`
    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    const measured = (label: string, surv: number[], color: string) => ({
      label,
      data: pts(surv),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.1,
    })
    const geometric = (label: string, surv: number[], color: string) => ({
      label,
      data: pts(geo(surv)),
      borderColor: color,
      backgroundColor: color,
      borderWidth: 1.5,
      borderDash: [5, 4],
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0,
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          measured('MTP measured', MTP_SURV, mtp),
          geometric('MTP  α^k', MTP_SURV, mtp),
          measured('DFlash measured', DF_SURV, df),
          geometric('DFlash  α^k', DF_SURV, df),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: { boxWidth: 18, boxHeight: 2, color: theme.foreground, font: { family: theme.fontFamily } },
          },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `depth ${items[0].parsed.x}` : ''),
              label: (ctx) => `  ${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(3)}`,
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 1,
            max: 16,
            title: axisTitle('draft depth k'),
            ticks: { ...tickStyle, stepSize: 1 },
            grid: { color: theme.grid },
          },
          y: {
            type: 'logarithmic',
            min: 0.03,
            max: 1,
            title: axisTitle('survival (P commit ≥ k)'),
            grid: { color: theme.grid },
            ticks: {
              ...tickStyle,
              callback: (v) => {
                const a = [1, 0.5, 0.2, 0.1, 0.05, 0.03]
                return a.includes(Number(v)) ? `${v}` : ''
              },
            },
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
    <div class="my-6" style="position: relative; height: 360px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
