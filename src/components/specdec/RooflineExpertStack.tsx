import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Lin-lin roofline: arithmetic intensity vs batch, MoE FFN vs isoparametric dense
// FFN. On linear axes the slopes read true: dense is a steep spike to its knee,
// MoE the long shallow diagonal out to a far-larger knee, and the shaded
// free-token band (memory-bound, where speculated tokens are ~free) is the wide
// triangle under MoE. Both FFNs are MXFP4, sharing one ridge; the only difference
// is routing. DeepSeek-V4-Flash FFN on B200.

const R = 1125 // B200 fp4 (MXFP4) ridge point, FLOP/byte (9 PFLOP/s / 8 TB/s)
const B_FP4 = 0.53 // bytes/elem for MXFP4 (4-bit + 8-bit scale per 32-block)
const AI = 2 / B_FP4 // FLOP/byte added per token by an fp4 GEMM (~3.77)
const E = 256
const K = 6

function eAct(n: number): number {
  // Exact distinct-top-k coupon: each token picks k distinct experts of E.
  return E * (1 - Math.pow(1 - K / E, n))
}

const denseAI = (n: number) => Math.min(AI * n, R)
const moeAI = (n: number) => Math.min((AI * n * K) / eAct(n), R)

const DENSE_KNEE = R / AI // ~298 tokens, dense FFN hits R
const MOE_SAT_KNEE = (DENSE_KNEE * E) / K // ~12,700 tokens, MoE hits R once experts saturate
const BMAX = 13000

// log sweep plus the exact roofline corners so the apexes render crisply
const N_GRID: number[] = []
for (let e = 0; e <= 280; e++) {
  N_GRID.push(Math.pow(10, (e / 280) * Math.log10(BMAX)))
}
N_GRID.push(DENSE_KNEE, MOE_SAT_KNEE)
const GRID = N_GRID.filter((n) => n <= BMAX).sort((a, b) => a - b)
const DENSE = GRID.map((n) => ({ x: n, y: denseAI(n) }))
const MOE = GRID.map((n) => ({ x: n, y: moeAI(n) }))
const DENSE_FREE = GRID.filter((n) => n <= DENSE_KNEE).map((n) => ({ x: n, y: denseAI(n) }))
const MOE_FREE = GRID.filter((n) => n <= MOE_SAT_KNEE).map((n) => ({ x: n, y: moeAI(n) }))

const fmtK = (b: number) => (b >= 1000 ? `${b / 1000}k` : `${b}`)

export default function RooflineExpertStack() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const denseL = theme.isDark ? 65 : 45
    const moeL = theme.isDark ? 62 : 48
    const denseColor = `hsl(220, 70%, ${denseL}%)`
    const moeColor = `hsl(28, 80%, ${moeL}%)`
    const denseFill = `hsla(220, 70%, ${denseL}%, 0.16)`
    const moeFill = `hsla(28, 80%, ${moeL}%, 0.12)`

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }
    const lineDS = (label: string, data: { x: number; y: number }[], color: string) => ({
      label,
      data,
      borderColor: color,
      backgroundColor: color,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.1,
      borderWidth: 2,
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          { label: '', data: MOE_FREE, borderColor: 'transparent', backgroundColor: moeFill, fill: 'start', pointRadius: 0, pointHoverRadius: 0, borderWidth: 0, tension: 0.1 },
          { label: '', data: DENSE_FREE, borderColor: 'transparent', backgroundColor: denseFill, fill: 'start', pointRadius: 0, pointHoverRadius: 0, borderWidth: 0, tension: 0.1 },
          lineDS('dense FFN (same params)', DENSE, denseColor),
          lineDS('MoE FFN (6 of 256)', MOE, moeColor),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: { boxWidth: 14, boxHeight: 2, color: theme.foreground, font: { family: theme.fontFamily }, filter: (item) => !!item.text },
          },
          tooltip: {
            filter: (item) => !!item.dataset.label,
            callbacks: {
              title: (items) => (items.length > 0 ? `batch ≈ ${Math.round(items[0].parsed.x ?? 0)}` : ''),
              label: (ctx) => `  ${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(1)} FLOP/byte`,
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: BMAX,
            title: axisTitle('batch size B (tokens)'),
            ticks: { ...tickStyle, stepSize: 3000, callback: (v) => fmtK(Number(v)) },
            grid: { color: theme.grid },
          },
          y: {
            type: 'linear',
            position: 'left',
            min: 0,
            max: 1200,
            title: axisTitle('intensity (FLOP/byte)'),
            ticks: tickStyle,
            grid: { color: theme.grid },
          },
        },
      },
      plugins: [
        {
          id: 'ridge',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart
            const yR = scales.y.getPixelForValue(R)
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.setLineDash([4, 4])
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(chartArea.left, yR)
            ctx.lineTo(chartArea.right, yR)
            ctx.stroke()
            ctx.restore()
          },
        },
      ],
    })

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  return (
    <div class="my-6" style="position: relative; height: 400px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
