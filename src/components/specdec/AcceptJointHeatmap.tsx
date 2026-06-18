import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Joint distribution of the gate's *predicted* accept length (its per-round
// confidence, ∑_d ∏_{k≤d} conf_k) against the *actual* accept length, as density
// over all rounds. Mass concentrates on the diagonal (the confidence predicts the
// outcome: corr 0.76 MTP / 0.84 DFlash) with the two corners hottest — (0,0)
// near-misses and (D,D) clean sweeps — and the off-diagonal spread is the gate's
// per-round uncertainty, i.e. the calibration headroom the oracle reaches.
// Qwen3.6-35B-A3B, SPEED-Bench, temp 0.6. M[actual][predicted] = % of rounds.

// prettier-ignore
const MTP = [
  [7.10,7.00,2.41,1.57,0.42,0.13,0.05,0.01,0.00],
  [2.72,8.96,4.48,2.28,0.70,0.27,0.08,0.03,0.00],
  [1.08,4.19,5.07,2.46,0.59,0.29,0.08,0.04,0.00],
  [0.45,2.16,2.96,2.04,0.87,0.40,0.17,0.03,0.00],
  [0.23,1.08,1.50,1.77,1.48,0.65,0.21,0.08,0.00],
  [0.16,0.54,0.89,1.61,1.20,0.98,0.34,0.55,0.00],
  [0.04,0.36,0.46,0.67,0.94,0.83,0.59,0.19,0.00],
  [0.03,0.24,0.25,0.40,0.43,0.44,0.62,1.34,0.00],
  [0.07,0.37,0.74,2.48,1.56,1.71,2.23,8.65,0.00],
]
// prettier-ignore
const DF = [
  [12.98,6.21,1.38,0.32,0.10,0.06,0.05,0.02,0.00,0.01,0.01,0.00,0.00,0.00,0.00,0.00],
  [6.27,11.22,2.38,0.59,0.12,0.07,0.02,0.01,0.33,0.01,0.00,0.00,0.00,0.00,0.00,0.00],
  [2.00,6.44,5.13,1.10,0.16,0.08,0.02,0.01,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
  [0.71,2.92,3.32,1.96,0.73,0.11,0.06,0.01,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
  [0.33,1.46,1.83,1.52,1.31,0.32,0.06,0.04,0.01,0.00,0.00,0.00,0.00,0.85,0.00,0.00],
  [0.10,0.56,0.82,1.33,0.95,0.67,0.22,0.04,0.02,0.01,0.00,0.01,0.00,0.00,0.00,0.00],
  [0.09,0.35,0.63,0.51,0.57,0.57,0.44,0.14,0.03,0.01,0.00,0.00,0.00,0.00,0.00,0.00],
  [0.02,0.17,0.38,0.43,0.44,0.35,1.31,0.33,0.09,0.02,0.02,0.00,0.01,0.00,0.00,0.00],
  [0.03,0.11,0.18,0.22,0.29,0.30,0.25,0.24,0.38,0.17,0.01,0.01,0.00,0.00,0.00,0.00],
  [0.03,0.06,0.13,0.13,0.22,0.26,0.16,0.17,0.29,0.58,0.22,0.16,0.01,0.00,0.00,0.00],
  [0.00,0.03,0.06,0.06,0.14,0.18,0.13,0.13,0.18,0.55,0.09,0.07,0.03,0.00,0.00,0.00],
  [0.00,0.02,0.04,0.07,0.08,0.06,0.09,0.06,0.10,0.18,0.09,0.05,0.02,0.01,0.00,0.00],
  [0.00,0.01,0.03,0.04,0.03,0.04,0.22,0.04,0.49,0.03,0.07,0.20,0.05,0.03,0.01,0.00],
  [0.00,0.01,0.02,0.04,0.03,0.04,0.04,0.06,0.04,0.04,0.03,0.12,0.09,0.06,0.01,0.00],
  [0.00,0.00,0.02,0.01,0.01,0.02,0.02,0.03,0.05,0.02,0.04,0.05,0.04,0.11,0.07,0.00],
  [0.00,0.02,0.02,0.01,0.04,0.05,0.07,0.10,0.12,0.11,0.14,0.16,0.15,0.37,2.64,0.00],
]

const VMAX = 13 // % in the hottest cell; sqrt-scaled so the diagonal stays visible

export default function AcceptJointHeatmap() {
  const mtpRef = useRef<HTMLCanvasElement>(null)
  const dfRef = useRef<HTMLCanvasElement>(null)
  const charts = useRef<Chart[]>([])
  const theme = useChartTheme()

  useEffect(() => {
    charts.current.forEach((c) => c.destroy())
    charts.current = []
    applyChartDefaults(theme)

    const fill = (v: number) => {
      const m = Math.min(1, Math.sqrt(Math.max(0, v) / VMAX))
      return `hsla(205, 75%, ${theme.isDark ? 58 : 46}%, ${(0.04 + 0.92 * m).toFixed(3)})`
    }
    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    const build = (canvas: HTMLCanvasElement, M: number[][], showY: boolean): Chart => {
      const D = M.length - 1
      const heatmap = {
        id: 'heatmap',
        beforeDatasetsDraw(chart: Chart) {
          const { ctx, chartArea, scales } = chart
          ctx.save()
          ctx.beginPath()
          ctx.rect(chartArea.left, chartArea.top, chartArea.width, chartArea.height)
          ctx.clip()
          for (let a = 0; a <= D; a++) {
            for (let p = 0; p <= D; p++) {
              const xl = scales.x.getPixelForValue(p - 0.5)
              const xr = scales.x.getPixelForValue(p + 0.5)
              const yt = scales.y.getPixelForValue(a + 0.5)
              const yb = scales.y.getPixelForValue(a - 0.5)
              ctx.fillStyle = fill(M[a][p])
              ctx.fillRect(xl, yt, xr - xl + 0.6, yb - yt + 0.6)
            }
          }
          ctx.restore()
        },
      }
      const diagonal = {
        id: 'diagonal',
        afterDatasetsDraw(chart: Chart) {
          const { ctx, scales } = chart
          ctx.save()
          ctx.strokeStyle = theme.mutedForeground
          ctx.setLineDash([4, 4])
          ctx.lineWidth = 1
          ctx.beginPath()
          ctx.moveTo(scales.x.getPixelForValue(-0.5), scales.y.getPixelForValue(-0.5))
          ctx.lineTo(scales.x.getPixelForValue(D + 0.5), scales.y.getPixelForValue(D + 0.5))
          ctx.stroke()
          ctx.restore()
        },
      }
      const tip: { x: number; y: number }[] = []
      for (let a = 0; a <= D; a++) for (let p = 0; p <= D; p++) tip.push({ x: p, y: a })
      return new Chart(canvas, {
        type: 'scatter',
        data: { datasets: [{ label: '', data: tip, pointRadius: 0, pointHitRadius: 10 }] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          interaction: { mode: 'nearest', intersect: false },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => {
                  const p = items[0]?.parsed
                  return p ? `predicted ${p.x}, committed ${p.y}` : ''
                },
                label: (ctx) => {
                  const x = ctx.parsed.x
                  const y = ctx.parsed.y
                  if (typeof x !== 'number' || typeof y !== 'number') return ''
                  return `  ${(M[y]?.[x] ?? 0).toFixed(2)}% of rounds`
                },
              },
            },
          },
          scales: {
            x: {
              type: 'linear',
              min: -0.5,
              max: D + 0.5,
              title: axisTitle('predicted accept (confidence)'),
              grid: { display: false },
              ticks: { ...tickStyle, stepSize: D > 8 ? 3 : 2 },
            },
            y: {
              type: 'linear',
              min: -0.5,
              max: D + 0.5,
              title: showY ? axisTitle('actual accept') : { display: false },
              grid: { display: false },
              ticks: { ...tickStyle, stepSize: D > 8 ? 3 : 2, callback: (v) => (showY ? `${v}` : '') },
            },
          },
        },
        plugins: [heatmap, diagonal],
      })
    }

    if (mtpRef.current) charts.current.push(build(mtpRef.current, MTP, true))
    if (dfRef.current) charts.current.push(build(dfRef.current, DF, false))

    return () => {
      charts.current.forEach((c) => c.destroy())
      charts.current = []
    }
  }, [theme])

  const panel = (title: string, r: typeof mtpRef) => (
    <div style="flex: 1; min-width: 0;">
      <div style={`text-align: center; font-size: 0.85rem; font-weight: 600; color: ${theme.foreground};`}>
        {title}
      </div>
      <div style="position: relative; height: 300px;">
        <canvas ref={r} />
      </div>
    </div>
  )

  return (
    <div class="my-6">
      <div style="display: flex; gap: 0.75rem;">
        {panel('MTP', mtpRef)}
        {panel('DFlash', dfRef)}
      </div>
      <div style={`text-align: center; font-size: 0.75rem; margin-top: 0.4rem; color: ${theme.mutedForeground};`}>
        density of rounds (√-scaled); dashed line is perfect calibration (predicted = actual)
      </div>
    </div>
  )
}
