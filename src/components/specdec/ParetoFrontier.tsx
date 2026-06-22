import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'
import grid from './drafterGrid.json'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Latency–throughput Pareto. x = throughput, y = TPOT (per-token latency); good
// is bottom-right (more throughput, less latency). Each thin line is one fixed
// draft depth γ swept over batch; the bold line per drafter is the priced
// (adaptive) policy. Because throughput × TPOT = B is fixed by the load, at any
// one batch every γ lands on the same load contour and the best γ sits furthest
// down-right on it — so the adaptive policy is the lower-right frontier of the
// whole family. A fixed γ rides the frontier at one batch and peels inside it
// elsewhere. Log–log axes: load contours are the −1-slope diagonals.
// Qwen3.6-35B-A3B verifier, B200, decode-only (ISL=1, OSL=1024).

const CONC: number[] = grid.conc
type Series = { gamma: number; goodput: number[]; tpot: number[] }
type Drafter = {
  gamma_max: number
  nospec: { goodput: number[]; tpot: number[] }
  fixed: Series[]
  adaptive: { goodput: number[]; tpot: number[]; gstar: number[] }
}
const MTP = grid.drafters.mtp as Drafter
const DF = grid.drafters.dflash as Drafter

const xy = (gp: number[], tp: number[]) => gp.map((g, i) => ({ x: g, y: tp[i] }))

export default function ParetoFrontier() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const mtpL = theme.isDark ? 65 : 45
    const dfL = theme.isDark ? 62 : 48
    const mtp = `hsl(220, 70%, ${mtpL}%)`
    const df = `hsl(28, 80%, ${dfL}%)`
    const mtpFaint = `hsla(220, 55%, ${mtpL}%, 0.22)`
    const dfFaint = `hsla(28, 65%, ${dfL}%, 0.22)`

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    const familyDS = (d: Drafter, color: string) =>
      d.fixed.map((s) => ({
        label: '',
        data: xy(s.goodput, s.tpot),
        borderColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 0,
        tension: 0.2,
        order: 3,
      }))

    const envDS = (label: string, d: Drafter, color: string, bg: string) => ({
      label,
      data: xy(d.adaptive.goodput, d.adaptive.tpot),
      borderColor: color,
      backgroundColor: bg,
      borderWidth: 2.5,
      pointRadius: 3,
      pointHoverRadius: 5,
      pointBackgroundColor: theme.isDark ? '#0a0a0a' : '#ffffff',
      pointBorderColor: color,
      pointBorderWidth: 1.5,
      tension: 0.2,
      order: 1,
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          ...familyDS(MTP, mtpFaint),
          ...familyDS(DF, dfFaint),
          {
            label: 'no speculation',
            data: xy(MTP.nospec.goodput, MTP.nospec.tpot),
            borderColor: theme.foreground,
            backgroundColor: theme.foreground,
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2,
            order: 2,
          },
          envDS('MTP — adaptive', MTP, mtp, mtp),
          envDS('DFlash — adaptive', DF, df, df),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        layout: { padding: { top: 18, right: 12 } },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              boxWidth: 14,
              boxHeight: 2,
              color: theme.foreground,
              font: { family: theme.fontFamily },
              filter: (item) => !!item.text,
            },
          },
          tooltip: {
            filter: (item) => !!item.dataset.label,
            callbacks: {
              title: () => '',
              label: (ctx) => {
                const x = Math.round(ctx.parsed.x as number)
                const y = (ctx.parsed.y as number).toFixed(1)
                if (ctx.dataset.label === 'no speculation') {
                  const i = MTP.nospec.goodput.indexOf(ctx.parsed.x as number)
                  const b = i >= 0 ? CONC[i] : '?'
                  return `  no speculation @ B=${b}: ${x} tok/s, ${y} ms`
                }
                const d = ctx.dataset.label?.startsWith('MTP') ? MTP : DF
                const i = d.adaptive.goodput.indexOf(ctx.parsed.x as number)
                const b = i >= 0 ? CONC[i] : '?'
                const g = i >= 0 ? d.adaptive.gstar[i] : 0
                const gtxt = g === 0 ? 'no-spec' : `γ${g}`
                return `  ${ctx.dataset.label} @ B=${b}: ${x} tok/s, ${y} ms  (${gtxt})`
              },
            },
          },
        },
        scales: {
          x: {
            type: 'logarithmic',
            title: axisTitle('throughput (committed tok/s)  →  better'),
            grid: { color: theme.grid },
            ticks: {
              ...tickStyle,
              callback: (v) => {
                const n = Number(v)
                if (![1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000].includes(n)) return ''
                return n >= 1000 ? `${n / 1000}k` : `${n}`
              },
            },
          },
          y: {
            type: 'logarithmic',
            reverse: true,
            title: axisTitle('TPOT (ms)  →  better'),
            grid: { color: theme.grid },
            ticks: {
              ...tickStyle,
              callback: (v) => {
                const n = Number(v)
                if (![0.5, 1, 2, 5, 10, 20].includes(n)) return ''
                return `${n}`
              },
            },
          },
        },
      },
      plugins: [
        {
          id: 'familyKey',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea } = chart
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.globalAlpha = 0.4
            ctx.lineWidth = 1.5
            ctx.beginPath()
            ctx.moveTo(chartArea.left + 6, chartArea.bottom - 14)
            ctx.lineTo(chartArea.left + 24, chartArea.bottom - 14)
            ctx.stroke()
            ctx.globalAlpha = 1
            ctx.fillStyle = theme.mutedForeground
            ctx.font = `11px ${theme.fontFamily}`
            ctx.textAlign = 'left'
            ctx.fillText('each thin line: one fixed γ', chartArea.left + 30, chartArea.bottom - 11)
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
    <div class="my-6" style="position: relative; height: 460px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
