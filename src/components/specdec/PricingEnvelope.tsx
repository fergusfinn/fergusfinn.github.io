import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Headroom of adaptive speculation, normalised to no-spec so the batch-size
// scaling divides out and the drafter difference is legible. Each drafter's
// best-in-hindsight fixed-γ throughput is plotted as a % of no-spec; the adaptive
// policy's throughput is the faint shadow underneath (it coincides to <0.5%, so
// the line riding the shadow IS the envelope claim). Markers carry γ*, the
// hindsight-optimal draft depth, which moves with load. The two curves cross at
// ~conc 256: DFlash's flat cost wins the mid-batch band, MTP's sparse per-pass
// cost reclaims the compute-bound top. Qwen3.6-35B-A3B / B200, decode-only.

const CONC = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
const NOSPEC = [1394, 2076, 2789, 3488, 4292, 5561, 8157, 14096, 25788, 45169, 72388, 103604]
const MTP_BEST = [1618, 2263, 2925, 3680, 6197, 11701, 22073, 41889, 78841, 140324, 196811, 235818]
const MTP_ADP = [1618, 2263, 2925, 3682, 6201, 11704, 22084, 41930, 78924, 140499, 197194, 236380]
const MTP_G = [1, 1, 1, 2, 8, 8, 8, 8, 8, 7, 4, 3]
const DF_BEST = [1658, 2222, 2868, 4044, 7891, 15241, 28301, 50046, 80753, 102539, 112607, 114318]
const DF_ADP = [1658, 2222, 2868, 4048, 7896, 15253, 28316, 50065, 80944, 102874, 112846, 114448]
const DF_G = [3, 2, 2, 15, 15, 15, 15, 15, 14, 8, 5, 4]

const ratio = (a: number[]) => a.map((v, i) => ({ x: CONC[i], y: v / NOSPEC[i] }))

export default function PricingEnvelope() {
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
    const pageBg = theme.isDark ? '#0a0a0a' : '#ffffff'

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    // best-in-hindsight as the envelope line (no markers of its own)...
    const lineDS = (label: string, data: { x: number; y: number }[], color: string) => ({
      label,
      data,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0.25,
      order: 1,
    })
    // ...the adaptive policy as discrete markers riding on it. The markers
    // landing on the line ARE the envelope match; a hollow ring reads as a
    // measured point sitting on a reference rather than a fattened line.
    const ringDS = (data: { x: number; y: number }[], color: string) => ({
      label: '',
      data,
      showLine: false,
      pointStyle: 'circle',
      pointRadius: 4.5,
      pointHoverRadius: 6,
      pointBackgroundColor: pageBg,
      pointBorderColor: color,
      pointBorderWidth: 2,
      order: 0,
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          lineDS('MTP', ratio(MTP_BEST), mtp),
          lineDS('DFlash', ratio(DF_BEST), df),
          ringDS(ratio(MTP_ADP), mtp),
          ringDS(ratio(DF_ADP), df),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        layout: { padding: { top: 18 } },
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
              title: (items) => (items.length ? `concurrency ${items[0].parsed.x}` : ''),
              label: (ctx) => {
                const x = ctx.parsed.x
                const y = ctx.parsed.y
                if (typeof x !== 'number' || typeof y !== 'number') return ''

                const i = CONC.indexOf(x)
                const g = ctx.dataset.label === 'MTP' ? MTP_G[i] : DF_G[i]
                return `  ${ctx.dataset.label}: ${y.toFixed(2)}× no-spec  (γ${g})`
              },
            },
          },
        },
        scales: {
          x: {
            type: 'logarithmic',
            min: 1,
            max: 2048,
            title: axisTitle('decode batch B (concurrency)'),
            grid: { color: theme.grid },
            afterBuildTicks: (axis) => {
              axis.ticks = CONC.map((v) => ({ value: v }))
            },
            ticks: { ...tickStyle, callback: (v) => `${v}` },
          },
          y: {
            type: 'linear',
            min: 1,
            max: 3.8,
            title: axisTitle('throughput vs no-spec (×)'),
            grid: { color: theme.grid },
            ticks: { ...tickStyle, stepSize: 0.5, callback: (v) => `${v}×` },
          },
        },
      },
      plugins: [
        {
          // No-spec baseline at 100%.
          id: 'nospec',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart
            const y = scales.y.getPixelForValue(1)
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.setLineDash([4, 4])
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(chartArea.left, y)
            ctx.lineTo(chartArea.right, y)
            ctx.stroke()
            ctx.setLineDash([])
            ctx.fillStyle = theme.mutedForeground
            ctx.font = `11px ${theme.fontFamily}`
            ctx.textAlign = 'left'
            ctx.fillText('no speculation', chartArea.left + 4, y - 4)
            ctx.restore()
          },
        },
        {
          // γ* over MTP markers, under DFlash markers, so they don't collide.
          id: 'gammaLabels',
          afterDatasetsDraw(chart) {
            const { ctx, scales, chartArea } = chart
            ctx.save()
            ctx.font = `600 10px ${theme.fontFamily}`
            ctx.textAlign = 'center'
            CONC.forEach((c, i) => {
              const x = scales.x.getPixelForValue(c)
              const my = scales.y.getPixelForValue(MTP_BEST[i] / NOSPEC[i])
              const dy = scales.y.getPixelForValue(DF_BEST[i] / NOSPEC[i])
              ctx.fillStyle = `hsl(220, 70%, ${mtpL}%)`
              ctx.fillText(`γ${MTP_G[i]}`, x, Math.max(chartArea.top + 9, my - 9))
              ctx.fillStyle = `hsl(28, 80%, ${dfL}%)`
              ctx.fillText(`γ${DF_G[i]}`, x, Math.min(chartArea.bottom - 3, dy + 16))
            })
            ctx.restore()
          },
        },
        {
          // Key: a line sample and a ring sample so the encoding is self-explanatory.
          id: 'envelopeKey',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea } = chart
            const x = chartArea.left + 6
            const y = chartArea.top + 12
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.lineWidth = 2
            ctx.beginPath()
            ctx.moveTo(x, y)
            ctx.lineTo(x + 18, y)
            ctx.stroke()
            ctx.beginPath()
            ctx.arc(x + 9, y + 16, 4.5, 0, 2 * Math.PI)
            ctx.fillStyle = theme.isDark ? '#0a0a0a' : '#ffffff'
            ctx.fill()
            ctx.strokeStyle = theme.mutedForeground
            ctx.stroke()
            ctx.fillStyle = theme.mutedForeground
            ctx.font = `11px ${theme.fontFamily}`
            ctx.textAlign = 'left'
            ctx.fillText('best fixed γ (hindsight)', x + 24, y + 3)
            ctx.fillText('adaptive (priced)', x + 24, y + 19)
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
    <div class="my-6" style="position: relative; height: 420px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
