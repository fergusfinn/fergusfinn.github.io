import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Measured on an RTX 4090, 1 GiB working set, just bench bench_vecadd.py.
// N is FP8 bytes per stream.
const DATA: { n: number; amplification: number; bitsPerFp8: number }[] = [
  { n: 128, amplification: 1.063, bitsPerFp8: 7.443 },
  { n: 256, amplification: 1.081, bitsPerFp8: 7.179 },
  { n: 512, amplification: 1.099, bitsPerFp8: 7.034 },
  { n: 1024, amplification: 1.002, bitsPerFp8: 6.949 },
  { n: 2048, amplification: 0.833, bitsPerFp8: 6.898 },
]

const AMP_HUE = 220
const BITS_HUE = 0

export default function AmplificationVsStreamLength() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    if (chartInstance.current) chartInstance.current.destroy()
    applyChartDefaults(theme)

    const ampColor = `hsl(${AMP_HUE}, 70%, ${theme.isDark ? 65 : 45}%)`
    const bitsColor = `hsl(${BITS_HUE}, 70%, ${theme.isDark ? 65 : 45}%)`

    const labels = DATA.map((d) => d.n.toString())
    const amplifications = DATA.map((d) => d.amplification)
    const bitsPerFp8 = DATA.map((d) => d.bitsPerFp8)

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'amplification (× raw)',
            data: amplifications,
            borderColor: ampColor,
            backgroundColor: ampColor,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.2,
            yAxisID: 'yAmp',
            borderWidth: 2,
          },
          {
            label: 'compression (bits per FP8)',
            data: bitsPerFp8,
            borderColor: bitsColor,
            backgroundColor: bitsColor,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.2,
            yAxisID: 'yBits',
            borderWidth: 2,
            borderDash: [6, 4],
          },
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
            labels: {
              boxWidth: 14,
              boxHeight: 2,
              color: theme.foreground,
              font: { family: theme.fontFamily },
            },
          },
          tooltip: {
            callbacks: {
              title: (items) =>
                items.length > 0 ? `N = ${items[0].label} FP8 / stream` : '',
              label: (ctx) => {
                const v = ctx.parsed.y
                if (v == null) return ''
                if (ctx.dataset.yAxisID === 'yAmp') {
                  return `  amplification: ${v.toFixed(3)}× raw`
                }
                return `  compression:   ${v.toFixed(2)} bits / FP8`
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'stream length N (FP8 bytes / stream)',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            grid: { color: theme.grid },
          },
          yAmp: {
            type: 'linear',
            position: 'left',
            title: {
              display: true,
              text: 'amplification (× raw)',
              color: ampColor,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: ampColor,
              font: { family: theme.fontFamily },
              callback: (v) => (typeof v === 'number' ? v.toFixed(2) : v),
            },
            grid: { color: theme.grid },
            suggestedMin: 0.8,
            suggestedMax: 1.15,
          },
          yBits: {
            type: 'linear',
            position: 'right',
            title: {
              display: true,
              text: 'compression (bits / FP8)',
              color: bitsColor,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: bitsColor,
              font: { family: theme.fontFamily },
              callback: (v) => (typeof v === 'number' ? v.toFixed(2) : v),
            },
            grid: { drawOnChartArea: false },
            suggestedMin: 6.8,
            suggestedMax: 7.5,
          },
        },
      },
      plugins: [
        {
          id: 'breakEvenLine',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart
            const y = scales.yAmp.getPixelForValue(1.0)
            if (y < chartArea.top || y > chartArea.bottom) return
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.setLineDash([4, 4])
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(chartArea.left, y)
            ctx.lineTo(chartArea.right, y)
            ctx.stroke()
            ctx.fillStyle = theme.mutedForeground
            ctx.font = `11px ${theme.fontFamily}`
            ctx.textAlign = 'left'
            ctx.textBaseline = 'bottom'
            ctx.fillText('break-even (raw kernel)', chartArea.left + 6, y - 2)
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
    <div class="my-6" style="position: relative; height: 320px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
