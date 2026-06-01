import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'
import { useChartTheme, applyChartDefaults } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// DSV4-Flash on MI300X, vllm bench serve at ISL/OSL = 512/512. Throughput is
// per GPU so it is comparable across card counts. "Stock
// vLLM" is upstream main without any of the patches in this post: it does not
// produce output. "After bring-up" is the first throughput number once
// correctness is sorted. "After tuning" is the best stable 512/512 run.
const DATA: { stage: string; throughput: number; isResult: boolean }[] = [
  { stage: 'Stock vLLM', throughput: 0, isResult: false },
  { stage: 'After bring-up', throughput: 2485, isResult: false },
  { stage: 'After tuning', throughput: 2699, isResult: true },
]

const ACCENT_HUE = 220

export default function Mi300xThroughputBars() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    if (chartInstance.current) chartInstance.current.destroy()
    applyChartDefaults(theme)

    const accent = `hsl(${ACCENT_HUE}, 70%, ${theme.isDark ? 60 : 45}%)`
    const muted = theme.isDark
      ? 'rgba(250, 250, 250, 0.35)'
      : 'rgba(12, 12, 12, 0.35)'

    const labels = DATA.map((d) => d.stage)
    const values = DATA.map((d) => d.throughput)
    const colors = DATA.map((d) => (d.isResult ? accent : muted))

    chartInstance.current = new Chart(chartRef.current, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'output throughput',
            data: values,
            backgroundColor: colors,
            borderColor: colors,
            borderWidth: 0,
            barPercentage: 0.6,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { right: 64, top: 24 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ctx.parsed.x == null
                  ? ''
                  : `  ${ctx.parsed.x.toFixed(0)} tok/s/GPU`,
            },
          },
          datalabels: {
            anchor: 'end',
            align: 'right',
            offset: 6,
            color: theme.foreground,
            font: { family: theme.fontFamily, weight: 600 },
            formatter: (v: number) => `${v} tok/s/GPU`,
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            suggestedMax: 2900,
            title: {
              display: true,
              text: 'output throughput (tok/s per GPU)',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            grid: { color: theme.grid },
          },
          y: {
            ticks: {
              color: theme.foreground,
              font: { family: theme.fontFamily },
              autoSkip: false,
            },
            grid: { display: false },
          },
        },
      },
      plugins: [ChartDataLabels],
    })

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  return (
    <div class="my-6" style="position: relative; height: 240px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
