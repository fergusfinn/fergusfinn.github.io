import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'
import { useChartTheme, applyChartDefaults } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// All measured on the same RTX 4090, 1 GiB working set. torch.add is bf16
// (torch.add doesn't dispatch on fp8); fused is fp8 logical throughput.
// Same 2R + 1W access pattern in both cases.
const DATA: { kernel: string; bandwidth: number; isResult: boolean }[] = [
  { kernel: 'torch.add (bf16)', bandwidth: 922, isResult: false },
  { kernel: 'fused tANS vecadd (logical FP8)', bandwidth: 993, isResult: true },
]

const ACCENT_HUE = 220

export default function KernelBandwidthBars() {
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

    const labels = DATA.map((d) => d.kernel)
    const values = DATA.map((d) => d.bandwidth)
    const colors = DATA.map((d) => (d.isResult ? accent : muted))

    chartInstance.current = new Chart(chartRef.current, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'effective bandwidth',
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
        layout: { padding: { right: 56, top: 24 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ctx.parsed.x == null
                  ? ''
                  : `  ${ctx.parsed.x.toFixed(0)} GB/s`,
            },
          },
          datalabels: {
            anchor: 'end',
            align: 'right',
            offset: 6,
            color: theme.foreground,
            font: { family: theme.fontFamily, weight: 600 },
            formatter: (v: number) => `${v} GB/s`,
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            suggestedMax: 1050,
            title: {
              display: true,
              text: 'effective bandwidth (GB/s)',
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
    <div class="my-6" style="position: relative; height: 200px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
