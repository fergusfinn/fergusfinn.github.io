import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Distribution of accept lengths (tokens committed per round) for each head. The
// shape is what the per-depth average hides: a decay from zero plus a spike at
// the deepest draft, the censoring spike of rounds the drafter got entirely right
// where only the draft length stopped it (17.8% of MTP rounds commit all 8; the
// DFlash spike at 15 is smaller because its deeper drafts truncate fewer rounds).
// A round is mostly a near-miss or a clean sweep, which is the variation the
// confidence signal sorts out. Qwen3.6-35B-A3B, SPEED-Bench, temp 0.6.

const COMMITS = Array.from({ length: 16 }, (_, i) => i) // 0..15
const MTP = [18.7, 19.5, 13.8, 9.1, 7.0, 6.3, 4.1, 3.8, 17.8]
const DF = [21.1, 21.0, 14.9, 9.8, 7.7, 4.7, 3.3, 3.6, 2.2, 2.4, 1.7, 0.9, 1.3, 0.7, 0.5, 4.0]

export default function AcceptLengthHist() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const mtp = `hsla(220, 70%, ${theme.isDark ? 62 : 48}%, 0.85)`
    const df = `hsla(28, 80%, ${theme.isDark ? 60 : 50}%, 0.85)`
    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    chartInstance.current = new Chart(chartRef.current, {
      type: 'bar',
      data: {
        labels: COMMITS.map((c) => `${c}`),
        datasets: [
          {
            label: 'MTP (D=8)',
            data: COMMITS.map((c) => MTP[c] ?? null),
            backgroundColor: mtp,
            borderWidth: 0,
          },
          {
            label: 'DFlash (D=16)',
            data: COMMITS.map((c) => DF[c] ?? null),
            backgroundColor: df,
            borderWidth: 0,
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
            labels: { boxWidth: 14, boxHeight: 12, color: theme.foreground, font: { family: theme.fontFamily } },
          },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `${items[0].label} tokens committed` : ''),
              label: (ctx) => `  ${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(1)}% of rounds`,
            },
          },
        },
        scales: {
          x: {
            title: axisTitle('tokens committed in the round'),
            grid: { display: false },
            ticks: tickStyle,
          },
          y: {
            type: 'linear',
            min: 0,
            title: axisTitle('% of rounds'),
            grid: { color: theme.grid },
            ticks: { ...tickStyle, callback: (v) => `${v}%` },
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
    <div class="my-6" style="position: relative; height: 340px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
