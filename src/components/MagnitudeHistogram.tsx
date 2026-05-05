import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { models, ALL_FORMATS, type Model } from '../data/weight-entropy-models'
import { useChartTheme, applyChartDefaults, type ChartTheme } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

const LAB_HUES: Record<string, number> = {
  Qwen: 220, Google: 0, StepFun: 275, Zhipu: 35, OpenAI: 160,
  DeepSeek: 190, Moonshot: 320, MiniMax: 350, NVIDIA: 90,
}

function colorFor(m: Model, sizeRank: number, totalInLab: number): string {
  const hue = LAB_HUES[m.lab] ?? 0
  const t = totalInLab <= 1 ? 0.5 : sizeRank / (totalInLab - 1)
  const lightness = 65 - t * 30
  return `hsl(${hue}, 70%, ${lightness}%)`
}

interface Props {
  title?: string
  xLabel?: string
  defaultFormats?: string[]
}

export default function MagnitudeHistogram({
  title = '',
  xLabel = 'log₂ |weight|',
  defaultFormats = ['BF16'],
}: Props) {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const [active, setActive] = useState<Set<string>>(new Set(defaultFormats))
  const theme = useChartTheme()

  const visible = useMemo(
    () => models.filter((m) => active.has(m.format)),
    [active],
  )

  useEffect(() => {
    if (!chartRef.current) return
    if (chartInstance.current) chartInstance.current.destroy()
    applyChartDefaults(theme)

    const allKeys = new Set<number>()
    for (const m of visible) {
      for (const k of Object.keys(m.magnitudeData)) allKeys.add(Number(k))
    }
    const labels = [...allKeys].sort((a, b) => a - b)

    const labCounts: Record<string, number> = {}
    for (const m of visible) labCounts[m.lab] = (labCounts[m.lab] || 0) + 1
    const labRank: Record<string, number> = {}

    const datasets = visible.map((m) => {
      const rank = labRank[m.lab] ?? 0
      labRank[m.lab] = rank + 1
      const color = colorFor(m, rank, labCounts[m.lab])
      const dashed = m.format !== 'BF16'
      return {
        label: m.label,
        data: labels.map((k) => m.magnitudeData[k] ?? 0),
        borderColor: color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: dashed ? [6, 3] : [],
        pointRadius: 2,
        pointHitRadius: 15,
        pointHoverRadius: 4,
        tension: 0.3,
        fill: false,
      }
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: { labels: labels.map(String), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        plugins: {
          title: { display: !!title, text: title, font: { size: 15, weight: 'normal' }, padding: { bottom: 16 } },
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: { title: { display: !!xLabel, text: xLabel }, grid: { display: false } },
          y: { title: { display: true, text: 'Frequency (%)' }, min: 0, grid: { color: theme.grid } },
        },
      },
    })

    return () => { chartInstance.current?.destroy() }
  }, [title, xLabel, visible, theme])

  const toggle = (fmt: string) => {
    setActive((prev) => {
      const next = new Set(prev)
      if (next.has(fmt)) next.delete(fmt)
      else next.add(fmt)
      return next
    })
  }

  return (
    <div>
      <FormatFilter active={active} onToggle={toggle} theme={theme} />
      <div style={{ position: 'relative', width: '100%', height: '380px' }}>
        <canvas ref={chartRef} />
      </div>
    </div>
  )
}

function FormatFilter({
  active,
  onToggle,
  theme,
}: {
  active: Set<string>
  onToggle: (fmt: string) => void
  theme: ChartTheme
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
      {ALL_FORMATS.map((fmt) => {
        const on = active.has(fmt)
        return (
          <button
            key={fmt}
            onClick={() => onToggle(fmt)}
            style={{
              cursor: 'pointer',
              padding: '2px 8px',
              border: 'none',
              borderRadius: '3px',
              fontSize: '11px',
              fontWeight: 500,
              fontFamily: theme.fontFamily,
              background: on ? theme.buttonActiveBg : 'transparent',
              color: on ? theme.buttonActiveText : theme.buttonInactiveText,
              userSelect: 'none',
            }}
          >
            {fmt}
          </button>
        )
      })}
    </div>
  )
}
