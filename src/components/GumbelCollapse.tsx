import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { models, ALL_FORMATS, type Model } from '../data/weight-entropy-models'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
  Chart.defaults.font.family = "'Source Sans 3', sans-serif"
  Chart.defaults.font.size = 13
}

const LAB_HUES: Record<string, number> = {
  Qwen: 220, Google: 0, StepFun: 275, Zhipu: 35, OpenAI: 160,
  DeepSeek: 190, Moonshot: 320, MiniMax: 350, NVIDIA: 90,
}

interface Props {
  title?: string
  defaultFormats?: string[]
}

export default function GumbelCollapse({
  title = 'Normalized magnitude distributions',
  defaultFormats = ['BF16', 'FP8', 'MXFP8'],
}: Props) {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const [active, setActive] = useState<Set<string>>(new Set(defaultFormats))

  const visible = useMemo(
    () => models.filter((m) => active.has(m.format) && m.gumbelData),
    [active],
  )

  useEffect(() => {
    if (!chartRef.current) return
    if (chartInstance.current) chartInstance.current.destroy()

    // GEV reference curve (fitted from the dataset; see note in repo history)
    const c = -0.029
    const loc_z = -0.004
    const scale_z = 0.956
    const bin_width = 1.0 / 1.30
    const gev_points: { x: number; y: number }[] = []
    for (let z = -12; z <= 4; z += 0.05) {
      const arg = (-z - loc_z) / scale_z
      const t_base = 1 + c * arg
      if (t_base <= 0) continue
      const t = Math.pow(t_base, -1 / c)
      const pdf = (1 / scale_z) * Math.pow(t, c + 1) * Math.exp(-t)
      const pct = pdf * bin_width * 100
      if (pct > 0.003) gev_points.push({ x: z, y: pct })
    }

    const labCounts: Record<string, number> = {}
    for (const m of visible) labCounts[m.lab] = (labCounts[m.lab] || 0) + 1
    const labRank: Record<string, number> = {}

    const datasets: any[] = visible.map((m) => {
      const rank = labRank[m.lab] ?? 0
      labRank[m.lab] = rank + 1
      const total = labCounts[m.lab]
      const t = total <= 1 ? 0.5 : rank / (total - 1)
      const lightness = 60 - t * 25
      const hue = LAB_HUES[m.lab] ?? 0
      return {
        label: m.label,
        data: m.gumbelData ?? [],
        backgroundColor: `hsl(${hue}, 70%, ${lightness}%)`,
        borderColor: `hsl(${hue}, 70%, ${lightness}%)`,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointHitRadius: 10,
        showLine: false,
        order: 1,
      }
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: !!title, text: title, font: { size: 15, weight: 'normal' }, padding: { bottom: 16 } },
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}% at z=${ctx.parsed.x.toFixed(2)}`,
            },
          },
        },
        scales: {
          x: { title: { display: true, text: 'Normalized magnitude: (log₂|w| − μ) / β' }, min: -10, max: 4, grid: { color: '#e5e7eb' } },
          y: { title: { display: true, text: 'Frequency (%)' }, min: 0, grid: { color: '#e5e7eb' } },
        },
      },
    })

    return () => { chartInstance.current?.destroy() }
  }, [title, visible])

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
      <FormatFilter active={active} onToggle={toggle} />
      <div style={{ position: 'relative', width: '100%', height: '400px' }}>
        <canvas ref={chartRef} />
      </div>
    </div>
  )
}

function FormatFilter({
  active,
  onToggle,
}: {
  active: Set<string>
  onToggle: (fmt: string) => void
}) {
  // Only show formats that have gumbel data available
  const available = ALL_FORMATS.filter((fmt) => models.some((m) => m.format === fmt && m.gumbelData))
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
      {available.map((fmt) => {
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
              fontFamily: "'Source Sans 3', sans-serif",
              background: on ? '#e5e7eb' : 'transparent',
              color: on ? '#1f2937' : '#9ca3af',
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
