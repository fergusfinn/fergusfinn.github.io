import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'
import { models, ALL_FORMATS, type Model } from '../data/weight-entropy-models'
import { useChartTheme, applyChartDefaults, type ChartTheme } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

const FORMAT_ALLOC: Record<string, [number, number, number]> = {
  BF16: [1, 8, 7],
  FP8: [1, 4, 3],
  MXFP4: [1, 2, 1],
  MXFP8: [1, 4, 3],
  INT4: [1, 0, 3],
  NVFP4: [1, 2, 1],
}

interface Props {
  title?: string
  defaultFormats?: string[]
}

export default function EntropyBars({
  title = '',
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

    const groups: Record<number, Model[]> = {}
    for (const m of visible) {
      if (!groups[m.bitWidth]) groups[m.bitWidth] = []
      groups[m.bitWidth].push(m)
    }
    const sortedWidths = Object.keys(groups).map(Number).sort((a, b) => a - b)

    const labels: string[] = []
    const modelAtIdx: (Model | null)[] = []
    const mantUsed: number[] = []
    const mantUnused: number[] = []
    const expUsed: number[] = []
    const expUnused: number[] = []
    const signUsed: number[] = []
    const signUnused: number[] = []
    const scaleUsed: number[] = []
    const scaleUnused: number[] = []

    let idx = 0
    for (const bw of sortedWidths) {
      if (idx > 0) {
        labels.push('')
        modelAtIdx.push(null)
        mantUsed.push(0); mantUnused.push(0)
        expUsed.push(0); expUnused.push(0)
        signUsed.push(0); signUnused.push(0)
        scaleUsed.push(0); scaleUnused.push(0)
        idx++
      }
      for (const m of groups[bw]) {
        const alloc = FORMAT_ALLOC[m.format] || [1, 4, 3]
        const [sAlloc, eAlloc, mAlloc] = alloc
        labels.push(`${m.shortLabel} (${m.format})`)
        modelAtIdx.push(m)
        mantUsed.push(Math.min(m.mantissa, mAlloc))
        mantUnused.push(Math.max(0, mAlloc - m.mantissa))
        expUsed.push(Math.min(m.exponent, eAlloc))
        expUnused.push(Math.max(0, eAlloc - m.exponent))
        signUsed.push(Math.min(m.sign, sAlloc))
        signUnused.push(Math.max(0, sAlloc - m.sign))
        const sAllocAmort = m.scaleAllocBits ?? 0
        const sUsedAmort = m.scaleEntropy ?? 0
        scaleUsed.push(sUsedAmort)
        scaleUnused.push(Math.max(0, sAllocAmort - sUsedAmort))
        idx++
      }
    }

    chartInstance.current = new Chart(chartRef.current, {
      type: 'bar',
      plugins: [ChartDataLabels],
      data: {
        labels,
        datasets: [
          { label: 'Mantissa', data: mantUsed, backgroundColor: '#3b82f6', borderWidth: 0 },
          { label: '_unused', data: mantUnused, backgroundColor: theme.unusedFill, borderWidth: 0 },
          { label: 'Exponent', data: expUsed, backgroundColor: '#f59e0b', borderWidth: 0 },
          { label: '_unused', data: expUnused, backgroundColor: theme.unusedFill, borderWidth: 0 },
          { label: 'Sign', data: signUsed, backgroundColor: '#10b981', borderWidth: 0 },
          { label: '_unused', data: signUnused, backgroundColor: theme.unusedFill, borderWidth: 0 },
          { label: 'Scale (amortized)', data: scaleUsed, backgroundColor: '#8b5cf6', borderWidth: 0 },
          {
            label: 'Unused budget',
            data: scaleUnused,
            backgroundColor: theme.unusedFill,
            borderWidth: 0,
            datalabels: {
              display: (ctx: any) => modelAtIdx[ctx.dataIndex] !== null,
              anchor: 'end' as const,
              align: 'end' as const,
              formatter: (_: number, ctx: any) => {
                const m = modelAtIdx[ctx.dataIndex]
                if (!m) return ''
                const used = m.sign + m.exponent + m.mantissa + (m.scaleEntropy ?? 0)
                const alloc = m.bitWidth + (m.scaleAllocBits ?? 0)
                const waste = (1 - used / alloc) * 100
                return `${waste.toFixed(0)}%`
              },
              color: theme.mutedForeground,
              font: { size: 9 },
            },
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        layout: { padding: { top: 16 } },
        plugins: {
          datalabels: { display: false },
          title: { display: !!title, text: title, font: { size: 15, weight: 'normal' }, padding: { bottom: 16 } },
          legend: {
            position: 'bottom',
            labels: { filter: (item) => !item.text.startsWith('_'), padding: 16 },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const m = modelAtIdx[ctx.dataIndex]
                if (!m) return ''
                const alloc = FORMAT_ALLOC[m.format] || [1, 4, 3]
                const ds = ctx.dataset.label || ''
                if (ds === 'Mantissa') return `Mantissa: ${m.mantissa.toFixed(2)}/${alloc[2]} bits`
                if (ds === 'Exponent') return `Exponent: ${m.exponent.toFixed(2)}/${alloc[1]} bits`
                if (ds === 'Sign') return `Sign: ${m.sign.toFixed(2)}/${alloc[0]} bits`
                if (ds === 'Scale (amortized)') {
                  const sa = m.scaleAllocBits ?? 0
                  const se = m.scaleEntropy ?? 0
                  return sa > 0 ? `Scale: ${se.toFixed(2)}/${sa.toFixed(2)} bits/elem` : ''
                }
                return ''
              },
              afterBody: (items) => {
                const m = modelAtIdx[items[0].dataIndex]
                if (!m) return ''
                const total = m.sign + m.exponent + m.mantissa + (m.scaleEntropy ?? 0)
                const alloc = m.bitWidth + (m.scaleAllocBits ?? 0)
                return `Total: ${total.toFixed(1)}/${alloc.toFixed(1)} bits/elem`
              },
            },
          },
        },
        scales: {
          x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 90, minRotation: 45, autoSkip: false, font: { size: 9 } } },
          y: { stacked: true, title: { display: true, text: 'Bits per element' }, grid: { color: theme.grid }, grace: '8%' },
        },
      },
    })

    return () => { chartInstance.current?.destroy() }
  }, [title, visible, theme])

  const barCount = visible.length + Math.max(0, new Set(visible.map((m) => m.bitWidth)).size - 1)
  const height = Math.max(280, Math.min(460, barCount * 22 + 140))

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
      <div style={{ position: 'relative', width: '100%', height: `${height}px` }}>
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
