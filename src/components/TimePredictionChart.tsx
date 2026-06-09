import { useEffect, useRef, useState } from 'preact/hooks'
import { DECADES, MATRIX, OVERALL } from './date-prediction-data'

type Row = { actual: number; props: number[] }

let _rows: Row[] | null = null
function getRows(): Row[] {
  if (!_rows) {
    _rows = MATRIX.trim().split('\n').map(line => {
      const parts = line.split('|')
      return { actual: Number(parts[0]), props: parts.slice(1).map(Number) }
    })
  }
  return _rows
}

export default function TimePredictionChart() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    const rows = getRows()
    const nDec = DECADES.length
    const pad = { top: 28, right: 16, bottom: 48, left: 52 }
    const chartW = w - pad.left - pad.right
    const chartH = h - pad.top - pad.bottom
    const cellW = chartW / nDec
    const cellH = chartH / rows.length

    // Background
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, w, h)

    // Draw cells
    for (let r = 0; r < rows.length; r++) {
      for (let c = 0; c < nDec; c++) {
        const val = rows[r].props[c]
        const x = pad.left + c * cellW
        const y = pad.top + r * cellH

        // Color: blue intensity based on proportion
        // Diagonal cells get a slightly different treatment
        const isDiag = rows[r].actual === DECADES[c]
        if (val > 0) {
          const intensity = Math.min(val / 0.5, 1) // cap at 50% for full intensity
          const alpha = 0.08 + intensity * 0.92
          if (isDiag) {
            ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`
          } else {
            ctx.fillStyle = `rgba(59, 130, 246, ${alpha * 0.7})`
          }
        } else {
          ctx.fillStyle = '#0f172a'
        }
        ctx.fillRect(x + 0.5, y + 0.5, cellW - 1, cellH - 1)

        // Show percentage text in cells with enough value
        if (val >= 0.05) {
          const pct = Math.round(val * 100)
          ctx.fillStyle = val > 0.3 ? '#ffffff' : '#94a3b8'
          ctx.font = `${Math.min(cellW * 0.35, 11)}px ui-monospace, monospace`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(`${pct}`, x + cellW / 2, y + cellH / 2)
        }
      }
    }

    // Diagonal highlight border
    ctx.strokeStyle = '#3b82f640'
    ctx.lineWidth = 1
    for (let r = 0; r < rows.length; r++) {
      const c = DECADES.indexOf(rows[r].actual)
      if (c >= 0) {
        ctx.strokeRect(pad.left + c * cellW, pad.top + r * cellH, cellW, cellH)
      }
    }

    // Axis labels
    ctx.fillStyle = '#94a3b8'
    ctx.font = '10px ui-monospace, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (let c = 0; c < nDec; c++) {
      const x = pad.left + c * cellW + cellW / 2
      ctx.save()
      ctx.translate(x, pad.top + chartH + 4)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'right'
      ctx.fillText(`${DECADES[c]}s`, 0, 0)
      ctx.restore()
    }

    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let r = 0; r < rows.length; r++) {
      ctx.fillText(`${rows[r].actual}s`, pad.left - 4, pad.top + r * cellH + cellH / 2)
    }

    // Axis titles
    ctx.fillStyle = '#64748b'
    ctx.font = '11px ui-monospace, monospace'
    ctx.textAlign = 'center'
    ctx.fillText('guessed decade', pad.left + chartW / 2, h - 4)

    ctx.save()
    ctx.translate(10, pad.top + chartH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('actual decade', 0, 0)
    ctx.restore()

  }, [])

  return (
    <div style={{ margin: '1.5rem 0', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '440px', borderRadius: '8px', cursor: 'crosshair' }}
        onMouseMove={(e) => {
          const canvas = canvasRef.current
          if (!canvas) return
          const rect = canvas.getBoundingClientRect()
          const pad = { top: 28, right: 16, bottom: 48, left: 52 }
          const chartW = rect.width - pad.left - pad.right
          const chartH = rect.height - pad.top - pad.bottom
          const rows = getRows()
          const nDec = DECADES.length

          const mx = e.clientX - rect.left - pad.left
          const my = e.clientY - rect.top - pad.top
          const col = Math.floor(mx / (chartW / nDec))
          const row = Math.floor(my / (chartH / rows.length))

          if (col >= 0 && col < nDec && row >= 0 && row < rows.length) {
            const val = rows[row].props[col]
            const pct = (val * 100).toFixed(1)
            const actual = rows[row].actual
            const guessed = DECADES[col]
            setTooltip({
              x: e.clientX - rect.left,
              y: e.clientY - rect.top,
              text: `${actual}s → guessed ${guessed}s: ${pct}%`
            })
          } else {
            setTooltip(null)
          }
        }}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div style={{
          position: 'absolute',
          left: `${tooltip.x + 12}px`,
          top: `${tooltip.y - 28}px`,
          background: '#1e293bee',
          color: '#e2e8f0',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '11px',
          fontFamily: 'ui-monospace, monospace',
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
        }}>
          {tooltip.text}
        </div>
      )}
      <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.5rem', textAlign: 'center' }}>
        {OVERALL.n.toLocaleString()} speeches. Exact decade: {(OVERALL.exactPct * 100).toFixed(0)}%.
        Within one decade: {(OVERALL.within1Pct * 100).toFixed(0)}%. Cell values show percentages.
      </div>
    </div>
  )
}
