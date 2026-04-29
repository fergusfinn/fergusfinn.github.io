import { useEffect, useRef } from 'preact/hooks'

const DECADES = ['1870s','1880s','1890s','1900s','1910s','1920s','1930s','1940s','1950s','1960s','1970s','1980s','1990s','2000s']

// Percentage share of all foreign country mentions per decade
// Sorted by peak decade — reads chronologically top to bottom
const COUNTRIES: { name: string; data: number[]; color: string }[] = [
  { name: 'Britain',     data: [27.3,28.9,31.6,17.0,25.4,24.2,26.9,26.7,13.1,7.4,9.7,7.2,3.7,10.3],  color: '#1d4ed8' },
  { name: 'France',      data: [15.2,14.8,8.2,9.5,17.1,16.8,12.3,9.0,10.0,4.9,5.7,5.7,6.4,6.6],      color: '#7c3aed' },
  { name: 'Cuba',        data: [3.0,5.6,19.3,14.5,3.9,4.5,3.2,0.7,0.7,7.4,6.6,6.9,3.7,6.2],          color: '#059669' },
  { name: 'Germany',     data: [7.6,11.3,7.0,7.1,16.7,14.8,13.2,17.0,10.0,8.4,7.5,10.9,12.7,10.3],    color: '#475569' },
  { name: 'Japan',       data: [7.6,5.6,1.8,9.1,6.1,8.4,9.6,14.0,7.6,4.4,5.3,9.5,12.0,7.4],          color: '#ec4899' },
  { name: 'Russia/USSR', data: [0.0,4.9,5.8,3.7,7.9,4.8,10.0,18.7,32.5,27.9,27.7,42.4,22.1,11.1],     color: '#dc2626' },
  { name: 'Vietnam',     data: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.4,29.3,23.6,5.2,7.1,7.0],           color: '#0891b2' },
  { name: 'China',       data: [12.1,7.7,6.4,12.4,4.4,7.1,7.3,4.0,10.4,3.7,5.0,2.6,13.1,21.0],        color: '#ea580c' },
  { name: 'Mexico',      data: [21.2,10.6,11.1,16.2,8.3,8.7,3.7,4.7,5.2,3.3,3.1,4.9,10.9,11.1],       color: '#65a30d' },
  { name: 'Canada',      data: [6.1,10.6,8.8,10.4,10.1,10.6,13.7,5.3,8.0,3.3,5.7,4.9,8.2,9.1],        color: '#92400e' },
]

const MAX_VAL = 42.4 // Russia 1980s

// Find local maxima (higher than both neighbours), sorted by value descending.
// Endpoints count as peaks if they're higher than their one neighbour.
// Falls back to global top-N if fewer than n local maxima exist.
function findPeaks(data: number[], n: number): number[] {
  const peaks: { v: number; i: number }[] = []
  for (let i = 0; i < data.length; i++) {
    const prev = i > 0 ? data[i - 1] : -1
    const next = i < data.length - 1 ? data[i + 1] : -1
    if (data[i] > prev && data[i] > next && data[i] > 0) {
      peaks.push({ v: data[i], i })
    }
  }
  peaks.sort((a, b) => b.v - a.v)
  if (peaks.length >= n) return peaks.slice(0, n).map(x => x.i)
  // Fallback: top N by value
  return data
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, n)
    .map(x => x.i)
}

function drawSparkline(canvas: HTMLCanvasElement, data: number[], color: string) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  canvas.width = w * dpr
  canvas.height = h * dpr
  ctx.scale(dpr, dpr)

  const pad = { top: 14, bottom: 3 }
  const chartH = h - pad.top - pad.bottom
  const step = w / (data.length - 1)

  const pts = data.map((v, i) => ({
    x: i * step,
    y: pad.top + chartH * (1 - v / MAX_VAL),
  }))

  function traceCurve() {
    ctx.moveTo(pts[0].x, pts[0].y)
    for (let i = 1; i < pts.length; i++) {
      const cp = (pts[i].x - pts[i-1].x) * 0.4
      ctx.bezierCurveTo(pts[i-1].x + cp, pts[i-1].y, pts[i].x - cp, pts[i].y, pts[i].x, pts[i].y)
    }
  }

  // Fill
  ctx.beginPath()
  ctx.moveTo(pts[0].x, h - pad.bottom)
  ctx.lineTo(pts[0].x, pts[0].y)
  traceCurve()
  ctx.lineTo(pts[pts.length-1].x, h - pad.bottom)
  ctx.closePath()
  ctx.fillStyle = color + '1a'
  ctx.fill()

  // Stroke
  ctx.beginPath()
  traceCurve()
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.stroke()

  // Top 2 peaks — dots + labels
  const peaks = findPeaks(data, 2)
  for (let pi = 0; pi < peaks.length; pi++) {
    const idx = peaks[pi]
    const pt = pts[idx]
    const isPrimary = pi === 0

    // Dot
    ctx.beginPath()
    ctx.arc(pt.x, pt.y, isPrimary ? 4 : 3, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
    if (isPrimary) {
      ctx.beginPath()
      ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2)
      ctx.fillStyle = '#fff'
      ctx.fill()
    }

    // Label above the dot
    const label = DECADES[idx].slice(0, 4)
    ctx.font = `${isPrimary ? 'bold ' : ''}9px "Source Sans 3", sans-serif`
    ctx.textAlign = 'center'
    ctx.fillStyle = color + (isPrimary ? '' : 'aa')
    ctx.fillText(label, pt.x, pt.y - 6)
  }
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => { if (ref.current) drawSparkline(ref.current, data, color) }, [])
  return <canvas ref={ref} style={{ width: '100%', height: '50px', display: 'block' }} />
}

export default function CountrySparklines() {
  return (
    <div style={{ margin: '1.5em 0', fontFamily: "'Source Sans 3', sans-serif" }}>
      {/* Timeline header */}
      <div style={{ display: 'flex', alignItems: 'flex-end', marginBottom: '2px' }}>
        <div style={{ width: '86px', flexShrink: 0 }} />
        <div style={{ flex: 1, position: 'relative', height: '16px' }}>
          {[0, 3, 6, 9, 13].map(i => (
            <span key={i} style={{
              position: 'absolute',
              left: `${(i / 13) * 100}%`,
              transform: 'translateX(-50%)',
              fontSize: '10px',
              color: '#94a3b8',
            }}>{DECADES[i].slice(0, 4)}</span>
          ))}
        </div>
        <div style={{ width: '36px', flexShrink: 0 }} />
      </div>

      {COUNTRIES.map(({ name, data, color }) => {
        const peaks = findPeaks(data, 2)
        return (
          <div key={name} style={{
            display: 'flex',
            alignItems: 'center',
            borderTop: '1px solid #f1f5f9',
          }}>
            <div style={{
              width: '86px', flexShrink: 0,
              fontSize: '12px', fontWeight: 600, color,
              paddingRight: '6px',
            }}>{name}</div>
            <div style={{ flex: 1 }}>
              <Sparkline data={data} color={color} />
            </div>
            <div style={{
              width: '36px', flexShrink: 0,
              textAlign: 'right', fontSize: '11px', color: '#94a3b8',
            }}>{Math.round(data[peaks[0]])}%</div>
          </div>
        )
      })}

      <div style={{ borderTop: '1px solid #f1f5f9', fontSize: '10px', color: '#94a3b8', marginTop: '0', paddingTop: '6px', textAlign: 'right' }}>
        % share of all foreign country mentions per decade. Top two peaks labelled. 27K speeches sampled.
      </div>
    </div>
  )
}
