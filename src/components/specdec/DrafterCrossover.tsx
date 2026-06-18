import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Where each drafter wins, as a field over (draft depth gamma, decode batch B).
// Colour is the cost ratio t_MTP / t_DFlash: orange where DFlash is cheaper,
// blue where MTP is cheaper, with the equal-cost contour drawn through ratio = 1.
// Qwen3.6-35B-A3B / B200 bf16. Both heads borrow the target's 0.509 B vocab
// projection; MTP re-reads it on each of its gamma serial passes (intensity B),
// DFlash reads it once for the whole gamma-block (intensity B*gamma). So DFlash
// goes compute-bound far sooner and the orange region is the low-batch interior;
// MTP's smaller FLOP bill wins above the contour (B ~ 161) and on the gamma = 1
// edge, where an eight-layer block head never pays off to make a single token.

const PEAK = 2.25e15 // B200 bf16, FLOP/s
const BW = 8.0e12 // HBM bandwidth, B/s
const BPP = 2 // bytes per bf16 param

// Params each head streams per pass (from the HF configs). Shared head dominates.
const P_HEAD = 508_559_360 // tied target vocab projection, d*V = 2048*248320
const P_MTP = P_HEAD + 55_574_528 // + one EAGLE-style MoE layer  -> 0.564 B
const P_DF = P_HEAD + 473_956_352 // + eight dense layers + 5-layer fusion -> 0.983 B

const tMtp = (g: number, B: number) =>
  g * Math.max((2 * P_MTP * B) / PEAK, (P_MTP * BPP) / BW)
const tDflash = (g: number, B: number) =>
  Math.max((2 * P_DF * B * g) / PEAK, (P_DF * BPP) / BW)
const logRatio = (g: number, B: number) => Math.log10(tMtp(g, B) / tDflash(g, B))

const GAMMAS = Array.from({ length: 16 }, (_, i) => i + 1)
const BMIN = 1
const BMAX = 256
const NROWS = 96
const bEdge = (i: number) => BMIN + (i / NROWS) * (BMAX - BMIN) // linear cells

// Symmetric diverging fill over the page background, in log-ratio v = log10(r).
// Orange = DFlash cheaper (v > 0), blue = MTP cheaper (v < 0); intensity is |v|
// on ONE shared scale (VMAX = 1 dex, i.e. 10x either way) so equal magnitudes
// read as equal saturation. The field is genuinely lopsided -- DFlash wins by up
// to ~9x, MTP by at most ~1.7x -- so the plot is deep orange against pale blue,
// which is the honest picture. Composites the same on the canvas or as a CSS
// gradient over the same background.
const VMAX = 1.0 // dex; colour saturates at a 10x ratio either way
function fillColor(v: number, isDark: boolean): string {
  // Perceptual boost: alpha rises as |v|^0.55 so the mid-range ratios read as
  // colour instead of washing out, while the full 0..10x range stays distinct.
  const m = Math.min(1, Math.abs(v) / VMAX)
  const a = (0.12 + 0.82 * Math.pow(m, 0.55)).toFixed(3)
  return v >= 0
    ? `hsla(28, 85%, ${isDark ? 58 : 48}%, ${a})`
    : `hsla(220, 72%, ${isDark ? 62 : 44}%, ${a})`
}

// Batch where the two costs are equal, per gamma column (null if MTP wins for
// all B, which happens at gamma = 1: a block head is never worth one token).
function crossoverB(g: number): number | null {
  if (logRatio(g, BMIN) <= 0) return null
  let lo = BMIN
  let hi = BMAX
  for (let k = 0; k < 40; k++) {
    const mid = 0.5 * (lo + hi)
    if (logRatio(g, mid) > 0) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}

const TOOLTIP_B = [1, 4, 16, 48, 96, 128, 161, 200, 256]

// Colourbar: bar fraction f maps linearly to log-ratio, f=0 -> -VMAX (deep blue,
// MTP 10x), f=0.5 -> 0 (equal), f=1 -> +VMAX (deep orange, DFlash 10x). Ticks are
// evenly spaced so they never collide; direction words live in a header instead.
const barV = (f: number) => (f - 0.5) * 2 * VMAX
const BAR_TICKS = [
  { f: 0, label: '10×' },
  { f: 0.25, label: '3×' },
  { f: 0.5, label: '1×' },
  { f: 0.75, label: '3×' },
  { f: 1, label: '10×' },
]

export default function DrafterCrossover() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const barRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }

    const heatmap = {
      id: 'heatmap',
      beforeDatasetsDraw(chart: Chart) {
        const { ctx, chartArea, scales } = chart
        const xs = scales.x
        const ys = scales.y
        ctx.save()
        ctx.beginPath()
        ctx.rect(chartArea.left, chartArea.top, chartArea.width, chartArea.height)
        ctx.clip()
        for (const g of GAMMAS) {
          const xl = xs.getPixelForValue(g - 0.5)
          const xr = xs.getPixelForValue(g + 0.5)
          for (let i = 0; i < NROWS; i++) {
            const blo = bEdge(i)
            const bhi = bEdge(i + 1)
            const ytop = ys.getPixelForValue(bhi)
            const ybot = ys.getPixelForValue(blo)
            ctx.fillStyle = fillColor(logRatio(g, 0.5 * (blo + bhi)), theme.isDark)
            ctx.fillRect(xl, ytop, xr - xl + 0.6, ybot - ytop + 0.6)
          }
        }
        ctx.restore()
      },
    }

    const contour = {
      id: 'contour',
      afterDatasetsDraw(chart: Chart) {
        const { ctx, scales } = chart
        const xs = scales.x
        const ys = scales.y
        const pts: { x: number; y: number }[] = []
        for (const g of GAMMAS) {
          const b = crossoverB(g)
          if (b != null) pts.push({ x: xs.getPixelForValue(g), y: ys.getPixelForValue(b) })
        }
        ctx.save()
        if (pts.length > 1) {
          ctx.strokeStyle = theme.foreground
          ctx.setLineDash([5, 4])
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(pts[0].x, pts[0].y)
          pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y))
          ctx.stroke()
          ctx.setLineDash([])
          const last = pts[pts.length - 1]
          ctx.fillStyle = theme.foreground
          ctx.font = `600 12px ${theme.fontFamily}`
          ctx.textAlign = 'right'
          ctx.fillText('equal cost  (B ≈ 161)', last.x - 4, last.y - 7)
        }
        ctx.font = `600 13px ${theme.fontFamily}`
        ctx.textAlign = 'center'
        ctx.fillStyle = theme.mutedForeground
        ctx.fillText('MTP cheaper', xs.getPixelForValue(9.5), ys.getPixelForValue(225))
        ctx.fillText('DFlash cheaper', xs.getPixelForValue(9.5), ys.getPixelForValue(70))
        ctx.restore()
      },
    }

    chartInstance.current = new Chart(chartRef.current, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '',
            data: GAMMAS.flatMap((g) => TOOLTIP_B.map((B) => ({ x: g, y: B }))),
            pointRadius: 0,
            pointHoverRadius: 0,
            pointHitRadius: 14,
            showLine: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const p = items[0]?.parsed
                return p ? `γ = ${p.x},  B = ${p.y} tok` : ''
              },
              label: (ctx) => {
                const g = ctx.parsed.x
                const B = ctx.parsed.y
                const r = tMtp(g, B) / tDflash(g, B)
                return r > 1
                  ? `  DFlash ${r.toFixed(1)}× cheaper`
                  : `  MTP ${(1 / r).toFixed(1)}× cheaper`
              },
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0.5,
            max: 16.5,
            title: axisTitle('draft depth γ (tokens per round)'),
            ticks: { ...tickStyle, stepSize: 1, autoSkip: false },
            grid: { display: false },
          },
          y: {
            type: 'linear',
            min: BMIN,
            max: BMAX,
            title: axisTitle('decode batch B (tokens)'),
            grid: { display: false },
            afterBuildTicks: (axis) => {
              axis.ticks = [1, 64, 128, 192, 256].map((v) => ({ value: v }))
            },
            ticks: { ...tickStyle, callback: (v) => `${v}` },
          },
        },
      },
      plugins: [heatmap, contour],
    })

    // Colourbar drawn as fillColor strips on its own canvas: matches the plot
    // exactly and avoids the muddy hue/alpha interpolation a CSS gradient gives
    // between semi-transparent blue and orange stops.
    const bar = barRef.current
    if (bar) {
      const w = bar.clientWidth || 360
      const h = 14
      const dpr = window.devicePixelRatio || 1
      bar.width = Math.round(w * dpr)
      bar.height = Math.round(h * dpr)
      const bctx = bar.getContext('2d')
      if (bctx) {
        bctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        bctx.clearRect(0, 0, w, h)
        const N = Math.max(160, Math.round(w))
        for (let i = 0; i < N; i++) {
          bctx.fillStyle = fillColor(barV(i / (N - 1)), theme.isDark)
          bctx.fillRect((i * w) / N, 0, w / N + 1, h)
        }
      }
    }

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  return (
    <div class="my-6">
      <div style="position: relative; height: 400px; width: 100%;">
        <canvas ref={chartRef} />
      </div>
      <div style="max-width: 380px; margin: 0.75rem auto 0;">
        <div style={`display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 3px; color: ${theme.mutedForeground};`}>
          <span>← MTP cheaper</span>
          <span>DFlash cheaper →</span>
        </div>
        <canvas
          ref={barRef}
          style={`display: block; width: 100%; height: 14px; border-radius: 3px; border: 1px solid ${theme.grid};`}
        />
        <div style="position: relative; height: 1.05rem; margin-top: 3px;">
          {BAR_TICKS.map((t) => (
            <span
              style={`position: absolute; left: ${t.f * 100}%; transform: translateX(${
                t.f === 0 ? '0' : t.f === 1 ? '-100%' : '-50%'
              }); font-size: 0.72rem; font-variant-numeric: tabular-nums; color: ${theme.mutedForeground}; white-space: nowrap;`}
            >
              {t.label}
            </span>
          ))}
        </div>
        <div style={`font-size: 0.72rem; text-align: center; margin-top: 2px; color: ${theme.mutedForeground};`}>
          cost ratio t<sub>MTP</sub> / t<sub>DFlash</sub>
        </div>
      </div>
    </div>
  )
}
