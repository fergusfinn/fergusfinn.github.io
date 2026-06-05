import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// MHA vs MLA attention roofline at decode, same model (DeepSeek-V4 dims), both
// doing the matmul in bf16 (FlashMLA) so they share the bf16 ridge. One decode
// step's attention slides right along the arithmetic-intensity axis as the
// number of query tokens T (= gamma+1 in a verify) grows: compute ∝ T·S, but the
// KV read ∝ S is paid once, so AI ∝ T. MHA stores full per-head K,V (large
// bytes/token) so its intensity is low and it stays memory-bound for hundreds of
// tokens. MLA stores one compressed latent (tiny bytes/token) but runs the
// attention in that 512-wide latent across all heads (large FLOPs/pair), so its
// intensity is ~240x higher and it sits at the compute ceiling from T=1.

const R = 281 // bf16 ridge, FLOP/byte (B200: 2.25 PFLOP/s / 8 TB/s)
const H = 128 // attention heads
const D_H = 128 // per-head dim (MHA)
const D_C = 512 // MLA latent dim
const ROPE = 64
const B_KV = 1 // fp8 KV storage, bytes/elem

// f = FLOP per query-key pair (2x MACs); mc = bytes per context token in cache.
const F_MHA = 2 * (2 * H * D_H) // score + AV over the head dim, all heads
const MC_MHA = 2 * H * D_H * B_KV // K + V, full per-head, fp8 store
const F_MLA = 2 * H * (D_C + ROPE + D_C) // score (latent+rope) + AV (latent), all heads
const MC_MLA = (D_C + ROPE) * B_KV // single shared latent + rope, fp8 store

const ai = (f: number, mc: number, T: number) => (f * T) / mc
const perf = (a: number) => Math.min(1, a / R) // attainable fraction of peak

const T_MHA = [1, 2, 4, 8, 16, 32, 64, 128, 256]
const T_MLA = [1, 2, 4, 8, 16]

interface Pt {
  x: number
  y: number
  t: number
}

export default function SpecDecAttnRoofline() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    applyChartDefaults(theme)

    const xMin = 1
    const xMax = 1e4

    // Roofline backbone: P(AI) = min(1, AI/R). Bracket the ridge so the corner
    // renders sharp rather than as a slanted secant.
    const backbone: { x: number; y: number }[] = []
    for (let i = 0; i <= 200; i++) {
      const a = xMin * Math.pow(xMax / xMin, i / 200)
      backbone.push({ x: a, y: perf(a) })
    }
    backbone.push({ x: R * (1 - 1e-6), y: perf(R * (1 - 1e-6)) })
    backbone.push({ x: R, y: 1 })
    backbone.sort((p, q) => p.x - q.x)

    const mha: Pt[] = T_MHA.map((t) => {
      const a = ai(F_MHA, MC_MHA, t)
      return { x: a, y: perf(a), t }
    })
    const mla: Pt[] = T_MLA.map((t) => {
      const a = ai(F_MLA, MC_MLA, t)
      return { x: a, y: perf(a), t }
    })

    const mhaColor = `hsl(220, 75%, ${theme.isDark ? 64 : 48}%)`
    const mlaColor = `hsl(28, 85%, ${theme.isDark ? 62 : 48}%)`

    chartInstance.current?.destroy()
    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'roofline',
            data: backbone,
            borderColor: theme.mutedForeground,
            backgroundColor: theme.mutedForeground,
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0,
            order: 3,
          },
          {
            label: 'MHA',
            data: mha,
            borderColor: mhaColor,
            backgroundColor: mhaColor,
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0,
            order: 1,
          },
          {
            label: 'MLA',
            data: mla,
            borderColor: mlaColor,
            backgroundColor: mlaColor,
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0,
            order: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: true },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              boxWidth: 14,
              boxHeight: 2,
              color: theme.foreground,
              font: { family: theme.fontFamily },
              filter: (item) => item.text !== 'roofline',
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => {
                const raw = items[0]?.raw as Pt | undefined
                return raw?.t ? `T = ${raw.t} query token${raw.t > 1 ? 's' : ''}` : ''
              },
              label: (ctx) => {
                const raw = ctx.raw as Pt
                const at = `AI ${Math.round(raw.x)} FLOP/byte`
                const util = `${Math.round(raw.y * 100)}% of peak`
                return `  ${ctx.dataset.label}: ${at}, ${util}`
              },
            },
          },
        },
        scales: {
          x: {
            type: 'logarithmic',
            min: xMin,
            max: xMax,
            title: {
              display: true,
              text: 'arithmetic intensity (FLOP / byte, log)',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
              callback: (v) => {
                const n = Number(v)
                if ([1, 10, 100, 1000, 10000].includes(n)) return n.toString()
                return ''
              },
            },
            grid: { color: theme.grid },
          },
          y: {
            type: 'logarithmic',
            min: 0.005,
            max: 1.2,
            title: {
              display: true,
              text: 'attainable compute (fraction of peak, log)',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
              callback: (v) => {
                const n = Number(v)
                if ([0.01, 0.1, 1].includes(n)) return `${n}`
                return ''
              },
            },
            grid: { color: theme.grid },
          },
        },
      },
      plugins: [
        {
          id: 'ridge',
          afterDatasetsDraw(chart) {
            const { ctx, chartArea, scales } = chart
            const x = scales.x.getPixelForValue(R)
            if (x < chartArea.left || x > chartArea.right) return
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.setLineDash([4, 4])
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(x, chartArea.top)
            ctx.lineTo(x, chartArea.bottom)
            ctx.stroke()
            ctx.fillStyle = theme.mutedForeground
            ctx.font = `11px ${theme.fontFamily}`
            ctx.textAlign = 'left'
            ctx.textBaseline = 'top'
            ctx.fillText('ridge (compute-bound →)', x + 4, chartArea.top + 2)
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
    <div class="my-6">
      <p style="font-size: 0.8rem; opacity: 0.7; margin: 0 0 0.4rem;">
        One decode attention step for the same model run two ways, as T = 1, 2, 4,
        … query tokens ride along. MHA stores full per-head K,V and crawls up the
        memory-bound slope, only reaching the compute ridge near T ≈ 140. MLA
        stores one compressed latent but computes in it across every head, so it is
        already at the ridge at T = 1: speculated tokens are free for attention
        under MHA, and not under MLA.
      </p>
      <div style="position: relative; height: 360px; width: 100%;">
        <canvas ref={chartRef} />
      </div>
    </div>
  )
}
