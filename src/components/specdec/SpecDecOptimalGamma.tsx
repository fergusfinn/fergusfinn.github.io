import { useEffect, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Optimal draft length vs batch. The standard speedup S(g) = (1 - a^{g+1}) /
// ((1-a)(1 + c*g)) treats the verify cost as a linear c*g. That undercounts it:
// a draft of g moves the dense GEMMs up the roofline by B*(g+1) tokens, not g,
// and pushes attention's compute branch to q = g+1. So instead of a linearised
// slope we price the verify exactly through the per-component cost model at its
// real operating point, and take g* = argmax of the resulting throughput ratio.
// DeepSeek-V4-Flash on B200: attn projections fp8, attention matmul bf16 (FlashMLA),
// MoE experts MXFP4, KV fp8.

const R_FP8 = 563 // fp8 ridge, FLOP/byte (4.5 PFLOP/s / 8 TB/s)
const R_FP4 = 1125 // MXFP4 ridge, FLOP/byte (9 PFLOP/s / 8 TB/s)
const R_BF16 = 281 // bf16 ridge, FLOP/byte (2.25 PFLOP/s / 8 TB/s) -- FlashMLA attention matmul
const B_FP8 = 1 // bytes/elem, fp8 weights
const B_FP4 = 0.53 // bytes/elem, MXFP4 (4-bit + 8-bit scale / 32-block)
const B_KV = 1 // bytes/elem, fp8 KV cache
const E = 256
const KACT = 6
const SHARED = 1
const W_ATTN = 107e6 // per-layer MLA QKVO projection params (V4-Flash), fp8
const W_E = 25.2e6 // params per expert (3·4096·2048), MXFP4
const KAPPA = 576
const PHI = 69632 // absorbed-MLA attention MACs/pair: n_h heads dot the 512 latent (score + AV) + rope = 64*1088 (V4-Flash, n_h=64)

function eAct(t: number): number {
  return E * (1 - Math.pow(1 - KACT / E, t))
}
function cAttnProj(t: number): number {
  return Math.max((2 * W_ATTN * t) / R_FP8, W_ATTN * B_FP8)
}
function cMoE(t: number): number {
  return Math.max((2 * (KACT + SHARED) * W_E * t) / R_FP4, (eAct(t) + SHARED) * W_E * B_FP4)
}
function cAttn(B: number, q: number, S: number): number {
  // fp8 KV read (memory), bf16 matmul (compute) -- the FlashMLA split.
  return B * Math.max((2 * PHI * S * q) / R_BF16, KAPPA * S * B_KV)
}

// Per-step cost of verifying q query tokens/sequence across batch B at context
// S. The dense GEMMs process all B*q query tokens, so speculation moves them up
// the roofline by the factor q = γ+1 (the same token axis as batch, which is
// why a draft of γ costs like B*(γ+1) tokens, not γ). Attention reads each
// sequence's KV once (∝ S, independent of q) and computes scores ∝ q*S, so only
// its compute branch grows with the draft length.
function stepCost(B: number, q: number, S: number): number {
  return cAttnProj(B * q) + cMoE(B * q) + cAttn(B, q, S)
}

// g* = argmax_g [ N(α,g) * C(B, q=1) / C(B, q=g+1) ], the true per-step
// throughput ratio against no-speculation (N(α,g) = (1-α^{g+1})/(1-α) committed
// tokens). Pricing the verify at its actual operating point q = g+1 (not a
// linearised slope at q=1) is what surfaces the dense knees at B*(g+1) and the
// MLA attention compute wall at q ≈ κR/2Φ. g starts at 0 (no speculation,
// ratio exactly 1), so g* drops to 0 and the speedup floors at 1x wherever even
// one draft token loses.
function bestGamma(
  alpha: number,
  B: number,
  S: number,
  cDraft: number,
): { gamma: number; speedup: number } {
  const c0 = stepCost(B, 1, S)
  let gamma = 0
  let speedup = 1
  for (let g = 1; g <= 64; g++) {
    const committed = (1 - Math.pow(alpha, g + 1)) / (1 - alpha)
    // Exact verify cost (nonlinear in g) plus the drafter: g sequential draft
    // passes, each ~cDraft of one target decode step. Both relative to c0, so
    // the linear drafter term is kept separate from the roofline verify cost.
    const cost = stepCost(B, g + 1, S) / c0 + cDraft * g
    const s = committed / cost
    if (s > speedup) {
      speedup = s
      gamma = g
    }
  }
  return { gamma, speedup }
}

interface Point {
  x: number
  y: number
}

export default function SpecDecOptimalGamma() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()
  const [seqPos, setSeqPos] = useState(37.5) // -> 4096 tok
  const [alphaPct, setAlphaPct] = useState(75)
  const [cDraftPct, setCDraftPct] = useState(10) // drafter cost, % of a target step
  const seqLen = Math.round(512 * Math.pow(2, (seqPos / 100) * 8))
  const alpha = alphaPct / 100
  const cDraft = cDraftPct / 100

  useEffect(() => {
    if (!chartRef.current) return
    applyChartDefaults(theme)

    const gammaPts: Point[] = []
    let gMax = 2
    for (let i = 0; i <= 300; i++) {
      const B = Math.pow(10, (i / 300) * Math.log10(30000))
      const { gamma } = bestGamma(alpha, B, seqLen, cDraft)
      gammaPts.push({ x: B, y: gamma })
      if (gamma > gMax) gMax = gamma
    }

    const gammaColor = theme.foreground

    chartInstance.current?.destroy()
    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'optimal γ',
            data: gammaPts,
            yAxisID: 'yGamma',
            borderColor: gammaColor,
            backgroundColor: gammaColor,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 3,
            stepped: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: {
            position: 'top',
            align: 'end',
            labels: {
              boxWidth: 14,
              boxHeight: 2,
              color: theme.foreground,
              font: { family: theme.fontFamily },
            },
          },
          tooltip: {
            callbacks: {
              title: (items) =>
                items.length > 0 ? `batch ≈ ${Math.round(items[0].parsed.x ?? 0)} seqs` : '',
              label: (ctx) => {
                const v = ctx.parsed.y ?? 0
                return Math.round(v) === 0 ? '  speculation off' : `  draft ${Math.round(v)} tokens`
              },
            },
          },
        },
        scales: {
          x: {
            type: 'logarithmic',
            min: 1,
            max: 30000,
            title: {
              display: true,
              text: 'batch size B (sequences, log scale)',
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
          yGamma: {
            type: 'linear',
            position: 'left',
            min: 0,
            max: gMax + 2,
            title: {
              display: true,
              text: 'optimal draft length γ',
              color: theme.foreground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.foreground,
              font: { family: theme.fontFamily },
              callback: (v) => `${Number(v).toFixed(0)}`,
            },
            grid: { color: theme.grid },
          },
        },
      },
    })

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  // Update data in place when the sliders move, so the line morphs instead of
  // replaying its mount animation from zero on every change.
  useEffect(() => {
    const chart = chartInstance.current
    if (!chart) return
    const gammaPts: Point[] = []
    let gMax = 2
    for (let i = 0; i <= 300; i++) {
      const B = Math.pow(10, (i / 300) * Math.log10(30000))
      const { gamma } = bestGamma(alpha, B, seqLen, cDraft)
      gammaPts.push({ x: B, y: gamma })
      if (gamma > gMax) gMax = gamma
    }
    chart.data.datasets[0].data = gammaPts
    const yGamma = chart.options.scales?.yGamma as { max?: number } | undefined
    if (yGamma) yGamma.max = gMax + 2
    chart.update('none')
  }, [seqLen, alpha, cDraft])

  const row = 'display: flex; align-items: center; gap: 0.6rem; font-size: 0.85rem;'
  return (
    <div class="my-6">
      <div style="display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 0.6rem; align-items: center;">
        <div style={row}>
          <label style="white-space: nowrap;">avg seq len</label>
          <input
            type="range"
            min={0}
            max={100}
            value={seqPos}
            onInput={(e) => setSeqPos(Number((e.target as HTMLInputElement).value))}
            style="width: 150px;"
          />
          <span style="font-variant-numeric: tabular-nums; min-width: 4.5rem;">
            {seqLen.toLocaleString()} tok
          </span>
        </div>
        <div style={row}>
          <label style="white-space: nowrap;">acceptance α</label>
          <input
            type="range"
            min={1}
            max={99}
            value={alphaPct}
            onInput={(e) => setAlphaPct(Number((e.target as HTMLInputElement).value))}
            style="width: 150px;"
          />
          <span style="font-variant-numeric: tabular-nums; min-width: 4.5rem;">{alphaPct}%</span>
        </div>
        <div style={row}>
          <label style="white-space: nowrap;">drafter cost</label>
          <input
            type="range"
            min={0}
            max={50}
            value={cDraftPct}
            onInput={(e) => setCDraftPct(Number((e.target as HTMLInputElement).value))}
            style="width: 150px;"
          />
          <span style="font-variant-numeric: tabular-nums; min-width: 4.5rem;">{cDraftPct}%</span>
        </div>
      </div>
      <div style="position: relative; height: 340px; width: 100%;">
        <canvas ref={chartRef} />
      </div>
    </div>
  )
}
