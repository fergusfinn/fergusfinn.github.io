import { useEffect, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Per-component marginal cost of a speculated token vs batch size, on a roofline
// cost model. DeepSeek-V4-Flash on a B200: attention projections fp8, attention
// matmul bf16 (FlashMLA), MoE experts MXFP4, KV cache fp8. Each component is
// priced at its own precision's ridge.
//
// Cost of a pass over `tokens` tokens for a component is C = max(compute, memory):
//   attn-proj GEMM: memory = weights (const), compute ∝ tokens. Knee at fp8 ridge.
//   MoE GEMM:       memory = distinct experts loaded (coupon-collector),
//                   compute ∝ tokens. Knee ~ (E/k) further out.
//   attention op:   memory = fp8 KV read (∝ S, once per step), compute ∝ S·queries
//                   in bf16. MLA's compressed KV is so cheap to read that q=1 sits
//                   just under the ridge: the first speculated token (q=2) tips it
//                   compute-bound, so the marginal verify token is priced there.
//
// In steady-state batched decoding a real token costs the AVERAGE C(B)/B; a
// speculated token squeezed into the same step costs the MARGINAL C'(B). Their
// ratio is the local roofline slope (the cost elasticity):
//   r = C'(B) / (C(B)/B) = d(log C) / d(log tokens)               in [0, 1]
//   r -> 0 : memory bound, the extra token rides a load already paid for (free)
//   r -> 1 : compute bound, it costs real FLOPs (full price)
// No draft length or acceptance rate enters. r is a pure per-token cost, and it is
// the hurdle a token's accept-prob must clear: profit on accept 1-r, loss on
// reject r, break even at accept-prob = r.

// B200 per-precision ridges (datasheet dense FLOP/s ÷ 8 TB/s) and storage bytes.
const R_FP8 = 563 // fp8 ridge, FLOP/byte (4.5 PFLOP/s)
const R_FP4 = 1125 // MXFP4 ridge, FLOP/byte (9 PFLOP/s)
const R_BF16 = 281 // bf16 ridge, FLOP/byte (2.25 PFLOP/s) -- FlashMLA runs the attention matmul in bf16
const B_FP8 = 1 // bytes/elem, fp8 weights
const B_FP4 = 0.53 // bytes/elem, MXFP4 (4-bit + 8-bit scale / 32-block)
const B_KV = 1 // bytes/elem, fp8 KV cache

const E = 256 // routed experts
const KACT = 6 // active routed experts / token
const SHARED = 1 // always-on shared expert

// Relative element counts (per layer; layer count cancels in the ratios).
const W_ATTN = 107e6 // per-layer MLA QKVO projection params (V4-Flash), fp8
const W_E = 25.2e6 // params per expert (3·4096·2048), MXFP4
const KAPPA = 576 // KV-cache elements / context token (MLA latent + rope), fp8
const PHI = 69632 // absorbed-MLA attention MACs/pair: n_h heads each dot the 512 latent (score + AV) + rope = 64*1088 (V4-Flash, n_h=64)

function eAct(tokens: number): number {
  // Exact distinct-top-k coupon: each token routes to k distinct experts.
  return E * (1 - Math.pow(1 - KACT / E, tokens))
}
function eActPrime(tokens: number): number {
  // d(eAct)/d(tokens): how many fresh experts the next token drags in.
  const rho = 1 - KACT / E
  return -E * Math.log(rho) * Math.pow(rho, tokens)
}

// Component cost in bytes-equivalent: max(2·MACs / R_prec, elements · bytes).
// tokens = B * queries-per-sequence.
function cAttnProj(tokens: number): number {
  return Math.max((2 * W_ATTN * tokens) / R_FP8, W_ATTN * B_FP8)
}
function cMoE(tokens: number): number {
  const compute = (2 * (KACT + SHARED) * W_E * tokens) / R_FP4
  const memory = (eAct(tokens) + SHARED) * W_E * B_FP4
  return Math.max(compute, memory)
}
// Attention cost: per sequence, KV read once per step, compute ∝ S·queries.
function cAttn(B: number, q: number, S: number): number {
  // fp8 KV read (memory), bf16 matmul (compute) -- the FlashMLA split.
  return B * Math.max((2 * PHI * S * q) / R_BF16, KAPPA * S * B_KV)
}

interface Point {
  x: number
  y: number
}
interface Curves {
  attnproj: Point[]
  moe: Point[]
  attn: Point[]
  drafter: Point[]
  total: Point[]
}

const X_MIN = 1
const X_MAX = 30000
// Batch grid: a log sweep plus the two roofline knees where the slope jumps 0->1,
// each bracketed so the step renders vertical instead of as a slanted secant.
const ATTNPROJ_KNEE = R_FP8 / 2 // fp8 GEMM tips compute-bound (~281 tokens)
const MOE_KNEE = ((E + SHARED) * B_FP4 * R_FP4) / (2 * (KACT + SHARED)) // ~10,950 tokens
function batchGrid(): number[] {
  const xs: number[] = []
  const N = 280
  const logMax = Math.log10(X_MAX)
  for (let i = 0; i <= N; i++) xs.push(Math.pow(10, (i / N) * logMax))
  for (const knee of [ATTNPROJ_KNEE, MOE_KNEE]) xs.push(knee * (1 - 1e-6), knee * (1 + 1e-6))
  return xs.filter((x) => x >= X_MIN && x <= X_MAX).sort((a, b) => a - b)
}

function computeCurves(S: number, cDraft: number): Curves {
  const cost: Curves = { attnproj: [], moe: [], attn: [], drafter: [], total: [] }
  for (const B of batchGrid()) {
    // r = marginal token cost / average token cost = local roofline slope in [0,1].
    // Analytic: on the compute branch r = 1; on the memory branch r = t·M'(t)/M(t),
    // which is 0 for a constant weight load and the expert-growth elasticity for MoE.
    const cAP = cAttnProj(B)
    const cMo = cMoE(B)
    const cAt = cAttn(B, 1, S)
    const rAP = (2 * W_ATTN * B) / R_FP8 >= W_ATTN * B_FP8 ? 1 : 0
    const moCompute = (2 * (KACT + SHARED) * W_E * B) / R_FP4
    const rMo =
      moCompute >= (eAct(B) + SHARED) * W_E * B_FP4 ? 1 : (B * eActPrime(B)) / (eAct(B) + SHARED)
    // A speculated token widens the verify to q=2; for MLA q=1 sits just under the
    // ridge, so the first speculated token tips compute-bound. Price the hurdle there.
    const rAt = (2 * PHI * S * 2) / R_BF16 >= KAPPA * S * B_KV ? 1 : 0
    cost.attnproj.push({ x: B, y: rAP })
    cost.moe.push({ x: B, y: rMo })
    cost.attn.push({ x: B, y: rAt })
    cost.drafter.push({ x: B, y: cDraft })
    // total hurdle: cost-share-weighted blend of the verify slopes, plus the
    // flat drafter cost paid on every draft token (accepted or rejected).
    const verify = (rAP * cAP + rMo * cMo + rAt * cAt) / (cAP + cMo + cAt)
    cost.total.push({ x: B, y: verify + cDraft })
  }
  return cost
}

function datasets(c: Curves, theme: ReturnType<typeof useChartTheme>, showComponents: boolean) {
  const mk = (label: string, data: Point[], hue: number, dash = false) => ({
    label,
    data,
    borderColor: `hsl(${hue}, 75%, ${theme.isDark ? 62 : 46}%)`,
    backgroundColor: `hsl(${hue}, 75%, ${theme.isDark ? 62 : 46}%)`,
    pointRadius: 0,
    pointHoverRadius: 4,
    tension: 0.1,
    borderWidth: 2,
    borderDash: dash ? [6, 4] : undefined,
  })
  const totalDs = {
    ...mk('total', c.total, theme.isDark ? 0 : 0),
    borderColor: theme.foreground,
    backgroundColor: theme.foreground,
    borderWidth: 3,
  }
  return showComponents
    ? [
        mk('attn-proj GEMMs', c.attnproj, 220),
        mk('MoE GEMMs', c.moe, 28),
        mk('attention', c.attn, 160),
        mk('drafter', c.drafter, 280, true),
        totalDs,
      ]
    : [totalDs]
}

function baseOptions(
  theme: ReturnType<typeof useChartTheme>,
  yTitle: string,
  yMin: number,
  yMax: number,
  showComponents: boolean
) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'nearest' as const, axis: 'x' as const, intersect: false },
    plugins: {
      legend: {
        display: showComponents,
        position: 'top' as const,
        align: 'end' as const,
        labels: {
          boxWidth: 14,
          boxHeight: 2,
          color: theme.foreground,
          font: { family: theme.fontFamily },
        },
      },
      tooltip: {
        callbacks: {
          title: (items: { parsed: { x: number | null } }[]) =>
            items.length > 0 ? `batch ≈ ${Math.round(items[0].parsed.x ?? 0)} seqs` : '',
          label: (ctx: { dataset: { label?: string }; parsed: { y: number | null } }) => {
            const m = ctx.parsed.y ?? 0
            return `  ${ctx.dataset.label}: costs ${m.toFixed(2)}× (keep → +${(1 - m).toFixed(2)}×)`
          },
        },
      },
    },
    scales: {
      x: {
        type: 'logarithmic' as const,
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
          callback: (v: string | number) => {
            const n = Number(v)
            if ([1, 10, 100, 1000, 10000].includes(n)) return n.toString()
            return ''
          },
        },
        grid: { color: theme.grid },
      },
      y: {
        type: 'linear' as const,
        min: yMin,
        max: yMax,
        title: {
          display: true,
          text: yTitle,
          color: theme.mutedForeground,
          font: { family: theme.fontFamily },
        },
        ticks: {
          color: theme.mutedForeground,
          font: { family: theme.fontFamily },
          callback: (v: string | number) => `${Number(v).toFixed(1)}×`,
        },
        grid: { color: theme.grid },
      },
    },
  }
}

function refLinePlugin(theme: ReturnType<typeof useChartTheme>, yVal: number, label: string) {
  return {
    id: 'refLine',
    afterDatasetsDraw(chart: Chart) {
      const { ctx, chartArea, scales } = chart
      const y = scales.y.getPixelForValue(yVal)
      if (y < chartArea.top || y > chartArea.bottom) return
      ctx.save()
      ctx.strokeStyle = theme.mutedForeground
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(chartArea.left, y)
      ctx.lineTo(chartArea.right, y)
      ctx.stroke()
      ctx.fillStyle = theme.mutedForeground
      ctx.font = `11px ${theme.fontFamily}`
      ctx.textAlign = 'left'
      ctx.textBaseline = 'bottom'
      ctx.fillText(label, chartArea.left + 6, y - 2)
      ctx.restore()
    },
  }
}

export default function SpecDecLedger() {
  const costRef = useRef<HTMLCanvasElement>(null)
  const costChart = useRef<Chart | null>(null)
  const theme = useChartTheme()
  const [sliderPos, setSliderPos] = useState(63) // ~8192
  const [cDraftPct, setCDraftPct] = useState(10) // drafter cost, % of a real token
  const [showComponents, setShowComponents] = useState(false)
  const seqLen = Math.round(512 * Math.pow(2, (sliderPos / 100) * 8))
  const cDraft = cDraftPct / 100

  useEffect(() => {
    if (!costRef.current) return
    applyChartDefaults(theme)
    const cost = computeCurves(seqLen, cDraft)

    costChart.current?.destroy()
    costChart.current = new Chart(costRef.current, {
      type: 'line',
      data: { datasets: datasets(cost, theme, showComponents) },
      options: baseOptions(theme, 'speculated-token cost (× a real token)', 0, 1.3, showComponents),
      plugins: [refLinePlugin(theme, 1, 'full price — pure waste if rejected')],
    })

    return () => {
      costChart.current?.destroy()
      costChart.current = null
    }
  }, [theme, seqLen, cDraft, cDraftPct, showComponents])

  const sliderRow = 'display: flex; align-items: center; gap: 0.6rem; font-size: 0.85rem;'
  return (
    <div class="my-6">
      <div style="display: flex; flex-wrap: wrap; gap: 1.25rem; margin-bottom: 0.6rem; align-items: center;">
        <div style={sliderRow}>
          <label style="white-space: nowrap;">avg seq len</label>
          <input
            type="range"
            min={0}
            max={100}
            value={sliderPos}
            onInput={(e) => setSliderPos(Number((e.target as HTMLInputElement).value))}
            style="width: 160px;"
          />
          <span style="font-variant-numeric: tabular-nums; min-width: 4.5rem;">
            {seqLen.toLocaleString()} tok
          </span>
        </div>
        <div style={sliderRow}>
          <label style="white-space: nowrap;">drafter cost</label>
          <input
            type="range"
            min={0}
            max={50}
            value={cDraftPct}
            onInput={(e) => setCDraftPct(Number((e.target as HTMLInputElement).value))}
            style="width: 160px;"
          />
          <span style="font-variant-numeric: tabular-nums; min-width: 4.5rem;">{cDraftPct}%</span>
        </div>
        <label style="display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; cursor: pointer;">
          <input
            type="checkbox"
            checked={showComponents}
            onInput={(e) => setShowComponents((e.target as HTMLInputElement).checked)}
          />
          show components
        </label>
      </div>
      <div style="position: relative; height: 340px; width: 100%;">
        <canvas ref={costRef} />
      </div>
    </div>
  )
}
