import { useEffect, useRef, useState } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from './chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// ----------------------------------------------------------------------------
// Per-GPU decode throughput vs how thinly the experts are sharded, measured as
// EFFECTIVE EP WIDTH w = E / E_g (the number of GPUs one full expert copy is
// spread across), marked against the smallest DP unit that holds the weights
// plus 64k KV tokens/GPU. The whole trade is: holding fewer experts per GPU frees HBM,
// which becomes KV, which becomes batch — paid for in all-to-all comms.
//
// The placement is "fill each node before spilling across nodes", so a node
// collectively covers a full expert set whenever w <= M: all dispatch stays on
// NVLink, zero scale-out. Past w = M a copy spills across nodes and the off-node
// fraction (1 - M/w) starts paying scale-out, ~10-40x slower. Per-GPU throughput
// depends only on w (and the hardware / model) — the number of DP replicas is an
// orthogonal multiplier on TOTAL throughput, so it drops out here. First-order:
// ignores all-to-all fan-out overhead, which grows with the cooperating group.
// Sources for every spec are in the post body.
// ----------------------------------------------------------------------------

// bw = HBM bandwidth (GB/s); cap = HBM capacity (GB); nvlink = scale-up
// (intra-node) bandwidth per GPU, PER DIRECTION (GB/s) — the relevant quantity
// for a directional dispatch, i.e. half the bidirectional datasheet aggregate.
// Roofline input: per-GPU AGGREGATE all-to-all NVLink rate, PER DIRECTION (the
// half of the bidirectional datasheet aggregate that a directional dispatch
// uses). This is the sum over all of a GPU's links, NOT the per-peer rate — a
// balanced all-to-all lights up every link at once, so the GH200 quad's
// switchless 6-per-peer split (159 GB/s to one peer; confirmed by nvidia-smi NV6
// + p2p measurement on Isambard) is irrelevant here; what counts is its 18-link
// aggregate. An NVSwitch (H200/B200/B300) is a non-blocking crossbar with
// multipath routing — it gives any *permutation* at full rate, but can't exceed
// each GPU's own per-dir link budget, so the all-to-all aggregate is capped at
// that budget either way, switch or no switch.
//   GH200 / H200: NVLink4, 18 links × 25 GB/s/dir (= 50 GB/s/link BIDIRECTIONAL,
//   the "900 GB/s" headline) = 450 GB/s/dir per card.  B200/B300: NVLink5, 18 ×
//   50 GB/s/dir (100 GB/s/link bidir) = 900 GB/s/dir.  We want per-direction here.
// (Realized all-to-all runs below roofline — measured ~315 on the GH200 quad,
// ~66% — but that's an efficiency factor, not part of the peak model.)
const ACCELERATORS = {
  H200: { label: 'H200', bw: 4800, cap: 141, nvlink: 450 },
  B200: { label: 'B200', bw: 8000, cap: 192, nvlink: 900 },
  B300: { label: 'B300', bw: 8000, cap: 288, nvlink: 900 },
  GH200: { label: 'GH200', bw: 4023, cap: 96, nvlink: 450 },
} as const
type AccelKey = keyof typeof ACCELERATORS

// Scale-out per-NIC rate in Gb/s (bytes/s = gbps / 8 * 1e9).
const SCALEOUT = {
  '200': { label: '200 Gb/s · HDR / Slingshot', gbps: 200 },
  '400': { label: '400 Gb/s · NDR', gbps: 400 },
  '800': { label: '800 Gb/s · XDR', gbps: 800 },
  '1600': { label: '1.6 Tb/s · roadmap', gbps: 1600 },
} as const
type ScaleKey = keyof typeof SCALEOUT

// Expert-weight dtype. 'native' resolves to each model's released dtype (below).
const WEIGHT_DTYPE = {
  native: { label: 'Native', bytes: null },
  fp4: { label: 'FP4', bytes: 0.5 },
  fp8: { label: 'FP8', bytes: 1 },
  bf16: { label: 'BF16', bytes: 2 },
} as const
type WeightKey = keyof typeof WEIGHT_DTYPE

// Dispatch activation dtype — bytes/element on the wire, tunable.
const DISPATCH_DTYPE = {
  fp8: { label: 'FP8', bytes: 1 },
  fp4: { label: 'FP4', bytes: 0.5 },
  bf16: { label: 'BF16', bytes: 2 },
} as const
type DispatchKey = keyof typeof DISPATCH_DTYPE

const GPUS_PER_NODE = { '4': 4, '8': 8 } as const
type NodeKey = keyof typeof GPUS_PER_NODE

const T_STEPS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
const fmtT = (t: number) => (t >= 1024 ? `${t / 1024}k` : `${t}`)

const KV_BASELINE_TOKENS = 64_000 // KV tokens/GPU the min DP unit must hold
const X_MAX = 256 // widest effective EP width plotted

// Per-model weight accounting, computed tensor-by-tensor from each model's real
// architecture and its real mixed-precision scheme (sources + full breakdown in
// the post). The weight splits two ways:
//   expertParams — routed-expert param count, the ONLY pool that shards with EP
//     width w. Bytes = expertParams × (native MoE dtype `bytes`, re-quantizable).
//   denseGB — everything replicated on every rank (attention, embeddings, shared
//     experts, dense FFN, router, Mamba), in absolute GB at its real MIXED quant
//     (e.g. DeepSeek: MXFP8 attn + BF16 embed). Not params×one-dtype — it's a sum.
// kv = real KV bytes/token (κ) at the serving KV dtype. mambaState = fixed per-
// sequence state bytes for hybrids (Nemotron's 48 Mamba-2 layers; 0 otherwise).
// Lmoe = number of MoE (dispatch) layers; latent = MoE dispatch width if the
// model down-projects before dispatch (Nemotron LatentMoE), else null → use H.
interface ModelSpec {
  name: string
  H: number
  k: number
  latent: number | null
  bytes: number // native MoE expert dtype, bytes/param
  dtype: string
  expertParams: number // routed-expert params, sharded by w
  denseGB: number // replicated weight, absolute GB at real mixed quant
  // KV decode model from each model's real inference code (bytes/token unless noted):
  kvStore: number // resident KV per token → sets batch. DeepSeek-V4 COMPRESSES storage
  //   (CSA keeps ~1/4, HCA ~1/128); MLA / GQA / NSA store the FULL KV (sparse READ only).
  kvScan: number // per-step read scaling with T: dense layers read all T; sparse layers
  //   scan an O(T) index (or, for V4, the compressed sequence + HCA's all-positions read).
  kvFixed: number // per-step read constant in T (bytes/seq): sliding window + top-k selected.
  kvResident: number // fixed per-seq resident KV, T-independent (V4 sliding-window buffer); 0 else
  mambaState: number // fixed per-sequence Mamba state bytes; read+written each step; 0 otherwise
  Lmoe: number
  color: string
}

const MODELS: ModelSpec[] = [
  // KV figures from each model's real inference code. Cache dtype is BF16 except
  // DeepSeek-V4, which every real engine serves with FP8 KV (so its code-BF16 numbers
  // are halved here). V4 alone compresses STORAGE (CSA 4× / HCA 128×) and keeps a huge
  // batch at long context; GLM & MiniMax sparse the READ but store full KV, so they
  // crater with context like dense Kimi. Nemotron stores Mamba state, not KV.
  { name: 'DeepSeek-V4-Pro', H: 7168, k: 6, latent: null, bytes: 0.5, dtype: 'MXFP4', expertParams: 1471.3e9, denseGB: 15.6, kvStore: 4924, kvScan: 1084, kvFixed: 1.973e7, kvResident: 4.0e6, mambaState: 0, Lmoe: 58, color: '#4e79a7' },
  { name: 'DeepSeek-V4-Flash', H: 4096, k: 6, latent: null, bytes: 0.5, dtype: 'MXFP4', expertParams: 257.7e9, denseGB: 5.7, kvStore: 3440, kvScan: 752, kvFixed: 8.33e6, kvResident: 2.8e6, mambaState: 0, Lmoe: 40, color: '#59a14f' },
  { name: 'Kimi K2.7', H: 7168, k: 8, latent: null, bytes: 0.5, dtype: 'INT4', expertParams: 1014.7e9, denseGB: 23.4, kvStore: 70272, kvScan: 70272, kvFixed: 0, kvResident: 0, mambaState: 0, Lmoe: 60, color: '#e15759' },
  { name: 'GLM 5.2', H: 6144, k: 8, latent: null, bytes: 2, dtype: 'BF16', expertParams: 724.8e9, denseGB: 37.2, kvStore: 94720, kvScan: 4864, kvFixed: 1.84e8, kvResident: 0, mambaState: 0, Lmoe: 75, color: '#f28e2b' },
  { name: 'MiniMax M3', H: 6144, k: 4, latent: null, bytes: 2, dtype: 'BF16', expertParams: 413.1e9, denseGB: 26.9, kvStore: 137472, kvScan: 20736, kvFixed: 2.39e8, kvResident: 0, mambaState: 0, Lmoe: 57, color: '#b07aa1' },
  { name: 'Nemotron 3 Ultra', H: 8192, k: 22, latent: 2048, bytes: 0.5, dtype: 'NVFP4', expertParams: 515.4e9, denseGB: 67.8, kvStore: 12288, kvScan: 12288, kvFixed: 0, kvResident: 0, mambaState: 413e6, Lmoe: 48, color: '#76b7b2' },
]

// Per-GPU decode throughput (relative units) at effective EP width w = E / E_g.
// w = 1: every GPU holds all experts (pure DP, zero dispatch). w = M: one full
// copy per node (max batch with zero scale-out). w > M: a copy spills off-node.
// Returns null when the weights don't leave room for any KV at this width.
function phi(
  m: ModelSpec,
  w: number,
  M: number,
  C: number,
  BW_HBM: number,
  BW_intra: number,
  BW_inter: number,
  weightBytes: number | null,
  T: number,
  ba: number,
): number | null {
  const bExp = weightBytes ?? m.bytes
  // Replicated dense weight (per GPU) + expert weight sharded across w GPUs.
  const weightPerGpu = m.denseGB * 1e9 + (m.expertParams * bExp) / w
  const kvPerGpu = C - weightPerGpu
  if (kvPerGpu <= 0) return null
  // Batch that fits: leftover HBM / per-seq footprint = per-token KV (resident,
  // possibly compressed) × T + fixed resident KV (window) + Mamba state.
  const Bi = kvPerGpu / (T * m.kvStore + m.kvResident + m.mambaState)
  // Decode roofline — HBM traffic PER STEP: stream the weights once; the attention
  // read = a T-scaling scan (kvScan: dense layers read all T, sparse layers scan a
  // cheap index / compressed sequence) + a fixed window+top-k read (kvFixed);
  // read+write the Mamba state.
  const kvRead = Bi * (T * m.kvScan + m.kvFixed)
  const mambaIO = 2 * Bi * m.mambaState // recurrent state read + write each step
  const tHBM = (weightPerGpu + kvRead + mambaIO) / BW_HBM
  const Hdisp = m.latent ?? m.H
  const coef = m.Lmoe * 2 * m.k * ba * Hdisp
  // Locality of a token's needed expert under fill-the-node-first placement.
  const fLocal = Math.min(1, 1 / w) // same GPU — free
  const fOnNode = Math.min(1, M / w) // somewhere on this node
  const fOnRemote = Math.max(0, fOnNode - fLocal) // on node, another GPU
  const fOff = Math.max(0, 1 - fOnNode) // off node
  const comms = Bi * coef * Math.max(fOnRemote / BW_intra, fOff / BW_inter)
  return Bi / (tHBM + comms)
}

// Smallest effective width whose per-GPU HBM holds KV_BASELINE_TOKENS of KV —
// the min DP unit we mark against. Token-budgeted, so it's independent of T
// (longer sequences just mean fewer of them in the same pool). Returns null if
// even the widest shard can't free the room.
function baselineWidth(m: ModelSpec, C: number, weightBytes: number | null): number | null {
  const bExp = weightBytes ?? m.bytes
  // HBM left for sharded experts after dense weight and the KV budget.
  const reserve = C - m.denseGB * 1e9 - KV_BASELINE_TOKENS * m.kvStore
  if (reserve <= 0) return null
  const w = Math.ceil((m.expertParams * bExp) / reserve)
  return w <= X_MAX ? Math.max(w, 1) : null
}

// Deployable effective widths: a copy spans an integer number of GPUs, and once
// it leaves a node it spans whole nodes. So 1..M intra-node, then M, 2M, 3M...
function realizableWidths(M: number): number[] {
  const ws: number[] = []
  for (let w = 1; w <= M; w++) ws.push(w)
  for (let n = 2; n * M <= X_MAX; n++) ws.push(n * M)
  return ws
}

// Round a continuous width up to the nearest deployable grid point.
function snapUp(w: number, grid: number[]): number | null {
  for (const g of grid) if (g >= w - 1e-9) return g
  return null
}

// Faint vertical line at w = M, where a copy stops fitting inside one node.
const nodeLinePlugin = {
  id: 'nodeline',
  afterDatasetsDraw(chart: Chart, _a: unknown, opts: { M: number; color: string; font: string }) {
    const { ctx, chartArea, scales } = chart
    const x = scales.x.getPixelForValue(opts.M)
    if (x < chartArea.left || x > chartArea.right) return
    ctx.save()
    ctx.strokeStyle = opts.color
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath()
    ctx.moveTo(x, chartArea.top)
    ctx.lineTo(x, chartArea.bottom)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = opts.color
    ctx.font = `600 11px ${opts.font}`
    ctx.textAlign = 'center'
    ctx.fillText('1 copy / node', x, chartArea.top + 12)
    ctx.restore()
  },
}

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <label class="flex flex-col gap-1 text-xs">
      <span class="font-semibold opacity-70">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange((e.target as HTMLSelectElement).value as T)}
        class="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {options.map((o) => (
          <option value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

export default function WideEPExplorer() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  const [accel, setAccel] = useState<AccelKey>('GH200')
  const [scale, setScale] = useState<ScaleKey>('200')
  const [node, setNode] = useState<NodeKey>('4')
  const [weight, setWeight] = useState<WeightKey>('native')
  const [dispatch, setDispatch] = useState<DispatchKey>('fp8')
  const [tIdx, setTIdx] = useState<number>(3) // 8192

  const a = ACCELERATORS[accel]
  const s = SCALEOUT[scale]
  const M = GPUS_PER_NODE[node]
  const T = T_STEPS[tIdx]
  const weightBytes = WEIGHT_DTYPE[weight].bytes
  const ba = DISPATCH_DTYPE[dispatch].bytes

  useEffect(() => {
    if (!chartRef.current) return
    if (chartInstance.current) chartInstance.current.destroy()
    applyChartDefaults(theme)

    const C = a.cap * 1e9
    const BW_HBM = a.bw * 1e9
    const BW_intra = a.nvlink * 1e9
    const BW_inter = (s.gbps / 8) * 1e9

    // Sample each model's curve continuously in w (partial replication makes
    // every width reachable), with extra points bracketing w = M so the scale-out
    // ramp renders crisply. The y value is absolute per-GPU throughput (relative
    // units) — divided by nothing, so curves sit at honest heights and never
    // vanish as T grows. Only the DP unit (smallest width holding 64k KV
    // tokens/GPU) is snapped to a deployable width and marked as a ◆.
    const grid = realizableWidths(M)
    const datasets = MODELS.map((m) => {
      const bExp = weightBytes ?? m.bytes
      const kvRoom = C - m.denseGB * 1e9 // HBM left once dense weight is in
      if (kvRoom <= 0) {
        return { label: m.name, data: [], borderColor: m.color, backgroundColor: m.color }
      }
      const floor = (m.expertParams * bExp) / kvRoom // narrowest width with any KV room
      const start = Math.max(floor * 1.02, 1)
      if (start >= X_MAX) {
        return { label: m.name, data: [], borderColor: m.color, backgroundColor: m.color }
      }
      const wBaseCont = baselineWidth(m, C, weightBytes) // T-independent (64k tok/GPU)
      const wBase = wBaseCont != null ? snapUp(wBaseCont, grid) : null
      const ws = new Set<number>()
      const N = 72
      for (let i = 0; i <= N; i++) ws.add(start * Math.pow(X_MAX / start, i / N))
      ;[M * 0.99, M, M * 1.02, M * 1.1, M * 1.3, M * 1.7, 2 * M].forEach((x) => {
        if (x > start && x < X_MAX) ws.add(x)
      })
      if (wBase != null && wBase > start && wBase < X_MAX) ws.add(wBase)
      const pts = [...ws]
        .sort((p, q) => p - q)
        .map((w) => ({ x: w, y: phi(m, w, M, C, BW_HBM, BW_intra, BW_inter, weightBytes, T, ba), w }))
        .filter((p): p is { x: number; y: number; w: number } => p.y != null)
        .map((p) => ({ x: p.x, y: p.y, isBase: wBase != null && Math.abs(p.w - wBase) < 1e-9 }))
      return {
        label: m.name,
        data: pts,
        borderColor: m.color,
        backgroundColor: m.color,
        borderWidth: 2,
        tension: 0.2,
        pointRadius: (ctx: { raw?: { isBase?: boolean } }) => (ctx.raw?.isBase ? 5 : 0),
        pointStyle: 'rectRot',
        pointHoverRadius: 4,
      }
    })

    chartInstance.current = new Chart(chartRef.current, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: 'nearest', intersect: false },
        layout: { padding: { top: 8, right: 12 } },
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: {
              color: theme.foreground,
              font: { family: theme.fontFamily, size: 11 },
              boxWidth: 18,
              boxHeight: 2,
            },
          },
          tooltip: {
            callbacks: {
              title: (items) => `w ≈ ${(items[0].parsed.x ?? 0).toFixed(1)} (E/E_g)`,
              label: (ctx) => {
                const tag = (ctx.raw as { isBase?: boolean })?.isBase ? '  · DP unit (64k tok/GPU)' : ''
                return `  ${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toExponential(1)} tok/s/GPU${tag}`
              },
            },
          },
          // @ts-expect-error custom plugin options
          nodeline: { M, color: theme.mutedForeground, font: theme.fontFamily },
        },
        scales: {
          x: {
            type: 'logarithmic',
            min: 1,
            max: X_MAX,
            title: {
              display: true,
              text: 'effective EP width  w = E/E_g',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            afterBuildTicks: (axis) => {
              axis.ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256].map((v) => ({ value: v }))
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
              callback: (v) => `${v}`,
            },
            grid: { color: theme.grid },
          },
          y: {
            type: 'logarithmic',
            title: {
              display: true,
              text: 'tok/s/GPU',
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
            },
            ticks: {
              color: theme.mutedForeground,
              font: { family: theme.fontFamily },
              callback: (v) => {
                const n = v as number
                return Number.isInteger(Math.log10(n)) ? n.toExponential(0) : ''
              },
            },
            grid: { color: theme.grid },
          },
        },
      },
      plugins: [nodeLinePlugin],
    })

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme, accel, scale, node, weight, dispatch, tIdx])

  return (
    <div class="my-6 not-prose">
      <div class="flex flex-wrap gap-3 mb-4 items-end">
        <Select
          label="Accelerator"
          value={accel}
          onChange={setAccel}
          options={(Object.keys(ACCELERATORS) as AccelKey[]).map((k) => ({
            value: k,
            label: ACCELERATORS[k].label,
          }))}
        />
        <Select
          label="GPUs / node"
          value={node}
          onChange={setNode}
          options={(Object.keys(GPUS_PER_NODE) as NodeKey[]).map((k) => ({ value: k, label: k }))}
        />
        <Select
          label="Scale-out / NIC"
          value={scale}
          onChange={setScale}
          options={(Object.keys(SCALEOUT) as ScaleKey[]).map((k) => ({
            value: k,
            label: SCALEOUT[k].label,
          }))}
        />
        <Select
          label="Expert dtype"
          value={weight}
          onChange={setWeight}
          options={(Object.keys(WEIGHT_DTYPE) as WeightKey[]).map((k) => ({
            value: k,
            label: WEIGHT_DTYPE[k].label,
          }))}
        />
        <Select
          label="Dispatch dtype"
          value={dispatch}
          onChange={setDispatch}
          options={(Object.keys(DISPATCH_DTYPE) as DispatchKey[]).map((k) => ({
            value: k,
            label: DISPATCH_DTYPE[k].label,
          }))}
        />
        <label class="flex flex-col gap-1 text-xs grow min-w-[150px]">
          <span class="font-semibold opacity-70">Avg sequence length T = {fmtT(T)}</span>
          <input
            type="range"
            min={0}
            max={T_STEPS.length - 1}
            step={1}
            value={tIdx}
            onInput={(e) => setTIdx(Number((e.target as HTMLInputElement).value))}
            class="accent-blue-500"
          />
        </label>
      </div>

      <div style="position: relative; height: 340px; width: 100%;">
        <canvas ref={chartRef} />
      </div>
    </div>
  )
}
