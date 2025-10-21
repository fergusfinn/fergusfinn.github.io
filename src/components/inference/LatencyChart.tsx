import { useState, useEffect, useRef } from 'preact/hooks'
import { useStore } from '@nanostores/preact'
import {
  configStore,
  modelStore,
  e2eLatency,
  ACCELERATORS,
} from '../../stores/inferenceStore'
import { chartControlsStore } from './ChartControls'
import { Chart, registerables } from 'chart.js'

// Register Chart.js components
if (typeof window !== 'undefined') {
  Chart.register(...registerables)

  // Set default font to match site
  Chart.defaults.font.family = "'Source Sans 3', sans-serif"
  Chart.defaults.font.size = 14
}

interface BenchmarkRow {
  Hardware: string
  Framework: string
  Precision: string
  TP: number
  Conc: number
  'TTFT (ms)': number
  'TPOT (ms)': number
  'E2EL (s)': number
  'TPUT per GPU': number
}

type XAxisVariable = 'concurrency' | 'tensorParallelism'

export default function LatencyChart() {
  const config = useStore(configStore)
  const model = useStore(modelStore)
  const calcLatency = useStore(e2eLatency)
  const chartControls = useStore(chartControlsStore)

  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  const xAxis = chartControls.latencyXAxis
  const isRelative = chartControls.latencyRelative

  // Calculate theoretical latency for given TP and concurrency
  const calculateTheoreticalLatency = (tp: number, concurrency: number): number => {
    const ISL = config.inputSeqLength
    const OSL = config.outputSeqLength
    const accelerator = ACCELERATORS[config.acceleratorType]

    // Get precision-specific compute
    let compute: number
    if (config.bytesPerParameter === 0.5) {
      compute = accelerator.computeFP4
    } else if (config.bytesPerParameter === 1) {
      compute = accelerator.computeFP8
    } else {
      compute = accelerator.computeFP16
    }

    // Scale by TP
    const totalCompute = compute * tp * 1e12 // Convert TFLOPS to FLOPS
    const totalBandwidth = accelerator.memoryBandwidth * tp * 1e12 // Convert TB/s to bytes/s

    // Prefill time (compute bound)
    const prefillFLOPs = ISL * 2 * model.modelSize * 1e9
    const prefillTime = (prefillFLOPs / totalCompute) * 1000 // ms

    // Decode time (memory bandwidth bound)
    const avgSeqLen = ISL + OSL / 2
    const kvCachePerToken = 2 * model.numLayers * model.numKVHeads * model.headDim * config.bytesPerParameter
    const kvCache = concurrency * avgSeqLen * kvCachePerToken
    const modelWeights = model.modelSize * 1e9 * config.bytesPerParameter
    const bytesPerDecode = modelWeights + kvCache
    const decodeTime = (bytesPerDecode / totalBandwidth) * 1000 // ms

    // Total latency
    const latency = prefillTime + OSL * decodeTime // ms
    return latency / 1000 // Convert to seconds
  }

  // Map our precision to benchmark precision names
  const getPrecisionName = (bytes: number): string => {
    if (bytes === 0.5) return 'FP4'
    if (bytes === 1) return 'FP8'
    if (bytes === 2) return 'FP16'
    return 'FP8' // default
  }

  // Map our accelerator names to benchmark hardware names
  const getHardwareName = (accelType: string): string => {
    return accelType
  }

  // Get sequence config (e.g., "1k1k-70b", "8k1k-70b")
  const getSeqConfig = (): string => {
    const isl = config.inputSeqLength
    const osl = config.outputSeqLength
    const formatSeqLen = (len: number) => len >= 1000 ? `${Math.round(len / 1000)}k` : `${len}`
    return `${formatSeqLen(isl)}${formatSeqLen(osl)}-70b`
  }

  const loadBenchmarkData = async () => {
    setLoading(true)
    setError(null)

    try {
      const seqConfig = getSeqConfig()
      const hardware = getHardwareName(config.acceleratorType)
      const precision = getPrecisionName(config.bytesPerParameter)

      const url = `/benchmark-data/${seqConfig}/${hardware}/${precision}.json`
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error(`No benchmark data available for ${seqConfig}/${hardware}/${precision}`)
      }

      const data: BenchmarkRow[] = await response.json()
      setBenchmarkData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load benchmark data')
      setBenchmarkData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadBenchmarkData()
  }, [config.acceleratorType, config.bytesPerParameter, config.inputSeqLength, config.outputSeqLength, config.tensorParallelism, config.concurrentUsers])

  // Create/update chart
  useEffect(() => {
    if (!benchmarkData || !chartRef.current) return

    // Filter and sort data based on x-axis variable
    let filteredData: BenchmarkRow[]
    if (xAxis === 'concurrency') {
      // Filter for current TP, vary concurrency
      filteredData = benchmarkData.filter(row => row.TP === config.tensorParallelism)
      if (filteredData.length === 0) return
      filteredData.sort((a, b) => a.Conc - b.Conc)
    } else {
      // Filter for current concurrency, vary TP
      filteredData = benchmarkData.filter(row => row.Conc === config.concurrentUsers)
      if (filteredData.length === 0) return
      filteredData.sort((a, b) => a.TP - b.TP)
    }

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const ctx = chartRef.current.getContext('2d')
    if (!ctx) return

    const xAxisLabel = xAxis === 'concurrency' ? 'Concurrent Users' : 'Tensor Parallelism'
    const chartTitle = `E2E Latency vs ${xAxisLabel}`

    const datasets = isRelative
      ? [
          {
            label: 'Efficiency (%)',
            data: filteredData.map(row => {
              const actual = row['E2EL (s)']
              const theoretical = calculateTheoreticalLatency(row.TP, row.Conc)
              // For latency, lower is better, so theoretical/actual gives efficiency
              return (theoretical / actual) * 100
            }),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 4,
            tension: 0.1,
          },
        ]
      : [
          {
            label: 'Actual Latency',
            data: filteredData.map(row => row['E2EL (s)']),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 4,
            tension: 0.1,
          },
          {
            label: 'Theoretical Latency',
            data: filteredData.map(row =>
              xAxis === 'tensorParallelism'
                ? calculateTheoreticalLatency(row.TP, row.Conc)
                : calculateTheoreticalLatency(row.TP, row.Conc)
            ),
            borderColor: 'rgb(234, 88, 12)',
            backgroundColor: 'rgba(234, 88, 12, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            tension: 0,
          },
        ]

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: filteredData.map(row => xAxis === 'concurrency' ? `${row.Conc}` : `${row.TP}`),
        datasets,
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
          },
          title: {
            display: false,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: xAxisLabel,
            },
          },
          y: {
            title: {
              display: true,
              text: isRelative ? 'Efficiency (% of Theoretical)' : 'Latency (seconds)',
            },
            beginAtZero: true,
          },
        },
      },
    })

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [benchmarkData, config.tensorParallelism, config.concurrentUsers, config.inputSeqLength, config.outputSeqLength, config.bytesPerParameter, config.acceleratorType, model, calcLatency, xAxis, isRelative])

  return (
    <div class="my-6">
      <span class="sidenote-unnumbered-wrapper">
        <span class="sidenote-unnumbered">
          <div class="sidebar-content">
            <h3 class="sidebar-title">Latency Chart</h3>
            <div class="space-y-3">
              <div>
                <label class="block text-xs font-semibold mb-1">X-axis</label>
                <select
                  value={xAxis}
                  onChange={(e) => chartControlsStore.setKey('latencyXAxis', (e.target as HTMLSelectElement).value as XAxisVariable)}
                  class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="concurrency">Concurrent Users</option>
                  <option value="tensorParallelism">Tensor Parallelism</option>
                </select>
              </div>
              <div>
                <label class="flex items-center text-xs font-semibold cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isRelative}
                    onChange={(e) => chartControlsStore.setKey('latencyRelative', (e.target as HTMLInputElement).checked)}
                    class="mr-2"
                  />
                  Show Relative Performance
                </label>
              </div>
            </div>
          </div>
        </span>
      </span>

      <div class="space-y-3">
        {loading && <div class="text-sm text-gray-600 dark:text-gray-400">Loading benchmark data...</div>}

        {error && <div class="text-sm text-red-600 dark:text-red-400">{error}</div>}

        {benchmarkData && benchmarkData.length > 0 && (
          <div style="height: 400px; position: relative;">
            <canvas ref={chartRef}></canvas>
          </div>
        )}
      </div>
    </div>
  )
}
