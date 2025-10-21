import { useState, useEffect, useRef } from 'preact/hooks'
import { useStore } from '@nanostores/preact'
import {
  configStore,
  modelStore,
  totalCompute,
  totalDecodeTime,
  computeBoundThreshold,
  chunkedPrefillingEnabled,
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

export default function ThroughputChart() {
  const config = useStore(configStore)
  const model = useStore(modelStore)
  const compute = useStore(totalCompute)
  const decodeTime = useStore(totalDecodeTime)
  const threshold = useStore(computeBoundThreshold)
  const chunkedMode = useStore(chunkedPrefillingEnabled)
  const chartControls = useStore(chartControlsStore)

  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  const xAxis = chartControls.throughputXAxis

  // Calculate theoretical throughput for a given concurrency
  const calculateTheoreticalThroughput = (concurrency: number): number => {
    const ISL = config.inputSeqLength
    const OSL = config.outputSeqLength

    // Calculate non-overlapped prefill tokens for this concurrency
    const nonOverlappedTokens = Math.max(0, concurrency * (ISL + OSL) - OSL * threshold)

    // Calculate time for non-overlapped prefill
    const nonOverlappedTime = (nonOverlappedTokens * 2 * model.modelSize * 1e9 / compute) * 1000

    // Total time (decode + non-overlapped prefill)
    const totalTime = chunkedMode.enabled ? (decodeTime + nonOverlappedTime) : (concurrency * (ISL * 2 * model.modelSize * 1e9 / compute) * 1000 + decodeTime)

    // Throughput = total output tokens / total time
    const totalOutputTokens = concurrency * OSL
    const throughput = (totalOutputTokens / totalTime) * 1000 // tokens/s

    return throughput / config.tensorParallelism // per GPU
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
    const chartTitle = `Throughput per GPU vs ${xAxisLabel}`

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: filteredData.map(row => xAxis === 'concurrency' ? `${row.Conc}` : `${row.TP}`),
        datasets: [
          {
            label: 'Actual Throughput',
            data: filteredData.map(row => row['TPUT per GPU']),
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 4,
            tension: 0.1,
          },
          {
            label: 'Theoretical Throughput',
            data: filteredData.map(row => calculateTheoreticalThroughput(row.Conc)),
            borderColor: 'rgb(234, 88, 12)',
            backgroundColor: 'rgba(234, 88, 12, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            tension: 0,
          },
        ],
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
              text: 'Throughput (tokens/s per GPU)',
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
  }, [benchmarkData, config.tensorParallelism, config.concurrentUsers, compute, decodeTime, threshold, chunkedMode, model.modelSize, config.inputSeqLength, config.outputSeqLength, xAxis])

  return (
    <div class="my-6">
      <span class="sidenote-unnumbered-wrapper">
        <span class="sidenote-unnumbered">
          <div class="sidebar-content">
            <h3 class="sidebar-title">Throughput Chart</h3>
            <div>
              <label class="block text-xs font-semibold mb-1">X-axis</label>
              <select
                value={xAxis}
                onChange={(e) => chartControlsStore.setKey('throughputXAxis', (e.target as HTMLSelectElement).value as XAxisVariable)}
                class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="concurrency">Concurrent Users</option>
                <option value="tensorParallelism">Tensor Parallelism</option>
              </select>
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
