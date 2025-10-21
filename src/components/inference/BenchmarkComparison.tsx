import { useState, useEffect } from 'preact/hooks'
import { useStore } from '@nanostores/preact'
import {
  configStore,
  throughputPerGPU,
  e2eLatency,
  formatNumber,
} from '../../stores/inferenceStore'

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

export default function BenchmarkComparison() {
  const config = useStore(configStore)
  const calcThroughput = useStore(throughputPerGPU)
  const calcLatency = useStore(e2eLatency)

  const [benchmarkData, setBenchmarkData] = useState<BenchmarkRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Map our precision to benchmark precision names
  const getPrecisionName = (bytes: number): string => {
    if (bytes === 0.5) return 'FP4'
    if (bytes === 1) return 'FP8'
    if (bytes === 2) return 'FP16'
    return 'FP8' // default
  }

  // Map our accelerator names to benchmark hardware names
  const getHardwareName = (accelType: string): string => {
    // Benchmark data uses simple names like "H100", "B200", etc.
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

  // Find matching row based on TP and Conc
  const matchingRow = benchmarkData?.find(
    row => row.TP === config.tensorParallelism && row.Conc === config.concurrentUsers
  )

  return (
    <div class="my-6">
      <div class="space-y-3">
        {loading && <div class="text-sm text-gray-600 dark:text-gray-400">Loading benchmark data...</div>}

        {error && <div class="text-sm text-red-600 dark:text-red-400">{error}</div>}

          {matchingRow && (
            <div class="overflow-x-auto">
              <table class="min-w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-300 dark:border-gray-700">
                    <th class="text-left py-2 px-2">Metric</th>
                    <th class="text-right py-2 px-2">Theoretical</th>
                    <th class="text-right py-2 px-2">Actual</th>
                    <th class="text-right py-2 px-2">% of Theoretical</th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="border-b border-gray-200 dark:border-gray-800">
                    <td class="py-2 px-2">E2E Latency</td>
                    <td class="text-right py-2 px-2">{formatNumber(calcLatency / 1000)} s</td>
                    <td class="text-right py-2 px-2">{formatNumber(matchingRow['E2EL (s)'])} s</td>
                    <td class="text-right py-2 px-2">
                      {formatNumber((matchingRow['E2EL (s)'] / (calcLatency / 1000)) * 100)}%
                    </td>
                  </tr>
                  <tr class="border-b border-gray-200 dark:border-gray-800">
                    <td class="py-2 px-2">Throughput/GPU</td>
                    <td class="text-right py-2 px-2">{formatNumber(calcThroughput, 0)} tok/s</td>
                    <td class="text-right py-2 px-2">{formatNumber((matchingRow.Conc / (matchingRow['TPOT (ms)'] / 1000)) / matchingRow.TP, 0)} tok/s</td>
                    <td class="text-right py-2 px-2">
                      {formatNumber(((matchingRow.Conc / (matchingRow['TPOT (ms)'] / 1000)) / matchingRow.TP / calcThroughput) * 100)}%
                    </td>
                  </tr>
                </tbody>
              </table>
              <div class="mt-2 text-xs text-gray-600 dark:text-gray-400">
                Framework: {matchingRow.Framework} | TP={matchingRow.TP} | Conc={matchingRow.Conc} | 21 Oct 2025 | Source: <a href="https://inferencemax.semianalysis.com/" target="_blank" rel="noopener noreferrer" class="underline hover:text-gray-800 dark:hover:text-gray-300">SemiAnalysis</a>
              </div>
            </div>
          )}

        {!loading && !error && benchmarkData && !matchingRow && (
          <div class="text-sm text-gray-600 dark:text-gray-400">
            No benchmark data for TP={config.tensorParallelism}, Conc={config.concurrentUsers}
          </div>
        )}
      </div>
    </div>
  )
}
