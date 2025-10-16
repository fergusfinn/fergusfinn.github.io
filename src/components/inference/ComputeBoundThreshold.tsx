import { useStore } from '@nanostores/preact'
import {
  computeBoundThreshold,
  totalCompute,
  totalMemoryBandwidth,
  formatLargeNumber,
} from '../../stores/inferenceStore'

export default function ComputeBoundThreshold() {
  const threshold = useStore(computeBoundThreshold)
  const compute = useStore(totalCompute)
  const bandwidth = useStore(totalMemoryBandwidth)

  return (
    <div class="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Threshold for matmuls to be compute bound:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            B ≥ {formatLargeNumber(compute)} ÷ {formatLargeNumber(bandwidth)} ={' '}
            <span class="font-bold text-blue-700 dark:text-blue-400">{threshold} tokens</span>
          </div>
        </div>
      </div>
    </div>
  )
}
