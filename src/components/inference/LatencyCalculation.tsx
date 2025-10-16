import { useStore } from '@nanostores/preact'
import {
  e2eLatency,
  prefillTime,
  decodeTime,
  configStore,
  formatNumber,
} from '../../stores/inferenceStore'

export default function LatencyCalculation() {
  const latency = useStore(e2eLatency)
  const prefill = useStore(prefillTime)
  const decode = useStore(decodeTime)
  const config = useStore(configStore)

  return (
    <div class="p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg border border-orange-200 dark:border-orange-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Prefill time:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatNumber(prefill)} ms</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Decode time per token:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatNumber(decode)} ms</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Output tokens:</div>
          <div class="text-base" style="font-family: var(--font-math)">{config.outputSeqLength} tokens</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total latency:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {formatNumber(prefill)} + ({config.outputSeqLength} × {formatNumber(decode)}) ={' '}
            <span class="font-bold text-orange-700 dark:text-orange-400">
              {formatNumber(latency)} ms ≈ {formatNumber(latency / 1000)} s
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
