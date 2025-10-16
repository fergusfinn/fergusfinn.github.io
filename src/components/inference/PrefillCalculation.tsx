import { useStore } from '@nanostores/preact'
import {
  prefillFLOPs,
  prefillTime,
  totalCompute,
  configStore,
  modelStore,
  formatLargeNumber,
  formatNumber,
} from '../../stores/inferenceStore'

export default function PrefillCalculation() {
  const flops = useStore(prefillFLOPs)
  const time = useStore(prefillTime)
  const compute = useStore(totalCompute)
  const config = useStore(configStore)
  const model = useStore(modelStore)

  return (
    <div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
      <div class="space-y-3 text-sm">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">FLOPs required:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {config.inputSeqLength} × 2 × {formatLargeNumber(model.modelSize * 1e9)} = {formatLargeNumber(flops)} FLOPs
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total compute available:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatLargeNumber(compute)} FLOP/s</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Prefill time per sequence:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {formatLargeNumber(flops)} ÷ {formatLargeNumber(compute)} ={' '}
            <span class="font-bold text-green-700 dark:text-green-400">{formatNumber(time)} ms</span>
          </div>
        </div>
      </div>
    </div>
  )
}
