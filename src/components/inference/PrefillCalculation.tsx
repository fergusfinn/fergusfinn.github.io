import { useStore } from '@nanostores/preact'
import {
  prefillFLOPs,
  matmulFLOPs,
  attentionFLOPs,
  prefillTime,
  totalCompute,
  configStore,
  modelStore,
  formatLargeNumber,
  formatNumber,
} from '../../stores/inferenceStore'
import Tooltip from '../Tooltip'

export default function PrefillCalculation() {
  const flops = useStore(prefillFLOPs)
  const matmulFlops = useStore(matmulFLOPs)
  const attentionFlops = useStore(attentionFLOPs)
  const time = useStore(prefillTime)
  const compute = useStore(totalCompute)
  const config = useStore(configStore)
  const model = useStore(modelStore)

  return (
    <div class="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
      <div class="space-y-3 text-sm">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Matmul FLOPs:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <Tooltip label="ISL">{config.inputSeqLength}</Tooltip> × <Tooltip label="2 FLOPs per param">2</Tooltip> × <Tooltip label="P">{formatLargeNumber(model.modelSize * 1e9)}</Tooltip> = {formatLargeNumber(matmulFlops)} FLOPs
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Attention FLOPs:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <Tooltip label="2 attention matmuls (QK^T + scores@V), each 2ISL²D">4</Tooltip> × <Tooltip label="ISL">{config.inputSeqLength}</Tooltip>² × <Tooltip label="D (hidden_size)">{model.hiddenSize}</Tooltip> × <Tooltip label="L (num_layers)">{model.numLayers}</Tooltip> = {formatLargeNumber(attentionFlops)} FLOPs
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total FLOPs:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {formatLargeNumber(matmulFlops)} + {formatLargeNumber(attentionFlops)} = {formatLargeNumber(flops)} FLOPs
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
