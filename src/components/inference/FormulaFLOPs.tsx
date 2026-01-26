import { useStore } from '@nanostores/preact'
import { configStore, modelStore, matmulFLOPs } from '../../stores/inferenceStore'

export default function FormulaFLOPs() {
  const config = useStore(configStore)
  const model = useStore(modelStore)
  const flops = useStore(matmulFLOPs)

  const ISL = config.inputSeqLength
  const P = model.modelSize
  const tflops = flops / 1e12

  return (
    <div class="text-center space-y-6">
      <p class="text-2xl font-mono">
        2 &times; <span class="note" data-note="Input sequence length">ISL</span> &times; <span class="note" data-note="Model parameters">P</span>
      </p>
      <p class="text-2xl font-mono">
        = 2 &times; {ISL.toLocaleString()} &times; {P}B = <strong>{tflops.toFixed(1)} TFLOPs</strong>
      </p>
    </div>
  )
}
