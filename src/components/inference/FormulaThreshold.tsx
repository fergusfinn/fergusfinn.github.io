import { useStore } from '@nanostores/preact'
import {
  computeBoundThreshold,
  totalCompute,
  totalMemoryBandwidth,
  configStore,
} from '../../stores/inferenceStore'

export default function FormulaThreshold() {
  const threshold = useStore(computeBoundThreshold)
  const compute = useStore(totalCompute)
  const bandwidth = useStore(totalMemoryBandwidth)
  const config = useStore(configStore)

  const computeTFLOPS = (compute / 1e12).toFixed(0)
  const bandwidthTBs = (bandwidth / 1e12).toFixed(1)

  return (
    <div class="text-center space-y-6">
      <p class="text-2xl font-mono">
        <span class="note" data-note="Total FLOP/s across all GPUs">compute</span>
        {' '}&divide;{' '}
        (2 &times; <span class="note" data-note="Total memory bandwidth across all GPUs">bandwidth</span>
        {' '}&divide; <span class="note" data-note="Bytes per parameter (FP8 = 1, FP4 = 0.5)">bytes_per_param</span>)
      </p>
      <p class="text-2xl font-mono">
        = {computeTFLOPS} TFLOP/s &divide; (2 &times; {bandwidthTBs} TB/s &divide; {config.bytesPerParameter})
      </p>
      <p class="text-2xl font-mono">
        = <strong>{threshold} tokens</strong>
      </p>
    </div>
  )
}
