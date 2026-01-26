import { useStore } from '@nanostores/preact'
import {
  configStore,
  modelStore,
  totalMemoryBandwidth,
  totalBytesPerDecode,
  decodeTime,
  kvCachePerSequence,
} from '../../stores/inferenceStore'

export default function FormulaDecode() {
  const config = useStore(configStore)
  const model = useStore(modelStore)
  const bandwidth = useStore(totalMemoryBandwidth)
  const totalBytes = useStore(totalBytesPerDecode)
  const time = useStore(decodeTime)
  const kvCache = useStore(kvCachePerSequence)

  const weightsGB = (model.modelSize * 1e9 * config.bytesPerParameter / 1e9).toFixed(1)
  const kvTotalGB = (config.concurrentUsers * kvCache / 1e9).toFixed(1)
  const totalGB = (totalBytes / 1e9).toFixed(1)
  const bandwidthTBs = (bandwidth / 1e12).toFixed(1)

  return (
    <div class="text-center space-y-6">
      <p class="text-2xl font-mono">
        (<span class="note" data-note="Model weights in memory">weights</span>
        {' '}+{' '}
        <span class="note" data-note="KV cache for all concurrent sequences">B &times; kv_cache</span>)
        {' '}&divide;{' '}
        <span class="note" data-note="Total memory bandwidth across all GPUs">bandwidth</span>
      </p>
      <p class="text-2xl font-mono">
        = ({weightsGB} + {kvTotalGB}) GB &divide; {bandwidthTBs} TB/s
      </p>
      <p class="text-2xl font-mono">
        = <strong>{time.toFixed(2)} ms per token</strong>
      </p>
    </div>
  )
}
