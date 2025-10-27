import { useStore } from '@nanostores/preact'
import {
  decodeTime,
  totalBytesPerDecode,
  totalMemoryBandwidth,
  kvCachePerToken,
  kvCachePerSequence,
  avgSeqLength,
  configStore,
  modelStore,
  formatLargeNumber,
  formatNumber,
} from '../../stores/inferenceStore'
import Tooltip from '../Tooltip'

export default function DecodeCalculation() {
  const time = useStore(decodeTime)
  const totalBytes = useStore(totalBytesPerDecode)
  const bandwidth = useStore(totalMemoryBandwidth)
  const cachePerToken = useStore(kvCachePerToken)
  const cachePerSeq = useStore(kvCachePerSequence)
  const seqLen = useStore(avgSeqLength)
  const config = useStore(configStore)
  const model = useStore(modelStore)

  const totalKVCache = config.concurrentUsers * cachePerSeq

  return (
    <div class="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">KV cache per token:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <Tooltip label="Both K and V">2</Tooltip> × <Tooltip label="n_layers">{model.numLayers}</Tooltip> × <Tooltip label="n_kv_heads">{model.numKVHeads}</Tooltip> × <Tooltip label="head_dim">{model.headDim}</Tooltip> × <Tooltip label="n_bytes">{config.bytesPerParameter}</Tooltip> = {formatLargeNumber(cachePerToken)} bytes
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Average sequence length:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatNumber(seqLen, 0)} tokens</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">KV cache per sequence:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {formatLargeNumber(cachePerSeq)} bytes ≈ {formatNumber(cachePerSeq / 1e6)} MB
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total KV cache:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {config.concurrentUsers} × {formatLargeNumber(cachePerSeq)} = {formatLargeNumber(totalKVCache)} bytes
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Model weights:</div>
          <div class="text-base" style="font-family: var(--font-math)"><Tooltip label="P">{formatLargeNumber(model.modelSize * 1e9)}</Tooltip> × <Tooltip label="n_bytes">{config.bytesPerParameter}</Tooltip> = {formatLargeNumber(model.modelSize * 1e9 * config.bytesPerParameter)} bytes</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total bytes per decode step:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatLargeNumber(model.modelSize * 1e9) + config.concurrentUsers}  × {formatLargeNumber(cachePerSeq)} =  {formatLargeNumber(totalBytes)} bytes</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Memory bandwidth:</div>
          <div class="text-base" style="font-family: var(--font-math)">{formatLargeNumber(bandwidth)} bytes/s</div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Average decode time per step:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {formatLargeNumber(totalBytes)} ÷ {formatLargeNumber(bandwidth)} ={' '}
            <span class="font-bold text-purple-700 dark:text-purple-400">{formatNumber(time)} ms</span>
          </div>
        </div>
      </div>
    </div>
  )
}
