import { useStore } from '@nanostores/preact'
import {
  throughput,
  throughputPerGPU,
  totalPrefillTime,
  totalDecodeTime,
  totalTimeWithMode,
  totalOutputTokens,
  configStore,
  chunkedPrefillingEnabled,
  canOverlapPrefills,
  formatNumber,
} from '../../stores/inferenceStore'

export default function ThroughputCalculation() {
  const tput = useStore(throughput)
  const tputPerGPU = useStore(throughputPerGPU)
  const prefillTime = useStore(totalPrefillTime)
  const decodeTime = useStore(totalDecodeTime)
  const totalT = useStore(totalTimeWithMode)
  const outputTokens = useStore(totalOutputTokens)
  const config = useStore(configStore)
  const chunkedMode = useStore(chunkedPrefillingEnabled)
  const canOverlap = useStore(canOverlapPrefills)

  const toggleChunkedPrefilling = () => {
    chunkedPrefillingEnabled.setKey('enabled', !chunkedMode.enabled)
  }

  return (
    <div class="p-4 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800">
      <div class="flex justify-end -mb-2">
        <label class="flex items-center gap-2 cursor-pointer select-none">
          <span class="text-sm font-semibold">Chunked prefilling</span>
          <div class="relative">
            <input
              type="checkbox"
              checked={chunkedMode.enabled}
              onChange={toggleChunkedPrefilling}
              class="peer sr-only"
            />
            <div class="w-4 h-4 border-2 border-cyan-600 dark:border-cyan-400 rounded peer-checked:bg-cyan-600 dark:peer-checked:bg-cyan-400 transition-colors">
              {chunkedMode.enabled && (
                <svg class="w-full h-full text-white dark:text-gray-900" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8L6.5 11.5L13 4.5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              )}
            </div>
          </div>
        </label>
      </div>
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total prefill time:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {chunkedMode.enabled ? (
              canOverlap ? (
                '0 ms (fully overlapped)'
              ) : (
                `${formatNumber(totalT - decodeTime)} ms (partial overlap)`
              )
            ) : (
              <>
                {config.concurrentUsers} sequences × {formatNumber(prefillTime / config.concurrentUsers)} ms ={' '}
                {formatNumber(prefillTime)} ms
              </>
            )}
          </div>
        </div>

        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total decode time:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {config.outputSeqLength} steps × {formatNumber(decodeTime / config.outputSeqLength)} ms ={' '}
            {formatNumber(decodeTime)} ms
          </div>
        </div>

        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Total time:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {!chunkedMode.enabled ? (
              <>
                {formatNumber(prefillTime)} + {formatNumber(decodeTime)} = {formatNumber(totalT)} ms ≈{' '}
                {formatNumber(totalT / 1000)} s
              </>
            ) : (
              <>
                {formatNumber(totalT)} ms ≈ {formatNumber(totalT / 1000)} s
              </>
            )}
          </div>
        </div>

        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Output tokens per cycle:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {config.concurrentUsers} × {config.outputSeqLength} = {outputTokens} tokens
          </div>
        </div>

        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Throughput:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {outputTokens} ÷ {formatNumber(totalT / 1000)} s ={' '}
            <span class="font-bold text-cyan-700 dark:text-cyan-400">
              {formatNumber(tput, 0)} tokens/s
            </span>
          </div>
        </div>

        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Per GPU (TP = {config.tensorParallelism}):</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <span class="font-bold text-cyan-700 dark:text-cyan-400">
              {formatNumber(tputPerGPU, 0)} tokens/s per GPU
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
