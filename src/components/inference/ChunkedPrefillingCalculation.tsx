import { useStore } from '@nanostores/preact'
import {
  maxNumBatchedTokens,
  decodeTokensUsed,
  availableTokensForPrefill,
  completePrefillsFit,
  canOverlapPrefills,
  configStore,
  formatNumber,
} from '../../stores/inferenceStore'

export default function ChunkedPrefillingCalculation() {
  const decodeTokens = useStore(decodeTokensUsed)
  const available = useStore(availableTokensForPrefill)
  const prefillsFit = useStore(completePrefillsFit)
  const canOverlap = useStore(canOverlapPrefills)
  const config = useStore(configStore)

  return (
    <div class="p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-200 dark:border-indigo-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Max batched tokens:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {maxNumBatchedTokens} tokens
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Tokens used by decodes (steady state):</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {config.concurrentUsers} users × 1 token = {decodeTokens} tokens
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Available for prefills:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {maxNumBatchedTokens} − {decodeTokens} = {available} tokens
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Complete prefills that fit per step:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            {available} ÷ {config.inputSeqLength} ={' '}
            <span class="font-bold text-indigo-700 dark:text-indigo-400">{prefillsFit} sequences</span>
          </div>
        </div>
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Can prefills be fully overlapped?</div>
          <div class="text-base" style="font-family: var(--font-math)">
            Capacity per cycle: {available} × {config.outputSeqLength} = {formatNumber(available * config.outputSeqLength, 0)} tokens
            <br />
            Need per cycle: {config.concurrentUsers} × {config.inputSeqLength} = {formatNumber(config.concurrentUsers * config.inputSeqLength, 0)} tokens
            <br />
            <span class={`font-bold ${canOverlap ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}>
              {canOverlap ? 'Yes — prefills fully overlap with decodes' : 'No — prefills will add overhead'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
