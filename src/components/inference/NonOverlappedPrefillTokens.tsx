import { useStore } from '@nanostores/preact'
import {
  nonOverlappedPrefillTokens,
  configStore,
  computeBoundThreshold,
  formatLargeNumber,
  formatNumber,
} from '../../stores/inferenceStore'
import Tooltip from '../Tooltip'

export default function NonOverlappedPrefillTokens() {
  const tokens = useStore(nonOverlappedPrefillTokens)
  const config = useStore(configStore)
  const threshold = useStore(computeBoundThreshold)

  const B = config.concurrentUsers
  const ISL = config.inputSeqLength
  const OSL = config.outputSeqLength

  return (
    <div class="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Non-overlapped prefill tokens:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <Tooltip label="concurrent_users">{B}</Tooltip> × (<Tooltip label="input_seq_len">{ISL}</Tooltip> + <Tooltip label="output_seq_len">{OSL}</Tooltip>) - <Tooltip label="output_seq_len">{OSL}</Tooltip> × <Tooltip label="compute_bound_threshold">{threshold}</Tooltip> ={' '}
            <span class="font-bold text-amber-700 dark:text-amber-400">{formatNumber(tokens, 0)} tokens</span>
          </div>
        </div>
        {tokens === 0 && (
          <div class="text-sm text-gray-600 dark:text-gray-400 italic">
            All prefills can be completely overlapped with decodes!
          </div>
        )}
      </div>
    </div>
  )
}
