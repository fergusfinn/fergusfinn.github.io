import { useStore } from '@nanostores/preact'
import {
  prefillTime,
  formatNumber,
} from '../../stores/inferenceStore'

export default function TimeToFirstToken() {
  const ttft = useStore(prefillTime)

  return (
    <div class="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
      <div class="space-y-3">
        <div class="leading-relaxed">
          <div class="text-gray-600 dark:text-gray-400 mb-1 text-sm">Time to first token:</div>
          <div class="text-base" style="font-family: var(--font-math)">
            <span class="font-bold text-blue-700 dark:text-blue-400">
              {formatNumber(ttft)} ms
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
