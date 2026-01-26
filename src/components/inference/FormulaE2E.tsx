import { useStore } from '@nanostores/preact'
import {
  configStore,
  prefillTime,
  decodeTime,
  e2eLatency,
  totalTime,
  totalOutputTokens,
} from '../../stores/inferenceStore'

export default function FormulaE2E() {
  const config = useStore(configStore)
  const tPrefill = useStore(prefillTime)
  const tDecode = useStore(decodeTime)
  const latency = useStore(e2eLatency)
  const batchTime = useStore(totalTime)
  const outputTokens = useStore(totalOutputTokens)

  const B = config.concurrentUsers
  const OSL = config.outputSeqLength
  const throughput = (outputTokens / batchTime) * 1000 // tokens/s

  const formatMs = (ms: number) => ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`

  return (
    <div class="space-y-10">
      <div class="space-y-4">
        <p class="text-xl leading-relaxed">End-to-end latency for a single request:</p>
        <div class="text-center space-y-3">
          <p class="text-2xl font-mono">
            <span class="note" data-note="Time to process the input prompt">t_prefill</span>
            {' '}+{' '}
            <span class="note" data-note="Output sequence length">OSL</span>
            {' '}&times;{' '}
            <span class="note" data-note="Time per generated token (average)">t_decode</span>
          </p>
          <p class="text-2xl font-mono">
            = {tPrefill.toFixed(1)} ms + {OSL} &times; {tDecode.toFixed(2)} ms
          </p>
          <p class="text-2xl font-mono">
            = <strong>{formatMs(latency)}</strong>
          </p>
        </div>
      </div>
      <div class="space-y-4">
        <p class="text-xl leading-relaxed">
          Throughput for{' '}
          <span class="note" data-note="Concurrent users (batch size)">B</span>
          {' '}= {B} concurrent users:
        </p>
        <div class="text-center space-y-3">
          <p class="text-2xl font-mono">
            (<span class="note" data-note="Concurrent users">B</span>
            {' '}&times;{' '}
            <span class="note" data-note="Output sequence length">OSL</span>)
            {' '}&divide;{' '}
            (<span class="note" data-note="Concurrent users">B</span>
            {' '}&times; t_prefill +{' '}
            <span class="note" data-note="Output sequence length">OSL</span>
            {' '}&times; t_decode)
          </p>
          <p class="text-2xl font-mono">
            = ({B} &times; {OSL}) &divide; ({B} &times; {tPrefill.toFixed(1)} ms + {OSL} &times; {tDecode.toFixed(2)} ms)
          </p>
          <p class="text-2xl font-mono">
            = <strong>{throughput.toFixed(0)} tok/s</strong>
          </p>
        </div>
      </div>
    </div>
  )
}
