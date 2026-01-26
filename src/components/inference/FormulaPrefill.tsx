import { useStore } from '@nanostores/preact'
import {
  configStore,
  modelStore,
  totalCompute,
  prefillTime,
  prefillFLOPs,
  attentionFLOPs,
} from '../../stores/inferenceStore'

export default function FormulaPrefill() {
  const config = useStore(configStore)
  const model = useStore(modelStore)
  const compute = useStore(totalCompute)
  const time = useStore(prefillTime)
  const flops = useStore(prefillFLOPs)
  const attention = useStore(attentionFLOPs)

  const ISL = config.inputSeqLength
  const P = model.modelSize
  const computeTFLOPS = (compute / 1e12).toFixed(0)

  // Format attention in scientific notation (e.g., "2.7×10¹²")
  const attentionExp = Math.floor(Math.log10(attention))
  const attentionMantissa = (attention / Math.pow(10, attentionExp)).toFixed(1)
  const superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
  const toSuperscript = (n: number) => String(n).split('').map(d => superscripts[parseInt(d)]).join('')
  const attentionSci = `${attentionMantissa}×10${toSuperscript(attentionExp)}`

  return (
    <div class="text-center space-y-6">
      <p class="text-2xl font-mono">
        (2 &times; <span class="note" data-note="Input sequence length">ISL</span>
        {' '}&times;{' '}
        <span class="note" data-note="Model parameters">P</span>
        {' '}+{' '}
        <span class="note" data-note="4 × ISL² × D × L">attention</span>)
        {' '}&divide;{' '}
        <span class="note" data-note="Total FLOP/s across all GPUs">compute</span>
      </p>
      <p class="text-2xl font-mono">
        = (2 &times; {ISL.toLocaleString()} &times; {P.toFixed(0)}B + {attentionSci}) &divide; {computeTFLOPS} TFLOP/s
      </p>
      <p class="text-2xl font-mono">
        = <strong>{time.toFixed(1)} ms</strong>
      </p>
    </div>
  )
}
