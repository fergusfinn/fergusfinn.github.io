import { useState } from 'preact/hooks'
import { useStore } from '@nanostores/preact'
import { configStore, ACCELERATORS } from '../../stores/inferenceStore'

export default function SlideConfig() {
  const config = useStore(configStore)
  const [open, setOpen] = useState(false)
  const [islText, setIslText] = useState(String(config.inputSeqLength))
  const [oslText, setOslText] = useState(String(config.outputSeqLength))
  const [islError, setIslError] = useState(false)
  const [oslError, setOslError] = useState(false)

  const commitNumericField = (
    text: string,
    key: 'inputSeqLength' | 'outputSeqLength',
    setError: (v: boolean) => void
  ) => {
    const n = parseInt(text, 10)
    if (isNaN(n) || n < 1) {
      setError(true)
    } else {
      setError(false)
      configStore.setKey(key, n)
    }
  }

  const updateConfig = (key: keyof typeof config, value: number | string) => {
    configStore.setKey(key, value as any)
  }

  const getSupportedPrecisions = () => {
    const accelerator = ACCELERATORS[config.acceleratorType]
    const precisions = []
    if (accelerator.computeFP4) {
      precisions.push({ value: 0.5, label: 'FP4' })
    }
    precisions.push({ value: 1, label: 'FP8' })
    return precisions
  }

  const supportedPrecisions = getSupportedPrecisions()
  const currentPrecisionSupported = supportedPrecisions.some(p => p.value === config.bytesPerParameter)
  if (!currentPrecisionSupported && config.bytesPerParameter === 0.5) {
    updateConfig('bytesPerParameter', 1)
  }

  const spec = ACCELERATORS[config.acceleratorType]
  const precisionLabel = config.bytesPerParameter === 0.5 ? 'FP4' : config.bytesPerParameter === 1 ? 'FP8' : 'FP16'

  return (
    <div class="relative">
      <button
        onClick={() => setOpen(!open)}
        class="text-sm px-3 py-1.5 border border-foreground/15 dark:border-foreground-dark/15 rounded hover:bg-foreground/5 dark:hover:bg-foreground-dark/5 transition-colors"
      >
        Llama 70B &middot; {config.tensorParallelism}x {spec.name} &middot; {precisionLabel}
      </button>

      {open && (
        <div class="absolute top-full right-0 mt-2 w-64 p-4 rounded-lg border border-foreground/10 dark:border-foreground-dark/10 bg-background dark:bg-background-dark shadow-lg z-30 space-y-3">
          <div>
            <label class="block text-xs font-semibold mb-1">Accelerator</label>
            <select
              value={config.acceleratorType}
              onChange={(e) => updateConfig('acceleratorType', (e.target as HTMLSelectElement).value)}
              class="w-full px-2 py-1 text-sm border border-foreground/15 dark:border-foreground-dark/15 rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-primary-dark"
            >
              {Object.entries(ACCELERATORS).map(([key, s]) => (
                <option key={key} value={key}>{s.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label class="block text-xs font-semibold mb-1">Precision</label>
            <select
              value={config.bytesPerParameter}
              onChange={(e) => updateConfig('bytesPerParameter', parseFloat((e.target as HTMLSelectElement).value))}
              class="w-full px-2 py-1 text-sm border border-foreground/15 dark:border-foreground-dark/15 rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-primary-dark"
            >
              {supportedPrecisions.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label class="block text-xs font-semibold mb-1">Tensor Parallelism</label>
            <select
              value={config.tensorParallelism}
              onChange={(e) => updateConfig('tensorParallelism', parseInt((e.target as HTMLSelectElement).value))}
              class="w-full px-2 py-1 text-sm border border-foreground/15 dark:border-foreground-dark/15 rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-primary-dark"
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-semibold mb-1">Concurrent Users</label>
            <select
              value={config.concurrentUsers}
              onChange={(e) => updateConfig('concurrentUsers', parseInt((e.target as HTMLSelectElement).value))}
              class="w-full px-2 py-1 text-sm border border-foreground/15 dark:border-foreground-dark/15 rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 focus:ring-primary dark:focus:ring-primary-dark"
            >
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
              <option value={8}>8</option>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
              <option value={128}>128</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-semibold mb-1">Input / Output Tokens</label>
            <div class="flex items-center gap-2">
              <input
                type="text"
                inputMode="numeric"
                value={islText}
                onInput={(e) => {
                  const val = (e.target as HTMLInputElement).value
                  setIslText(val)
                  commitNumericField(val, 'inputSeqLength', setIslError)
                }}
                onBlur={() => {
                  if (islError) {
                    setIslText(String(config.inputSeqLength))
                    setIslError(false)
                  }
                }}
                class={`w-full px-2 py-1 text-sm border rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 ${
                  islError
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-foreground/15 dark:border-foreground-dark/15 focus:ring-primary dark:focus:ring-primary-dark'
                }`}
              />
              <span class="text-sm">/</span>
              <input
                type="text"
                inputMode="numeric"
                value={oslText}
                onInput={(e) => {
                  const val = (e.target as HTMLInputElement).value
                  setOslText(val)
                  commitNumericField(val, 'outputSeqLength', setOslError)
                }}
                onBlur={() => {
                  if (oslError) {
                    setOslText(String(config.outputSeqLength))
                    setOslError(false)
                  }
                }}
                class={`w-full px-2 py-1 text-sm border rounded bg-background dark:bg-background-dark focus:outline-none focus:ring-2 ${
                  oslError
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-foreground/15 dark:border-foreground-dark/15 focus:ring-primary dark:focus:ring-primary-dark'
                }`}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
