import { useStore } from '@nanostores/preact'
import { useState } from 'preact/hooks'
import { configStore, ACCELERATORS } from '../../stores/inferenceStore'

interface ConfigTableProps {
  compact?: boolean
}

export default function ConfigTable({ compact = false }: ConfigTableProps) {
  const config = useStore(configStore)
  const [seqLengthMode, setSeqLengthMode] = useState('1024/1024')

  const updateConfig = (key: keyof typeof config, value: number | string) => {
    configStore.setKey(key, value as any)
  }

  // Get supported precisions for current accelerator
  const getSupportedPrecisions = () => {
    const accelerator = ACCELERATORS[config.acceleratorType]
    const precisions = []

    if (accelerator.computeFP4) {
      precisions.push({ value: 0.5, label: 'FP4' })
    }
    precisions.push({ value: 1, label: 'FP8' })
    precisions.push({ value: 2, label: 'FP16/BF16' })

    return precisions
  }

  const supportedPrecisions = getSupportedPrecisions()

  // If current precision is not supported by new accelerator, switch to FP8
  const currentPrecisionSupported = supportedPrecisions.some(p => p.value === config.bytesPerParameter)
  if (!currentPrecisionSupported && config.bytesPerParameter === 0.5) {
    updateConfig('bytesPerParameter', 1) // Switch to FP8
  }

  const handleSeqLengthModeChange = (mode: string) => {
    setSeqLengthMode(mode)
    if (mode === '1024/1024') {
      updateConfig('inputSeqLength', 1024)
      updateConfig('outputSeqLength', 1024)
    } else if (mode === '1024/8192') {
      updateConfig('inputSeqLength', 1024)
      updateConfig('outputSeqLength', 8192)
    } else if (mode === '8192/1024') {
      updateConfig('inputSeqLength', 8192)
      updateConfig('outputSeqLength', 1024)
    }
  }

  return (
    <div class={compact ? 'space-y-3' : 'overflow-x-auto my-6'}>
      {compact ? (
        <>
          <div>
            <label class="block text-xs font-semibold mb-1">Tensor Parallelism</label>
            <select
              value={config.tensorParallelism}
              onChange={(e) =>
                updateConfig('tensorParallelism', parseInt((e.target as HTMLSelectElement).value))
              }
              class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
              onChange={(e) =>
                updateConfig('concurrentUsers', parseInt((e.target as HTMLSelectElement).value))
              }
              class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            <label class="block text-xs font-semibold mb-1">Sequence Length</label>
            <select
              value={seqLengthMode}
              onChange={(e) => handleSeqLengthModeChange((e.target as HTMLSelectElement).value)}
              class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="1024/1024">1k/1k</option>
              <option value="1024/8192">1k/8k</option>
              <option value="8192/1024">8k/1k</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          {seqLengthMode === 'custom' && (
            <>
              <div>
                <label class="block text-xs font-semibold mb-1">Input Seq Length</label>
                <input
                  type="number"
                  min="1"
                  max="32768"
                  step="1"
                  value={config.inputSeqLength}
                  onInput={(e) =>
                    updateConfig('inputSeqLength', parseInt((e.target as HTMLInputElement).value))
                  }
                  class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label class="block text-xs font-semibold mb-1">Output Seq Length</label>
                <input
                  type="number"
                  min="1"
                  max="32768"
                  step="1"
                  value={config.outputSeqLength}
                  onInput={(e) =>
                    updateConfig('outputSeqLength', parseInt((e.target as HTMLInputElement).value))
                  }
                  class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </>
          )}
          <div>
            <label class="block text-xs font-semibold mb-1">Accelerator</label>
            <select
              value={config.acceleratorType}
              onChange={(e) =>
                updateConfig('acceleratorType', (e.target as HTMLSelectElement).value)
              }
              class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {Object.entries(ACCELERATORS).map(([key, spec]) => (
                <option key={key} value={key}>
                  {spec.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label class="block text-xs font-semibold mb-1">Precision</label>
            <select
              value={config.bytesPerParameter}
              onChange={(e) =>
                updateConfig('bytesPerParameter', parseFloat((e.target as HTMLSelectElement).value))
              }
              class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {supportedPrecisions.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>
        </>
      ) : (
        <table class="min-w-full border-collapse">
          <thead>
            <tr class="border-b border-gray-300 dark:border-gray-700">
              <th class="text-left py-2 px-4 font-semibold">Parameter</th>
              <th class="text-left py-2 px-4 font-semibold">Value</th>
            </tr>
          </thead>
          <tbody>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Tensor parallelism</td>
            <td class="py-3 px-4">
              <select
                value={config.tensorParallelism}
                onChange={(e) =>
                  updateConfig('tensorParallelism', parseInt((e.target as HTMLSelectElement).value))
                }
                class="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={4}>4</option>
                <option value={8}>8</option>
              </select>
            </td>
          </tr>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Concurrent users</td>
            <td class="py-3 px-4">
              <select
                value={config.concurrentUsers}
                onChange={(e) =>
                  updateConfig('concurrentUsers', parseInt((e.target as HTMLSelectElement).value))
                }
                class="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            </td>
          </tr>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">ISL/OSL</td>
            <td class="py-3 px-4">
              <div class="flex items-center gap-2">
                <select
                  value={seqLengthMode}
                  onChange={(e) => handleSeqLengthModeChange((e.target as HTMLSelectElement).value)}
                  class="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="1024/1024">1k/1k</option>
                  <option value="1024/8192">1k/8k</option>
                  <option value="8192/1024">8k/1k</option>
                  <option value="custom">Custom</option>
                </select>
                {seqLengthMode === 'custom' && (
                  <>
                    <input
                      type="number"
                      min="1"
                      max="32768"
                      step="1"
                      value={config.inputSeqLength}
                      onInput={(e) =>
                        updateConfig('inputSeqLength', parseInt((e.target as HTMLInputElement).value))
                      }
                      class="w-24 px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <span>/</span>
                    <input
                      type="number"
                      min="1"
                      max="32768"
                      step="1"
                      value={config.outputSeqLength}
                      onInput={(e) =>
                        updateConfig('outputSeqLength', parseInt((e.target as HTMLInputElement).value))
                      }
                      class="w-24 px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </>
                )}
              </div>
            </td>
          </tr>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Accelerator</td>
            <td class="py-3 px-4">
              <select
                value={config.acceleratorType}
                onChange={(e) =>
                  updateConfig('acceleratorType', (e.target as HTMLSelectElement).value)
                }
                class="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {Object.entries(ACCELERATORS).map(([key, spec]) => (
                  <option key={key} value={key}>
                    {spec.name}
                  </option>
                ))}
              </select>
            </td>
          </tr>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Precision</td>
            <td class="py-3 px-4">
              <select
                value={config.bytesPerParameter}
                onChange={(e) =>
                  updateConfig('bytesPerParameter', parseFloat((e.target as HTMLSelectElement).value))
                }
                class="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {supportedPrecisions.map(p => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </td>
          </tr>
        </tbody>
      </table>
      )}
    </div>
  )
}
