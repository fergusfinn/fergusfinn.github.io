import { useStore } from '@nanostores/preact'
import { acceleratorSpec, configStore } from '../../stores/inferenceStore'

export default function GPUSpecsTable() {
  const spec = useStore(acceleratorSpec)
  const config = useStore(configStore)

  // Get current compute and precision label based on selected bytes per parameter
  const getCurrentCompute = () => {
    if (config.bytesPerParameter === 0.5) {
      return { compute: spec.computeFP4, precision: 'FP4' }
    } else if (config.bytesPerParameter === 1) {
      return { compute: spec.computeFP8, precision: 'FP8' }
    } else if (config.bytesPerParameter === 2) {
      return { compute: spec.computeFP16, precision: 'FP16/BF16' }
    }
    // Default to FP8
    return { compute: spec.computeFP8, precision: 'FP8' }
  }

  const { compute, precision } = getCurrentCompute()

  return (
    <div class="overflow-x-auto my-6">
      <table class="min-w-full border-collapse">
        <thead>
          <tr class="border-b border-gray-300 dark:border-gray-700">
            <th colSpan={2} class="text-left py-2 px-4 font-semibold">{spec.name}</th>
          </tr>
        </thead>
        <tbody>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Compute ({precision})</td>
            <td class="py-3 px-4">{compute} TFLOPS</td>
          </tr>
          <tr class="border-b border-gray-200 dark:border-gray-800">
            <td class="py-3 px-4">Memory bandwidth</td>
            <td class="py-3 px-4">{spec.memoryBandwidth} TB/s</td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
