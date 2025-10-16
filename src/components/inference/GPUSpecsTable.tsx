import { useStore } from '@nanostores/preact'
import { acceleratorSpec } from '../../stores/inferenceStore'

export default function GPUSpecsTable() {
  const spec = useStore(acceleratorSpec)

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
            <td class="py-3 px-4">Compute (FP8)</td>
            <td class="py-3 px-4">{spec.compute} TFLOPS</td>
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
