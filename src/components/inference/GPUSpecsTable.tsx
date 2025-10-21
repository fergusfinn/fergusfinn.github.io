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

  // Get datasheet URL based on accelerator type
  const getDatasheetUrl = () => {
    const urls: Record<string, string> = {
      H100: 'https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306',
      H200: 'https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200',
      B200: 'https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf',
      MI325X: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf',
      MI300X: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf',
      MI355X: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf',
    }
    return urls[config.acceleratorType] || ''
  }

  const { compute, precision } = getCurrentCompute()
  const datasheetUrl = getDatasheetUrl()

  return (
    <div class="overflow-x-auto my-6">
      <table class="min-w-full border-collapse">
        <thead>
          <tr class="border-b border-gray-300 dark:border-gray-700">
            <th colSpan={2} class="text-left py-2 px-4 font-semibold">
              {datasheetUrl ? (
                <a
                  href={datasheetUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="hover:underline underline-offset-4 text-primary dark:text-primary-dark"
                >
                  {spec.name}
                </a>
              ) : (
                spec.name
              )}
            </th>
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
