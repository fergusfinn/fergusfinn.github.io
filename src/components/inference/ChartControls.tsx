import { useStore } from '@nanostores/preact'
import { map } from 'nanostores'

type XAxisVariable = 'concurrency' | 'tensorParallelism'

export interface ChartControlsState {
  throughputXAxis: XAxisVariable
  latencyXAxis: XAxisVariable
}

export const chartControlsStore = map<ChartControlsState>({
  throughputXAxis: 'concurrency',
  latencyXAxis: 'concurrency',
})

export default function ChartControls() {
  const controls = useStore(chartControlsStore)

  return (
    <div class="sidebar-content">
      <h3 class="sidebar-title">Chart Controls</h3>
      <div class="space-y-3">
        <div>
          <label class="block text-xs font-semibold mb-1">Throughput X-axis</label>
          <select
            value={controls.throughputXAxis}
            onChange={(e) =>
              chartControlsStore.setKey('throughputXAxis', (e.target as HTMLSelectElement).value as XAxisVariable)
            }
            class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="concurrency">Concurrent Users</option>
            <option value="tensorParallelism">Tensor Parallelism</option>
          </select>
        </div>
        <div>
          <label class="block text-xs font-semibold mb-1">Latency X-axis</label>
          <select
            value={controls.latencyXAxis}
            onChange={(e) =>
              chartControlsStore.setKey('latencyXAxis', (e.target as HTMLSelectElement).value as XAxisVariable)
            }
            class="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="concurrency">Concurrent Users</option>
            <option value="tensorParallelism">Tensor Parallelism</option>
          </select>
        </div>
      </div>
    </div>
  )
}
