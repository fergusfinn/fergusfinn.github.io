import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'
import { useChartTheme, applyChartDefaults } from '../chartTheme'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
}

// Global expert popularity: the share of routed tokens each expert receives,
// ranked most -> least popular and averaged over the 40 layers. The router is
// far from balanced -- the busiest expert pulls ~8x its fair share, the curve
// crosses uniform around rank ~64, and the long tail is nearly dead. That skew
// is exactly why the "batch of N" null in the run-vs-batch chart sits below the
// uniform coupon-collector ceiling: a real batch draws these popular experts
// again and again, so it touches fewer distinct experts than uniform predicts.
// Qwen3.6-35B-A3B, SPEED-Bench qualitative, 11264 generated positions, k=8 of E=256.

const UNIFORM = 0.391 // 1/256, each expert's fair share of routed tokens, in %
const POP = [3.284, 2.770, 2.524, 2.281, 2.114, 1.964, 1.857, 1.744, 1.673, 1.591, 1.524, 1.446, 1.379, 1.327, 1.266, 1.228, 1.192, 1.159, 1.113, 1.080, 1.049, 1.016, 0.991, 0.963, 0.934, 0.917, 0.897, 0.878, 0.863, 0.842, 0.823, 0.809, 0.796, 0.784, 0.771, 0.760, 0.748, 0.738, 0.725, 0.711, 0.697, 0.687, 0.678, 0.668, 0.658, 0.646, 0.633, 0.621, 0.613, 0.604, 0.597, 0.588, 0.581, 0.571, 0.564, 0.557, 0.550, 0.543, 0.533, 0.524, 0.517, 0.512, 0.507, 0.500, 0.493, 0.488, 0.483, 0.476, 0.468, 0.462, 0.457, 0.449, 0.445, 0.439, 0.433, 0.426, 0.417, 0.412, 0.407, 0.402, 0.397, 0.393, 0.389, 0.383, 0.379, 0.373, 0.368, 0.363, 0.359, 0.354, 0.350, 0.346, 0.342, 0.337, 0.333, 0.330, 0.327, 0.322, 0.319, 0.314, 0.311, 0.307, 0.304, 0.301, 0.297, 0.293, 0.289, 0.285, 0.281, 0.279, 0.276, 0.272, 0.269, 0.266, 0.263, 0.259, 0.255, 0.252, 0.249, 0.246, 0.243, 0.240, 0.237, 0.234, 0.231, 0.228, 0.225, 0.222, 0.219, 0.217, 0.215, 0.212, 0.209, 0.206, 0.203, 0.201, 0.198, 0.194, 0.192, 0.189, 0.187, 0.185, 0.183, 0.181, 0.179, 0.177, 0.175, 0.172, 0.170, 0.168, 0.166, 0.164, 0.162, 0.160, 0.158, 0.157, 0.155, 0.152, 0.151, 0.148, 0.146, 0.144, 0.143, 0.141, 0.139, 0.136, 0.135, 0.133, 0.131, 0.129, 0.127, 0.126, 0.124, 0.122, 0.121, 0.119, 0.118, 0.116, 0.114, 0.111, 0.110, 0.108, 0.106, 0.105, 0.103, 0.102, 0.100, 0.099, 0.097, 0.095, 0.094, 0.093, 0.091, 0.089, 0.088, 0.087, 0.085, 0.084, 0.082, 0.081, 0.079, 0.078, 0.077, 0.075, 0.073, 0.072, 0.071, 0.070, 0.068, 0.067, 0.066, 0.064, 0.063, 0.062, 0.061, 0.059, 0.058, 0.057, 0.055, 0.054, 0.053, 0.052, 0.050, 0.049, 0.048, 0.046, 0.045, 0.044, 0.042, 0.041, 0.040, 0.038, 0.037, 0.036, 0.034, 0.033, 0.032, 0.030, 0.029, 0.028, 0.026, 0.025, 0.024, 0.023, 0.021, 0.020, 0.019, 0.018, 0.017, 0.015, 0.013, 0.012, 0.010, 0.009, 0.007, 0.005]

export default function ExpertPopularity() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)
  const theme = useChartTheme()

  useEffect(() => {
    if (!chartRef.current) return
    chartInstance.current?.destroy()
    applyChartDefaults(theme)

    const aboveC = `hsl(220, 70%, ${theme.isDark ? 60 : 48}%)` // pulls more than its share
    const belowC = `hsl(220, 28%, ${theme.isDark ? 42 : 78}%)` // starved tail

    const axisTitle = (text: string) => ({
      display: true,
      text,
      color: theme.mutedForeground,
      font: { family: theme.fontFamily },
    })
    const tickStyle = { color: theme.mutedForeground, font: { family: theme.fontFamily } }
    const showTicks = [0, 32, 64, 128, 192, 255]

    chartInstance.current = new Chart(chartRef.current, {
      type: 'bar',
      data: {
        labels: POP.map((_, i) => i),
        datasets: [
          {
            label: 'share of routed tokens',
            data: POP,
            backgroundColor: POP.map((v) => (v >= UNIFORM ? aboveC : belowC)),
            borderWidth: 0,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 10, right: 12 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => (items.length ? `expert rank ${items[0].label}` : ''),
              label: (ctx) =>
                `  ${(ctx.parsed.y as number).toFixed(2)}%  (${((ctx.parsed.y as number) / UNIFORM).toFixed(1)}x uniform)`,
            },
          },
        },
        scales: {
          x: {
            title: axisTitle('expert rank (most -> least popular, of 256)'),
            grid: { display: false },
            ticks: {
              ...tickStyle,
              autoSkip: false,
              maxRotation: 0,
              callback: (_v, i) => (showTicks.includes(i) ? `${i}` : ''),
            },
          },
          y: {
            min: 0,
            title: axisTitle('share of routed tokens (%)'),
            grid: { color: theme.grid },
            ticks: { ...tickStyle, callback: (v) => `${v}%` },
          },
        },
      },
      plugins: [
        {
          // dashed uniform reference line (each expert's fair share)
          id: 'uniform',
          afterDatasetsDraw(chart) {
            const { ctx, scales, chartArea } = chart
            const y = scales.y.getPixelForValue(UNIFORM)
            ctx.save()
            ctx.strokeStyle = theme.mutedForeground
            ctx.lineWidth = 1
            ctx.setLineDash([5, 4])
            ctx.beginPath()
            ctx.moveTo(chartArea.left, y)
            ctx.lineTo(chartArea.right, y)
            ctx.stroke()
            ctx.setLineDash([])
            ctx.font = `11px ${theme.fontFamily}`
            ctx.fillStyle = theme.mutedForeground
            ctx.textAlign = 'right'
            ctx.fillText('uniform', chartArea.right - 4, y - 5)
            ctx.restore()
          },
        },
      ],
    })

    return () => {
      chartInstance.current?.destroy()
      chartInstance.current = null
    }
  }, [theme])

  return (
    <div class="my-6" style="position: relative; height: 360px; width: 100%;">
      <canvas ref={chartRef} />
    </div>
  )
}
