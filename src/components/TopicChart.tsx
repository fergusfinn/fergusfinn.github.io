import { useEffect, useRef } from 'preact/hooks'
import { Chart, registerables } from 'chart.js'

if (typeof window !== 'undefined') {
  Chart.register(...registerables)
  Chart.defaults.font.family = "'Source Sans 3', sans-serif"
  Chart.defaults.font.size = 13
}

const TOPIC_DATA: Record<string, Record<string, number>> = {
  "1870s": { "economy": 15.4, "defense": 7.5, "civil-rights": 10.8, "healthcare": 0.7, "education": 0.4, "immigration": 0.4, "environment": 0.7, "judiciary": 49.5, "infrastructure": 12.5, "foreign-policy": 2.2 },
  "1880s": { "economy": 21.8, "defense": 7.5, "civil-rights": 7.3, "healthcare": 0.8, "education": 1.8, "immigration": 1.5, "environment": 1.0, "judiciary": 36.3, "infrastructure": 18.0, "foreign-policy": 4.0 },
  "1890s": { "economy": 25.0, "defense": 7.8, "civil-rights": 9.0, "healthcare": 1.5, "education": 0.8, "immigration": 1.0, "environment": 1.0, "judiciary": 32.0, "infrastructure": 16.5, "foreign-policy": 5.5 },
  "1900s": { "economy": 18.3, "defense": 8.8, "civil-rights": 8.8, "healthcare": 1.3, "education": 1.5, "immigration": 2.5, "environment": 4.8, "judiciary": 25.8, "infrastructure": 20.1, "foreign-policy": 8.3 },
  "1910s": { "economy": 23.3, "defense": 12.0, "civil-rights": 8.8, "healthcare": 2.5, "education": 2.0, "immigration": 1.0, "environment": 1.3, "judiciary": 25.1, "infrastructure": 17.5, "foreign-policy": 6.5 },
  "1920s": { "economy": 28.7, "defense": 9.8, "civil-rights": 6.5, "healthcare": 1.3, "education": 1.3, "immigration": 2.3, "environment": 2.3, "judiciary": 26.7, "infrastructure": 12.8, "foreign-policy": 8.3 },
  "1930s": { "economy": 35.4, "defense": 4.8, "civil-rights": 7.8, "healthcare": 1.8, "education": 1.8, "immigration": 1.0, "environment": 3.0, "judiciary": 20.5, "infrastructure": 12.9, "foreign-policy": 10.9 },
  "1940s": { "economy": 23.4, "defense": 18.6, "civil-rights": 8.1, "healthcare": 2.3, "education": 1.8, "immigration": 1.8, "environment": 1.8, "judiciary": 18.8, "infrastructure": 9.7, "foreign-policy": 13.7 },
  "1950s": { "economy": 19.0, "defense": 10.4, "civil-rights": 11.2, "healthcare": 4.3, "education": 4.1, "immigration": 2.0, "environment": 2.5, "judiciary": 14.2, "infrastructure": 15.0, "foreign-policy": 17.3 },
  "1960s": { "economy": 19.2, "defense": 10.1, "civil-rights": 15.9, "healthcare": 4.3, "education": 4.1, "immigration": 1.0, "environment": 3.3, "judiciary": 13.7, "infrastructure": 8.4, "foreign-policy": 20.0 },
  "1970s": { "economy": 23.3, "defense": 6.4, "civil-rights": 11.8, "healthcare": 5.6, "education": 3.6, "immigration": 1.5, "environment": 6.2, "judiciary": 18.2, "infrastructure": 7.9, "foreign-policy": 15.4 },
  "1980s": { "economy": 21.3, "defense": 6.9, "civil-rights": 14.1, "healthcare": 5.9, "education": 3.6, "immigration": 2.1, "environment": 7.2, "judiciary": 14.4, "infrastructure": 6.4, "foreign-policy": 18.0 },
  "1990s": { "economy": 18.3, "defense": 10.9, "civil-rights": 20.1, "healthcare": 8.9, "education": 4.3, "immigration": 0.8, "environment": 7.6, "judiciary": 10.4, "infrastructure": 8.1, "foreign-policy": 10.7 },
  "2000s": { "economy": 16.3, "defense": 10.7, "civil-rights": 14.5, "healthcare": 13.0, "education": 7.4, "immigration": 2.0, "environment": 6.9, "judiciary": 9.2, "infrastructure": 8.4, "foreign-policy": 11.5 },
}

const TOPICS: { key: string; label: string; color: string }[] = [
  { key: 'economy', label: 'Economy', color: '#2563eb' },
  { key: 'judiciary', label: 'Judiciary', color: '#7c3aed' },
  { key: 'infrastructure', label: 'Infrastructure', color: '#64748b' },
  { key: 'foreign-policy', label: 'Foreign Policy', color: '#dc2626' },
  { key: 'civil-rights', label: 'Civil Rights', color: '#f59e0b' },
  { key: 'defense', label: 'Defense', color: '#059669' },
  { key: 'healthcare', label: 'Healthcare', color: '#ec4899' },
  { key: 'environment', label: 'Environment', color: '#10b981' },
  { key: 'education', label: 'Education', color: '#8b5cf6' },
  { key: 'immigration', label: 'Immigration', color: '#f97316' },
]

const labels = Object.keys(TOPIC_DATA)

export default function TopicChart() {
  const chartRef = useRef<HTMLCanvasElement>(null)
  const chartInstance = useRef<Chart | null>(null)

  useEffect(() => {
    if (!chartRef.current) return

    if (chartInstance.current) {
      chartInstance.current.destroy()
    }

    const ctx = chartRef.current.getContext('2d')
    if (!ctx) return

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: TOPICS.map(({ key, label, color }) => ({
          label,
          data: labels.map(l => TOPIC_DATA[l][key] ?? 0),
          borderColor: color,
          backgroundColor: color + '18',
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              usePointStyle: true,
              padding: 16,
              font: { size: 12 },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { font: { size: 11 } },
          },
          y: {
            stacked: false,
            min: 0,
            max: 55,
            ticks: {
              callback: (v) => `${v}%`,
              font: { size: 11 },
            },
            title: {
              display: true,
              text: '% of speeches',
              font: { size: 12 },
            },
          },
        },
      },
    })

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy()
      }
    }
  }, [])

  return (
    <div style={{ position: 'relative', width: '100%', height: '420px' }}>
      <canvas ref={chartRef} />
    </div>
  )
}
