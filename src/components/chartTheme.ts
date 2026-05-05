import { useEffect, useState } from 'preact/hooks'
import { Chart } from 'chart.js'

export interface ChartTheme {
  isDark: boolean
  foreground: string
  mutedForeground: string
  grid: string
  unusedFill: string
  buttonActiveBg: string
  buttonActiveText: string
  buttonInactiveText: string
  fontFamily: string
}

const FONT_FAMILY = "'Source Sans 3', sans-serif"

function readTheme(): ChartTheme {
  const isDark =
    typeof document !== 'undefined' &&
    document.documentElement.classList.contains('dark')
  if (isDark) {
    return {
      isDark: true,
      foreground: '#fafafa',
      mutedForeground: 'rgba(250, 250, 250, 0.65)',
      grid: 'rgba(250, 250, 250, 0.12)',
      unusedFill: 'rgba(250, 250, 250, 0.1)',
      buttonActiveBg: 'rgba(250, 250, 250, 0.12)',
      buttonActiveText: '#fafafa',
      buttonInactiveText: 'rgba(250, 250, 250, 0.45)',
      fontFamily: FONT_FAMILY,
    }
  }
  return {
    isDark: false,
    foreground: '#0c0c0c',
    mutedForeground: 'rgba(12, 12, 12, 0.65)',
    grid: 'rgba(12, 12, 12, 0.1)',
    unusedFill: '#e5e7eb',
    buttonActiveBg: 'rgba(12, 12, 12, 0.08)',
    buttonActiveText: '#0c0c0c',
    buttonInactiveText: 'rgba(12, 12, 12, 0.4)',
    fontFamily: FONT_FAMILY,
  }
}

export function useChartTheme(): ChartTheme {
  const [theme, setTheme] = useState<ChartTheme>(readTheme)

  useEffect(() => {
    const update = () => setTheme(readTheme())
    const observer = new MutationObserver(update)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    })
    update()
    return () => observer.disconnect()
  }, [])

  return theme
}

export function applyChartDefaults(theme: ChartTheme) {
  Chart.defaults.font.family = theme.fontFamily
  Chart.defaults.font.size = 13
  Chart.defaults.color = theme.foreground
  Chart.defaults.borderColor = theme.grid
}
