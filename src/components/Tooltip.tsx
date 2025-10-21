import { useState } from 'preact/hooks'

interface TooltipProps {
  label: string
  children: preact.ComponentChildren
}

export default function Tooltip({ label, children }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  return (
    <span class="relative inline-block">
      <span
        class="border-b border-gray-400 dark:border-gray-500"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        {children}
      </span>
      {isVisible && (
        <span class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs whitespace-nowrap bg-gray-800 dark:bg-gray-700 text-white rounded shadow-lg z-10 pointer-events-none">
          {label}
          <span class="absolute top-full left-1/2 -translate-x-1/2 -mt-px border-4 border-transparent border-t-gray-800 dark:border-t-gray-700"></span>
        </span>
      )}
    </span>
  )
}
