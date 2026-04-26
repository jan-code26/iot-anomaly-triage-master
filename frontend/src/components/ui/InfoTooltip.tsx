import { HelpCircle } from 'lucide-react'

interface Props { text: string; side?: 'top' | 'bottom' }

export function InfoTooltip({ text, side = 'bottom' }: Props) {
  const posClass = side === 'bottom'
    ? 'top-full left-1/2 -translate-x-1/2 mt-2'
    : 'bottom-full left-1/2 -translate-x-1/2 mb-2'

  return (
    <span className="relative inline-flex group">
      <HelpCircle
        size={12}
        className="cursor-help ml-1 shrink-0"
        style={{ color: 'var(--text-muted)' }}
      />
      <span
        className={`pointer-events-none absolute ${posClass} w-52 rounded-md px-2.5 py-1.5 text-xs leading-snug opacity-0 group-hover:opacity-100 transition-opacity z-50`}
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
        }}
      >
        {text}
      </span>
    </span>
  )
}
