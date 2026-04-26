import { scoreColor } from '@/lib/utils'

interface ScoreBarProps {
  score: number
  showLabel?: boolean
  height?: number
}

export function ScoreBar({ score, showLabel = true, height = 6 }: ScoreBarProps) {
  const color = scoreColor(score)
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 rounded-full overflow-hidden" style={{ height, backgroundColor: 'var(--border)' }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      {showLabel && (
        <span className="text-xs font-mono tabular-nums w-10 text-right" style={{ color }}>
          {(score).toFixed(2)}
        </span>
      )}
    </div>
  )
}
