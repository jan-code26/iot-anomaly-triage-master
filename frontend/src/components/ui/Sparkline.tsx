interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
}

export function Sparkline({ data, width = 80, height = 28, color = 'var(--accent)' }: SparklineProps) {
  if (!data.length) return null
  const max = Math.max(...data, 0.01)
  const min = Math.min(...data)
  const range = max - min || 0.01
  const pad = 2
  const w = width - pad * 2
  const h = height - pad * 2

  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * w
    const y = pad + h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}
