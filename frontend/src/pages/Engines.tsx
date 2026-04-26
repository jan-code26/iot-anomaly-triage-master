import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import type { Dataset } from '@/lib/types'
import { api } from '@/lib/api'
import { regimeName, relativeTime } from '@/lib/utils'
import { Badge } from '@/components/ui/Badge'
import { ScoreBar } from '@/components/ui/ScoreBar'

interface Props { dataset: Dataset }

export function Engines({ dataset }: Props) {
  const nav = useNavigate()
  const isMulti = dataset === 'FD002' || dataset === 'FD004'
  const { data: engines = [], isLoading } = useQuery({
    queryKey: ['engines', dataset],
    queryFn: () => api.getEngines(dataset),
  })

  return (
    <div className="p-5">
      <h2 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-muted)' }}>
        All Engines — {dataset} ({engines.length} total)
      </h2>
      {isLoading ? (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      ) : (
        <div className="rounded-xl border overflow-auto" style={{ borderColor: 'var(--border)' }}>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: 'var(--bg-sidebar)', borderBottom: '1px solid var(--border)' }}>
                {['Engine', isMulti ? 'Condition' : null, 'Score', 'Decision', 'Cycle', 'RUL', 'Alerts', 'Last Seen'].filter(Boolean).map(h => (
                  <th key={h!} className="text-left px-4 py-2.5 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {engines.map((eng, i) => (
                <tr
                  key={eng.engine_id}
                  onClick={() => nav(`/engines/${eng.engine_id}?dataset=${dataset}`)}
                  className="cursor-pointer hover:bg-[var(--border)] transition-colors border-b"
                  style={{ borderColor: 'var(--border)', background: i % 2 === 0 ? 'var(--bg-card)' : 'var(--bg)' }}
                >
                  <td className="px-4 py-2.5 font-mono font-bold text-xs" style={{ color: 'var(--text-primary)' }}>#{eng.engine_id}</td>
                  {isMulti && (
                    <td className="px-4 py-2.5">
                      <Badge variant="outline" className="text-[10px]">{regimeName(eng.regime)}</Badge>
                    </td>
                  )}
                  <td className="px-4 py-2.5 w-32">
                    <ScoreBar score={eng.latest_score} height={5} />
                  </td>
                  <td className="px-4 py-2.5">
                    <Badge variant={eng.latest_decision === 'ALERT' ? 'alert' : eng.latest_decision === 'UNCERTAIN' ? 'warn' : 'ok'}>
                      {eng.latest_decision}
                    </Badge>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{eng.latest_cycle}</td>
                  <td className="px-4 py-2.5 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{eng.rul_at_end}</td>
                  <td className="px-4 py-2.5 font-mono text-xs" style={{ color: eng.alert_count > 0 ? 'var(--status-warn)' : 'var(--text-muted)' }}>{eng.alert_count}</td>
                  <td className="px-4 py-2.5 text-xs" style={{ color: 'var(--text-muted)' }}>{relativeTime(eng.last_seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
