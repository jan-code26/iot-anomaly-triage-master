import { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { Dataset, Decision, FeedbackLabel } from '@/lib/types'
import { api } from '@/lib/api'
import { scoreColor, regimeName, relativeTime } from '@/lib/utils'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ScoreBar } from '@/components/ui/ScoreBar'
import { InfoTooltip } from '@/components/ui/InfoTooltip'
import { sensorLabel, recommendedAction } from '@/lib/sensorNames'
import { AlertTriangle, ShieldCheck, Eye } from 'lucide-react'

interface Props { dataset: Dataset }

export function Alerts({ dataset }: Props) {
  const qc = useQueryClient()
  const [selected, setSelected] = useState<string | null>(null)
  const [minScore, setMinScore] = useState(0)
  const [filterRegime, setFilterRegime] = useState<number | undefined>(undefined)

  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts', dataset],
    queryFn: () => api.getAlerts(dataset),
  })

  const filtered = useMemo(() =>
    alerts.filter(a =>
      a.anomaly_score >= minScore &&
      (filterRegime === undefined || a.regime === filterRegime)
    ), [alerts, minScore, filterRegime])

  const sel = filtered.find(a => a.id === selected) ?? filtered[0]

  const { mutate: submitFeedback, isPending } = useMutation({
    mutationFn: ({ id, label }: { id: string; label: FeedbackLabel }) =>
      api.postFeedback(id, label, dataset),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts', dataset] }),
  })

  const isMulti = dataset === 'FD002' || dataset === 'FD004'

  // Recommended action for selected alert
  const topSensor = sel?.sensor_residuals?.[0]?.sensor ?? ''
  const rec = sel ? recommendedAction(sel.decision, sel.rul_at_end ?? 0, topSensor) : null

  return (
    <div className="flex h-full min-h-0">
      {/* Left list */}
      <div className="w-72 shrink-0 border-r flex flex-col" style={{ borderColor: 'var(--border)' }}>
        {/* Filters */}
        <div className="px-3 py-3 border-b space-y-2" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
          <div className="flex items-center gap-2">
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Min risk</label>
            <input
              type="range" min={0} max={0.9} step={0.05}
              value={minScore}
              onChange={e => setMinScore(+e.target.value)}
              className="flex-1 h-1 accent-indigo-500"
            />
            <span className="text-xs font-mono w-8" style={{ color: 'var(--accent)' }}>{minScore.toFixed(2)}</span>
          </div>
          {isMulti && (
            <div className="flex items-center gap-2">
              <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Condition</label>
              <select
                value={filterRegime ?? ''}
                onChange={e => setFilterRegime(e.target.value === '' ? undefined : +e.target.value)}
                className="flex-1 text-xs rounded-md px-2 py-1 border"
                style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', color: 'var(--text-primary)' }}
              >
                <option value="">All conditions</option>
                {[0,1,2,3,4,5].map(r => <option key={r} value={r}>C{r} — {regimeName(r)}</option>)}
              </select>
            </div>
          )}
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{filtered.length} engines flagged · sorted by risk</p>
        </div>

        {/* Alert list */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <p className="text-xs p-4" style={{ color: 'var(--text-muted)' }}>Loading…</p>
          ) : filtered.map(a => (
            <button
              key={a.id}
              onClick={() => setSelected(a.id)}
              className="w-full text-left px-3 py-2.5 border-b hover:bg-[var(--border)] transition-colors"
              style={{
                borderColor: 'var(--border)',
                background: sel?.id === a.id ? 'var(--accent-light)' : undefined,
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-mono font-bold" style={{ color: 'var(--text-primary)' }}>
                  Engine #{a.engine_id}
                </span>
                <span className="text-xs font-mono font-bold" style={{ color: scoreColor(a.anomaly_score) }}>
                  {a.anomaly_score.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <DecisionBadge d={a.decision} />
                {isMulti && <Badge variant="outline" className="text-[9px]">C{a.regime}</Badge>}
                {a.feedback && <FeedbackBadge label={a.feedback} />}
              </div>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                {a.rul_at_end != null ? `~${a.rul_at_end} cycles to failure` : `cycle ${a.cycle}`} · {relativeTime(a.triggered_at)}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Right detail */}
      {sel ? (
        <div className="flex-1 overflow-y-auto p-5 space-y-4">

          {/* Header */}
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
              Engine #{sel.engine_id} — Cycle {sel.cycle}
            </h2>
            <DecisionBadge d={sel.decision} />
            {isMulti && <Badge variant="indigo">{regimeName(sel.regime)}</Badge>}
            {sel.rul_at_end != null && (
              <span className="text-sm font-semibold" style={{ color: scoreColor(sel.anomaly_score) }}>
                ~{sel.rul_at_end} cycles to predicted failure
              </span>
            )}
          </div>

          {/* ── Recommended Action ── most important card */}
          {rec && (
            <Card style={{
              borderColor: rec.urgency === 'critical' ? 'var(--status-alert)' : rec.urgency === 'warn' ? 'var(--status-warn)' : 'var(--border)',
              borderWidth: rec.urgency !== 'monitor' ? 2 : 1,
            }}>
              <CardContent className="pt-4 pb-4">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 shrink-0">
                    {rec.urgency === 'critical' && <AlertTriangle size={18} style={{ color: 'var(--status-alert)' }} />}
                    {rec.urgency === 'warn'     && <AlertTriangle size={18} style={{ color: 'var(--status-warn)' }} />}
                    {rec.urgency === 'monitor'  && <Eye size={18} style={{ color: 'var(--text-muted)' }} />}
                  </div>
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide mb-1"
                      style={{ color: rec.urgency === 'critical' ? 'var(--status-alert)' : rec.urgency === 'warn' ? 'var(--status-warn)' : 'var(--text-muted)' }}>
                      Recommended Action
                    </p>
                    <p className="text-sm leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                      {rec.action}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Risk scores */}
          <div className="grid grid-cols-3 gap-3">
            {[
              {
                label: 'Overall Risk',
                score: sel.anomaly_score,
                main: true,
                hint: 'Combined score blending component health and fleet-wide comparison. ≥ 0.30 = action required, 0.20–0.30 = monitor, < 0.20 = normal',
              },
              {
                label: 'Component Health',
                score: sel.causal_score,
                hint: 'How far each sensor deviates from its expected value given current altitude, Mach and throttle — isolates true engine wear from operating-condition effects',
              },
              {
                label: 'Fleet Comparison',
                score: sel.z_score,
                hint: 'How this engine compares to the healthy training fleet across all sensors. High values mean readings are unusual fleet-wide',
              },
            ].map(s => (
              <Card key={s.label}>
                <CardContent className="pt-4 pb-4">
                  <p className="text-xs mb-1 flex items-center" style={{ color: 'var(--text-muted)' }}>
                    {s.label}
                    <InfoTooltip text={s.hint} />
                  </p>
                  <p className="text-2xl font-bold font-mono"
                    style={{ color: s.main ? scoreColor(s.score) : 'var(--text-primary)' }}>
                    {s.score.toFixed(3)}
                  </p>
                  <ScoreBar score={s.score} showLabel={false} height={4} />
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Sensor Integrity */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                {sel.physics_veto_active
                  ? <AlertTriangle size={14} className="mr-1.5" style={{ color: 'var(--status-veto)' }} />
                  : <ShieldCheck size={14} className="mr-1.5" style={{ color: 'var(--status-ok)' }} />
                }
                Sensor Integrity Check
                <InfoTooltip text="Cross-checks HPC Pressure Ratio and Bypass Ratio for physical consistency. If they decouple, the risk score is reduced by up to 50% to avoid false alarms from sensor faults rather than real engine wear" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-6">
                <div>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Sensor coupling</p>
                  <Badge variant={sel.physics_veto_active ? 'veto' : 'ok'}>
                    {sel.physics_veto_active ? '⚡ Sensors decoupled — possible sensor fault' : '✓ Sensors physically consistent'}
                  </Badge>
                </div>
                <div>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Score adjustment</p>
                  <p className="text-lg font-mono font-bold" style={{ color: 'var(--text-primary)' }}>
                    ×{sel.veto_factor.toFixed(3)}
                  </p>
                </div>
              </div>
              {sel.physics_veto_active && (
                <p className="text-xs mt-2 p-2 rounded" style={{ background: 'var(--accent-light)', color: 'var(--text-muted)' }}>
                  Risk score reduced because sensor readings are physically inconsistent. Rule out sensor failure before scheduling maintenance.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Top degraded sensors */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                Degraded Components
                <InfoTooltip text="Sensors ranked by how far they deviate from expected values for current operating conditions. Higher deviation = stronger sign of wear in that part of the engine" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[...sel.sensor_residuals].sort((a, b) => b.z - a.z).map(sr => (
                  <div key={sr.sensor}>
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {sensorLabel(sr.sensor)}
                        <span className="font-normal ml-1" style={{ color: 'var(--text-muted)' }}>({sr.sensor})</span>
                      </span>
                      <span className="font-mono" style={{ color: scoreColor(sr.z / 5) }}>
                        {sr.z.toFixed(2)}σ deviation
                      </span>
                    </div>
                    <div className="relative h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${Math.min(sr.z / 5 * 100, 100)}%`, background: scoreColor(sr.z / 5) }}
                      />
                      <div
                        className="absolute top-0 bottom-0 w-0.5"
                        style={{ left: `${sr.noise_floor / 5 * 100}%`, background: 'var(--text-muted)', opacity: 0.5 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* LLM explanation */}
          {sel.llm_explanation && (
            <Card>
              <CardHeader><CardTitle>Plain-English Summary</CardTitle></CardHeader>
              <CardContent>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                  {sel.llm_explanation}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Detection log (was Reasoning Trace) */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                Detection Log
                <InfoTooltip text="Step-by-step record of how the system reached its conclusion — from raw sensor ingestion through operating-condition adjustment to final decision" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {sel.trace.map((node, i) => (
                  <div key={node.node} className="flex gap-3 items-start">
                    <div className="flex flex-col items-center">
                      <div
                        className="w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold shrink-0"
                        style={{ background: 'var(--accent)' }}
                      >
                        {i + 1}
                      </div>
                      {i < sel.trace.length - 1 && (
                        <div className="w-px h-4 mt-1" style={{ background: 'var(--border)' }} />
                      )}
                    </div>
                    <div className="flex-1 pb-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                          {friendlyNodeName(node.node)}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{ background: 'var(--border)', color: 'var(--text-muted)' }}>
                          {node.latency_ms}ms
                        </span>
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{node.summary}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Operator feedback */}
          <Card>
            <CardHeader><CardTitle>Maintenance Verdict</CardTitle></CardHeader>
            <CardContent>
              {sel.feedback && (
                <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                  Current verdict: <FeedbackBadge label={sel.feedback} />
                </p>
              )}
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                After inspection, record your finding to improve future detections:
              </p>
              <div className="flex gap-2 flex-wrap">
                {(['TRUE_POSITIVE', 'FALSE_POSITIVE', 'UNCERTAIN'] as FeedbackLabel[]).map(label => (
                  <button
                    key={label}
                    disabled={isPending}
                    onClick={() => submitFeedback({ id: sel.id, label })}
                    className="px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors cursor-pointer disabled:opacity-50"
                    style={{
                      borderColor: label === 'TRUE_POSITIVE' ? 'var(--status-alert)' : label === 'FALSE_POSITIVE' ? 'var(--status-ok)' : 'var(--status-warn)',
                      color: label === 'TRUE_POSITIVE' ? 'var(--status-alert)' : label === 'FALSE_POSITIVE' ? 'var(--status-ok)' : 'var(--status-warn)',
                      background: 'var(--bg-card)',
                    }}
                  >
                    {label === 'TRUE_POSITIVE' ? 'Confirmed fault' : label === 'FALSE_POSITIVE' ? 'False alarm' : 'Inconclusive'}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center" style={{ color: 'var(--text-muted)' }}>
          <p className="text-sm">Select an engine to inspect it</p>
        </div>
      )}
    </div>
  )
}

function friendlyNodeName(node: string): string {
  const map: Record<string, string> = {
    ingest_validator:   'Sensor data validated',
    regime_classifier:  'Operating condition identified',
    causal_reasoner:    'Component health assessed',
    physics_veto:       'Sensor integrity checked',
    cache_lookup:       'Historical baseline retrieved',
    llm_explainer:      'Plain-English summary generated',
    decision_writer:    'Final decision recorded',
  }
  return map[node] ?? node
}

function DecisionBadge({ d }: { d: Decision }) {
  const v = d === 'ALERT' ? 'alert' : d === 'UNCERTAIN' ? 'warn' : 'ok'
  const label = d === 'ALERT' ? 'Action Required' : d === 'UNCERTAIN' ? 'Monitor' : 'Normal'
  return <Badge variant={v as 'alert' | 'warn' | 'ok'}>{label}</Badge>
}

function FeedbackBadge({ label }: { label: FeedbackLabel }) {
  const v = label === 'TRUE_POSITIVE' ? 'alert' : label === 'FALSE_POSITIVE' ? 'ok' : 'warn'
  const text = label === 'TRUE_POSITIVE' ? 'Confirmed fault' : label === 'FALSE_POSITIVE' ? 'False alarm' : 'Inconclusive'
  return <Badge variant={v as 'alert' | 'ok' | 'warn'}>{text}</Badge>
}
