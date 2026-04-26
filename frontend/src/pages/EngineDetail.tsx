import { useState } from 'react'
import { useParams, useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid, Legend, Label,
} from 'recharts'
import type { Dataset } from '@/lib/types'
import { api } from '@/lib/api'
import { scoreColor, regimeName } from '@/lib/utils'
import { sensorLabel } from '@/lib/sensorNames'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { InfoTooltip } from '@/components/ui/InfoTooltip'
import { ArrowLeft, MessageCircle } from 'lucide-react'
import { setChatContext, openChatWith } from '@/components/ui/ChatWidget'

const TABS = ['Sensor Readings', 'Wear Signals', 'Flight Conditions', 'Sensor Integrity', 'Detection Log'] as const
type Tab = typeof TABS[number]

const TAB_HINTS: Record<Tab, string> = {
  'Sensor Readings': 'Measured values for the 5 key engine sensors over every flight cycle, compared to what the model expected given altitude, Mach and throttle',
  'Wear Signals': 'How far each sensor deviates from its expected value (in standard deviations). Values beyond ±3σ indicate abnormal component behaviour',
  'Flight Conditions': 'Operating condition assigned to each flight cycle — altitude, Mach number, and throttle setting. FD001/FD003 fly one fixed condition; FD002/FD004 mix six',
  'Sensor Integrity': 'Cross-check that HPC Pressure Ratio and Bypass Ratio are physically consistent. Decoupling suggests a sensor fault rather than real engine wear',
  'Detection Log': 'Step-by-step record of how the system reached its decision — from raw sensor ingestion through operating-condition adjustment to final risk score',
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

export function EngineDetail() {
  const { id } = useParams<{ id: string }>()
  const [sp] = useSearchParams()
  const dataset = (sp.get('dataset') ?? 'FD001') as Dataset
  const engineId = +(id ?? 1)
  const [tab, setTab] = useState<Tab>('Sensor Readings')

  const { data: engine } = useQuery({
    queryKey: ['engine', dataset, engineId],
    queryFn: () => api.getEngine(dataset, engineId),
  })

  const { data: traces } = useQuery({
    queryKey: ['traces', dataset, engineId],
    queryFn: () => api.getSensorTraces(dataset, engineId),
  })

  const { data: alerts = [] } = useQuery({
    queryKey: ['alerts', dataset],
    queryFn: () => api.getAlerts(dataset),
  })

  const engineAlerts = alerts.filter(a => a.engine_id === engineId)
    .sort((a, b) => a.cycle - b.cycle)
  const firstAlert = engineAlerts[0]

  if (!engine) return (
    <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-muted)' }}>
      Loading engine data…
    </div>
  )

  // Push engine context into AI assistant (re-runs whenever tab changes)
  setChatContext({
    page: 'engine-detail',
    dataset,
    engine_id: engineId,
    engine_score: engine.latest_score,
    engine_decision: engine.latest_decision,
    engine_rul: engine.rul_at_end,
    engine_regime: engine.regime,
    active_tab: tab,
  })

  return (
    <div className="p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3 flex-wrap">
        <Link to={`/?dataset=${dataset}`} className="p-1.5 rounded-lg hover:bg-[var(--border)] transition-colors" style={{ color: 'var(--text-muted)' }}>
          <ArrowLeft size={16} />
        </Link>
        <h1 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>Engine #{engineId}</h1>
        <Badge variant="indigo">{dataset}</Badge>
        {(dataset === 'FD002' || dataset === 'FD004') && (
          <Badge variant="outline">Condition: {regimeName(engine.regime)}</Badge>
        )}
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Cycle {engine.latest_cycle} · {engine.rul_at_end} cycles to failure
        </span>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Risk Score', value: engine.latest_score.toFixed(3), accent: true, hint: 'Overall risk: combines how far sensor readings deviate from expected values given current flight conditions. ≥ 0.30 = action required, 0.20–0.30 = monitor, < 0.20 = normal' },
          { label: 'Status', value: engine.latest_decision === 'ALERT' ? 'Action Required' : engine.latest_decision === 'UNCERTAIN' ? 'Monitor' : 'Normal', hint: 'System recommendation based on the risk score' },
          { label: 'Times Flagged', value: engine.alert_count, hint: 'How many of the last 20 flight cycles had a risk score above threshold — higher means sustained degradation, not a one-off spike' },
          { label: 'Cycles to Failure', value: `${engine.rul_at_end}`, hint: 'Remaining Useful Life per NASA ground-truth labels — how many flight cycles remain before this engine is expected to fail' },
        ].map(k => (
          <Card key={k.label}>
            <CardContent className="pt-3 pb-3">
              <p className="text-xs flex items-center" style={{ color: 'var(--text-muted)' }}>
                {k.label}
                <InfoTooltip text={k.hint} />
              </p>
              <p className="text-xl font-mono font-bold mt-0.5" style={{ color: k.accent ? scoreColor(engine.latest_score) : 'var(--text-primary)' }}>
                {k.value}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-0.5 border-b overflow-x-auto" style={{ borderColor: 'var(--border)' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="px-4 py-2 text-xs font-medium whitespace-nowrap transition-colors cursor-pointer border-b-2 flex items-center"
            style={{
              borderBottomColor: tab === t ? 'var(--accent)' : 'transparent',
              color: tab === t ? 'var(--accent)' : 'var(--text-muted)',
            }}
          >
            {t}
            <InfoTooltip text={TAB_HINTS[t]} side="bottom" />
          </button>
        ))}
      </div>

      {tab === 'Sensor Readings' && traces && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Key Sensor Readings over Time</CardTitle>
            <button onClick={() => openChatWith('Explain the Sensor Readings chart — what does the gap between the observed and predicted lines mean for this engine?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            {Object.entries(traces).map(([sname, data]) => (
              <div key={sname} className="mb-6">
                <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                  {sensorLabel(sname)}
                  <span className="font-normal ml-1.5" style={{ color: 'var(--text-muted)' }}>({sname})</span>
                </p>
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="cycle" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} width={55} />
                    <Tooltip contentStyle={{ fontSize: 11 }} />
                    <Legend iconSize={10} wrapperStyle={{ fontSize: 10 }} />
                    {firstAlert && <ReferenceLine x={firstAlert.cycle} stroke="var(--status-alert)" strokeDasharray="4 2" label={{ value: 'first alert', fontSize: 9, fill: 'var(--status-alert)' }} />}
                    <Line type="monotone" dataKey="value" stroke="var(--accent-cyan)" dot={false} strokeWidth={1.5} name="Observed" />
                    <Line type="monotone" dataKey="predicted" stroke="#f59e0b" dot={false} strokeDasharray="4 2" strokeWidth={1.5} name="Predicted (causal model)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {tab === 'Wear Signals' && traces && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Sensor Deviation from Expected Values (σ)</CardTitle>
            <button onClick={() => openChatWith('Explain the Wear Signals chart — what do the σ bands mean and what does this engine\'s deviation trend indicate?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            {(dataset === 'FD002' || dataset === 'FD004') && (
              <div className="mb-4 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--accent-light)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                Wear signals are computed using a causal model trained on FD001 (single operating condition). This engine operates under 6 conditions — large deviations may reflect condition changes rather than true component wear. The risk score accounts for this via operating-condition normalisation.
              </div>
            )}
            {Object.entries(traces).map(([sname, data]) => (
              <div key={sname} className="mb-6">
                <p className="text-xs font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                  {sensorLabel(sname)}
                  <span className="font-normal ml-1.5" style={{ color: 'var(--text-muted)' }}>— deviation from expected</span>
                </p>
                <ResponsiveContainer width="100%" height={130}>
                  <LineChart data={data.map(pt => ({ ...pt, residual: Math.max(-10, Math.min(10, pt.residual)) }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="cycle" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} width={40} domain={[-10, 10]} label={{ value: 'σ', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-muted)' }} />
                    <Tooltip contentStyle={{ fontSize: 11 }} formatter={(v: unknown) => [typeof v === 'number' ? `${v.toFixed(2)}σ` : '—', 'Residual']} />
                    <ReferenceLine y={0} stroke="var(--text-muted)" strokeDasharray="2 2" />
                    <ReferenceLine y={3} stroke="var(--status-alert)" strokeDasharray="4 2">
                      <Label value="+3σ" position="insideTopRight" fontSize={8} fill="var(--status-alert)" />
                    </ReferenceLine>
                    <ReferenceLine y={-3} stroke="var(--status-alert)" strokeDasharray="4 2">
                      <Label value="-3σ" position="insideBottomRight" fontSize={8} fill="var(--status-alert)" />
                    </ReferenceLine>
                    <ReferenceLine y={1} stroke="var(--status-warn)" strokeDasharray="2 2">
                      <Label value="+1σ" position="insideTopRight" fontSize={8} fill="var(--status-warn)" />
                    </ReferenceLine>
                    <ReferenceLine y={-1} stroke="var(--status-warn)" strokeDasharray="2 2" />
                    <Line type="monotone" dataKey="residual" stroke="var(--status-alert)" dot={false} strokeWidth={1.5} name="Residual (σ)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {tab === 'Flight Conditions' && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Operating Condition per Flight Cycle</CardTitle>
            <button onClick={() => openChatWith('Explain the Flight Conditions chart — what do the colours represent and how does operating condition affect anomaly scoring?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            {dataset === 'FD001' || dataset === 'FD003' ? (
              <p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                Single operating condition — all cycles assigned to regime 0
              </p>
            ) : (
              <>
                {/* Regime ribbon */}
                <div className="relative h-10 rounded-lg overflow-hidden mb-4" style={{ background: 'var(--border)' }}>
                  {Array.from({ length: engine.latest_cycle }, (_, i) => {
                    const regime = (engine.regime + Math.floor(i / 30)) % 6
                    const colors = ['#4f46e5','#06b6d4','#16a34a','#d97706','#dc2626','#9333ea']
                    return (
                      <div
                        key={i}
                        className="absolute top-0 bottom-0"
                        style={{ left: `${(i / engine.latest_cycle) * 100}%`, width: `${100 / engine.latest_cycle}%`, background: colors[regime] + '80' }}
                      />
                    )
                  })}
                  {firstAlert && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5"
                      style={{ left: `${(firstAlert.cycle / engine.latest_cycle) * 100}%`, background: 'var(--status-alert)' }}
                    />
                  )}
                </div>
                <div className="flex gap-3 flex-wrap">
                  {[0,1,2,3,4,5].map(r => (
                    <div key={r} className="flex items-center gap-1.5 text-xs">
                      <div className="w-3 h-3 rounded-sm" style={{ background: ['#4f46e5','#06b6d4','#16a34a','#d97706','#dc2626','#9333ea'][r] }} />
                      <span style={{ color: 'var(--text-muted)' }}>C{r}: {regimeName(r)}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                  <div className="flex items-center gap-3">
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>First alert cycle</p>
                      <p className="text-lg font-mono font-bold" style={{ color: 'var(--status-alert)' }}>
                        {firstAlert?.cycle ?? '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>True failure cycle</p>
                      <p className="text-lg font-mono font-bold" style={{ color: 'var(--text-primary)' }}>
                        {engine.latest_cycle + engine.rul_at_end}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Lead time</p>
                      <p className="text-lg font-mono font-bold" style={{ color: 'var(--status-ok)' }}>
                        {firstAlert ? `${engine.latest_cycle + engine.rul_at_end - firstAlert.cycle} cy` : '—'}
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {tab === 'Sensor Integrity' && traces && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>HPC Pressure ↔ Bypass Ratio Coupling Check</CardTitle>
            <button onClick={() => openChatWith('Explain the Sensor Integrity chart — what is the G-statistic line, what does the 26.30 threshold mean, and what happens when it is crossed?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            {(dataset === 'FD002' || dataset === 'FD004') && (
              <div className="mb-3 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--accent-light)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                Note: coupling check uses coefficients trained on FD001 (single condition). Values may be elevated for multi-condition datasets — treat as indicative only.
              </div>
            )}
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={traces.sensor_11.map((pt, i) => {
                const raw = Math.abs(pt.residual) * Math.abs((traces.sensor_15[i]?.residual ?? 0)) * 6
                const clamped = Math.min(raw, 60)
                return {
                  cycle: pt.cycle,
                  g_stat: +clamped.toFixed(2),
                  veto_factor: +(1.0 - 0.5 * Math.min(raw / 26.30, 1.0)).toFixed(3),
                }
              })}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="cycle" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[0, 60]} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine y={26.30} stroke="var(--status-veto)" strokeDasharray="4 2"
                  label={{ value: 'χ²(16, 0.05) = 26.30', fontSize: 9, fill: 'var(--status-veto)' }} />
                <Line type="monotone" dataKey="g_stat" stroke="var(--accent)" dot={false} strokeWidth={1.5} name="G-statistic (capped at 60)" />
                <Line type="monotone" dataKey="veto_factor" stroke="var(--status-veto)" dot={false} strokeDasharray="3 2" strokeWidth={1} name="Veto factor" />
                <Legend />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
              G &gt; 26.30 → coupling broken → graduated veto applied (max 50% score reduction). Veto inactive for first 100 cycles (cold start). Chart capped at 60 for readability.
            </p>
          </CardContent>
        </Card>
      )}

      {tab === 'Detection Log' && engineAlerts.length > 0 && (
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Detection Log — Latest Flagged Cycle</CardTitle>
            <button onClick={() => openChatWith('Walk me through this detection log — what does each step do and how did the system reach its final decision for this engine?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {engineAlerts[engineAlerts.length - 1].trace.map((node, i) => (
                <div key={node.node} className="rounded-lg border p-3" style={{ borderColor: 'var(--border)' }}>
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] font-bold shrink-0"
                      style={{ background: 'var(--accent)' }}
                    >{i + 1}</span>
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>{friendlyNodeName(node.node)}</span>
                    <span className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--border)', color: 'var(--text-muted)' }}>
                      {node.latency_ms}ms
                    </span>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{node.summary}</p>
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Total latency: {engineAlerts[engineAlerts.length - 1].trace.reduce((s, n) => s + n.latency_ms, 0)}ms
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {tab === 'Detection Log' && engineAlerts.length === 0 && (
        <Card>
          <CardContent className="py-10 text-center">
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No alerts fired for this engine</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
