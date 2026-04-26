import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { Dataset } from '@/lib/types'
import { api } from '@/lib/api'
import { scoreColor, regimeName, fmtPct } from '@/lib/utils'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { InfoTooltip } from '@/components/ui/InfoTooltip'
import { ScoreBar } from '@/components/ui/ScoreBar'
import { Sparkline } from '@/components/ui/Sparkline'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, MessageCircle } from 'lucide-react'
import { setChatContext, openChatWith } from '@/components/ui/ChatWidget'

interface Props { dataset: Dataset }

// Evaluation-set metrics per dataset — sourced from ablation_table.csv (FD001/FD003)
// and fd002/fd004_regime_table.csv (regime-aware causal variant).
const AVG_WARNING_LEAD: Record<Dataset, number> = {
  FD001: 165,  // ablation_table.csv: Full pipeline mean = 164.9
  FD002: 168,  // fd002_regime_table.csv: Regime-aware causal mean = 167.5
  FD003: 218,  // fd003_ablation_table.csv: Full pipeline mean = 218.3
  FD004: 196,  // fd004_regime_table.csv: Regime-aware causal mean = 196.3
}
// Alert precision = TP / (TP + FP). Source: Precision column in each regime table.
const ALERT_PRECISION: Record<Dataset, string> = {
  FD001: '23%',  // ablation_table.csv: Full pipeline Precision = 0.232
  FD002: '25%',  // fd002_regime_table.csv: Regime-aware causal Precision = 0.247
  FD003: '16%',  // fd003_ablation_table.csv: Full pipeline Precision = 0.157
  FD004: '37%',  // fd004_regime_table.csv: Regime-aware causal Precision = 0.373
}
const DETECTION_RATE: Record<Dataset, string> = {
  FD001: '69%',  // ablation_table.csv: Full pipeline Coverage = 69%
  FD002: '66%',  // fd002_regime_table.csv: Regime-aware causal Coverage = 66%
  FD003: '89%',  // fd003_ablation_table.csv: Full pipeline Coverage = 89%
  FD004: '57%',  // fd004_regime_table.csv: Regime-aware causal Coverage = 57%
}

const DATASET_DESC: Record<Dataset, string> = {
  FD001: 'Single operating condition · HPC degradation · 100 engines',
  FD002: '6 operating conditions · HPC degradation · 259 engines',
  FD003: 'Single operating condition · Fan + HPC degradation · 100 engines',
  FD004: '6 operating conditions · Fan + HPC degradation · 248 engines',
}

export function Overview({ dataset }: Props) {
  const nav = useNavigate()
  const { data: engines = [], isLoading: loadE } = useQuery({
    queryKey: ['engines', dataset],
    queryFn: () => api.getEngines(dataset),
  })

  const isMulti = dataset === 'FD002' || dataset === 'FD004'
  const alerted = engines.filter(e => e.alerted)

  // Push fleet context into AI assistant whenever data changes
  if (engines.length > 0) {
    setChatContext({ page: 'overview', dataset, fleet_total: engines.length, fleet_alerted: alerted.length })
  }
  const avgLead = AVG_WARNING_LEAD[dataset]
  const alertPrecision = ALERT_PRECISION[dataset]

  // Most critical engines: ALERT decision sorted by lowest RUL
  const criticalEngines = engines
    .filter(e => e.latest_decision === 'ALERT')
    .sort((a, b) => a.rul_at_end - b.rul_at_end)
    .slice(0, 5)

  // Alerts per regime — use engines so NORMAL engines appear
  const regimeCounts = isMulti
    ? [0,1,2,3,4,5].map(r => ({
        name: `C${r} · ${regimeName(r)}`,
        'Action required': engines.filter(e => e.regime === r && e.latest_decision === 'ALERT').length,
        'Monitor': engines.filter(e => e.regime === r && e.latest_decision === 'UNCERTAIN').length,
        'Normal': engines.filter(e => e.regime === r && e.latest_decision === 'NORMAL').length,
      }))
    : []

  // RUL histogram
  const buckets = [0, 50, 100, 150, 200, 300]
  const rulHist = buckets.slice(0, -1).map((lo, i) => {
    const hi = buckets[i + 1]
    return {
      range: `${lo}–${hi < 300 ? hi : '300+'}`,
      'Action required': engines.filter(e => e.alerted && e.rul_at_end >= lo && e.rul_at_end < hi).length,
      'Normal': engines.filter(e => !e.alerted && e.rul_at_end >= lo && e.rul_at_end < hi).length,
    }
  })

  return (
    <div className="p-5 space-y-5">

      {/* Dataset description */}
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{DATASET_DESC[dataset]}</p>

      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        {[
          { label: 'Total Engines', value: engines.length, hint: 'Total number of turbofan engines in the test fleet for this dataset' },
          { label: 'Action Required', value: alerted.length, accent: true, hint: 'Engines with risk score ≥ 0.30 — these engines need maintenance attention' },
          { label: 'Avg Warning Lead', value: `${avgLead} cycles`, hint: `On average, the system flags an engine ${avgLead} flight cycles before failure — giving maintenance teams time to plan` },
          { label: 'Alert Precision', value: alertPrecision, hint: 'Fraction of flagged engines that are true positives (TP / alerted). Source: regime-aware causal evaluation on each dataset.' },
          { label: 'Dataset', value: dataset, hint: 'NASA CMAPSS benchmark fleet. FD001/FD003 = one fixed flight condition; FD002/FD004 = six different altitude/Mach/throttle combinations' },
        ].map(kpi => (
          <Card key={kpi.label}>
            <CardContent className="pt-4 pb-4">
              <p className="text-xs font-medium flex items-center" style={{ color: 'var(--text-muted)' }}>
                {kpi.label}
                <InfoTooltip text={kpi.hint} />
              </p>
              <p className="text-2xl font-bold font-mono mt-1"
                style={{ color: kpi.accent ? 'var(--status-alert)' : 'var(--text-primary)' }}>
                {kpi.value}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Most urgent engines banner */}
      {criticalEngines.length > 0 && (
        <Card style={{ borderColor: 'var(--status-alert)', borderWidth: 2 }}>
          <CardContent className="pt-3 pb-3">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle size={14} style={{ color: 'var(--status-alert)' }} />
              <p className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--status-alert)' }}>
                Engines needing immediate attention — sorted by proximity to failure
              </p>
            </div>
            <div className="flex gap-3 flex-wrap">
              {criticalEngines.map(e => (
                <button
                  key={e.engine_id}
                  onClick={() => nav(`/engines/${e.engine_id}?dataset=${dataset}`)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-colors hover:bg-[var(--accent-light)]"
                  style={{ borderColor: 'var(--status-alert)', color: 'var(--status-alert)', background: 'var(--bg-card)' }}
                >
                  Engine #{e.engine_id}
                  <span className="font-normal" style={{ color: 'var(--text-muted)' }}>~{e.rul_at_end} cycles left</span>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid lg:grid-cols-2 gap-5">
        {/* Status by operating condition or coverage bar */}
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>
              {isMulti ? 'Engine Status by Operating Condition' : 'Fleet Status Overview'}
            </CardTitle>
            <button onClick={() => openChatWith(isMulti ? 'Explain the Engine Status by Operating Condition chart — what do the coloured bars represent and what should I look for?' : 'Explain the Fleet Status Overview — what does this breakdown mean and what percentage of engines need attention?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            {isMulti ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={regimeCounts} barSize={16}>
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="Action required" fill="var(--status-alert)" radius={[4,4,0,0]} />
                  <Bar dataKey="Monitor" fill="var(--status-warn)" radius={[4,4,0,0]} />
                  <Bar dataKey="Normal" fill="var(--status-ok)" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="space-y-3 pt-2">
                {[
                  { label: 'Action required', pct: alerted.length / Math.max(engines.length, 1), color: 'var(--status-alert)' },
                  { label: 'Healthy', pct: (engines.length - alerted.length) / Math.max(engines.length, 1), color: 'var(--status-ok)' },
                ].map(row => (
                  <div key={row.label}>
                    <div className="flex justify-between text-xs mb-1">
                      <span style={{ color: 'var(--text-muted)' }}>{row.label}</span>
                      <span className="font-mono" style={{ color: row.color }}>{fmtPct(row.pct)}</span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                      <div className="h-full rounded-full transition-all duration-700" style={{ width: fmtPct(row.pct), background: row.color }} />
                    </div>
                  </div>
                ))}
                <p className="text-xs pt-2" style={{ color: 'var(--text-muted)' }}>
                  {`Detection rate: ${DETECTION_RATE[dataset]} of engines caught before failure`}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Cycles-to-failure distribution */}
        <Card>
          <CardHeader className="flex items-center justify-between">
            <CardTitle>
              Cycles to Failure — Flagged vs Healthy
              <InfoTooltip text="Engines flagged for action (orange) should cluster in the low cycle-to-failure buckets. Healthy engines (blue) should cluster in higher buckets. A good detector separates them cleanly." />
            </CardTitle>
            <button onClick={() => openChatWith('Explain the Cycles to Failure histogram — what does it mean when flagged and healthy engines overlap in the same bucket?')} className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg border transition-colors hover:bg-[var(--accent-light)]" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}><MessageCircle size={10} />Explain</button>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rulHist} barSize={20}>
                <XAxis dataKey="range" tick={{ fontSize: 10 }} label={{ value: 'cycles to failure', position: 'insideBottom', offset: -2, fontSize: 9, fill: 'var(--text-muted)' }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="Action required" fill="var(--status-warn)" radius={[3,3,0,0]} />
                <Bar dataKey="Normal" fill="var(--accent)" radius={[3,3,0,0]} opacity={0.6} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Engine grid */}
      <div>
        <h2 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
          Full Fleet — click any engine for detailed inspection
        </h2>
        {loadE ? (
          <div className="text-sm py-10 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2">
            {engines.map(eng => (
              <Card
                key={eng.engine_id}
                className="cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => nav(`/engines/${eng.engine_id}?dataset=${dataset}`)}
              >
                <CardContent className="pt-3 pb-3 px-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-mono font-bold" style={{ color: 'var(--text-primary)' }}>
                      #{eng.engine_id}
                    </span>
                    <span className="w-2 h-2 rounded-full" style={{ background: scoreColor(eng.latest_score) }} />
                  </div>
                  {isMulti && (
                    <Badge variant="outline" className="text-[10px] mb-1.5">{regimeName(eng.regime)}</Badge>
                  )}
                  <Sparkline data={eng.score_history} color={scoreColor(eng.latest_score)} width={100} height={24} />
                  <div className="mt-1.5">
                    <ScoreBar score={eng.latest_score} height={3} showLabel={false} />
                  </div>
                  <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                    {eng.rul_at_end} cycles left
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
