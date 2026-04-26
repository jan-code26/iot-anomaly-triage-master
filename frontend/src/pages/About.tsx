import { FileText, GitFork } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

const RESULTS = [
  { label: '4.1× IF coverage', sub: 'FD001 full pipeline vs Isolation Forest', color: 'var(--accent)' },
  { label: 'F1 = 0.352', sub: 'FD004 — highest across all datasets', color: 'var(--status-ok)' },
  { label: '100% FP rate → 0', sub: 'Global z-score vs regime-aware causal on FD002', color: 'var(--status-alert)' },
  { label: '53% earlier alerts', sub: 'Mean lead time 164.9 vs 107.4 cycles (FD001)', color: 'var(--accent-cyan)' },
]

export function About() {
  return (
    <div className="p-5 max-w-4xl space-y-8">
      {/* Hero */}
      <div className="rounded-2xl p-8 border" style={{ background: 'var(--accent-light)', borderColor: 'var(--accent)' }}>
        <h1 className="text-2xl font-bold leading-tight mb-3" style={{ color: 'var(--text-primary)' }}>
          Don't Trust the Sensors: Regime-Aware Causal Anomaly Triage for Industrial IoT
        </h1>
        <p className="text-sm font-medium mb-1" style={{ color: 'var(--accent)' }}>Jahnavi Patel</p>
        <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
          College of Engineering, Northeastern University · patel.jahnavi@northeastern.edu
        </p>
        <div className="flex gap-2 flex-wrap">
          <a
            href="https://github.com/jan-code26/iot-anomaly-triage"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
            style={{ background: 'var(--accent)', color: '#fff' }}
          >
            <GitFork size={14} /> Repository
          </a>
          <span
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium"
            style={{ background: 'var(--border)', color: 'var(--text-muted)' }}
          >
            <FileText size={14} /> NASA CMAPSS Benchmark
          </span>
        </div>
      </div>

      {/* Key results */}
      <div>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Key Results</h2>
        <div className="grid sm:grid-cols-2 gap-3">
          {RESULTS.map(r => (
            <Card key={r.label}>
              <CardContent className="pt-4 pb-4">
                <p className="text-2xl font-mono font-bold" style={{ color: r.color }}>{r.label}</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{r.sub}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Abstract */}
      <div>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Abstract</h2>
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Industrial IoT anomaly detection systems routinely generate false alarms because they compare sensor readings
              against global statistical thresholds that ignore operating conditions. A turbofan engine running at high altitude
              produces temperature and pressure readings that are normal for that regime but anomalous relative to global
              training-set means — triggering alarms that have nothing to do with engine health. We present a causal anomaly
              triage system that conditions scoring on a learned operating-regime classifier and augments it with a
              physics-based structural veto grounded in the isentropic compression relation, deployed with a seven-node
              LangGraph agent for operator-correctable explanations. We evaluate on all four NASA CMAPSS sub-datasets:
              FD001/FD003 (single operating condition) and FD002/FD004 (six operating conditions).
            </p>
            <p className="text-sm leading-relaxed mt-3" style={{ color: 'var(--text-muted)' }}>
              On multi-condition data, both a global z-score baseline and a retrained z-score baseline produce
              100% false positive rates (F1 = 0.000) — confirming that domain-shift correction alone is insufficient.
              Regime-aware causal scoring reduces false alarms to 66% coverage on FD002 (F1 = 0.279) and 57% on FD004
              (F1 = 0.352), with statistically significant false positive reduction (Fisher <em>p</em> &lt; 0.001) on both.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* System overview */}
      <div>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>System Overview</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              title: 'Causal DAG Scoring',
              body: 'Causal DAG structure defined and validated with DoWhy. Per-request scoring uses regime-conditioned LinearRegression with pre-computed coefficients for low-latency inference (DoWhy v0.11 requires ≥2 rows per call; live readings arrive one at a time). Anomaly score = residual from causally-predicted value, not global mean deviation.',
              badge: 'Core mechanism',
            },
            {
              title: 'Physics-Based Veto',
              body: 'G-test for independence on sensor_11 / sensor_15 coupling (5×5 table, χ²(df=16) = 26.30). Graduated penalty 1.0 − 0.5 × min(G/26.30, 1.0). Distinguishes sensor faults from real degradation.',
              badge: 'Structural layer',
            },
            {
              title: 'LangGraph 7-Node Agent',
              body: 'ingest_validator → regime_classifier → causal_reasoner → physics_veto → cache_lookup → llm_explainer → decision_writer. Full audit trail in reasoning_traces table.',
              badge: 'Operator interface',
            },
            {
              title: 'Human Feedback Loop',
              body: 'Operators submit TRUE_POSITIVE / FALSE_POSITIVE / UNCERTAIN labels. cache_lookup node applies 0.7× confidence penalty after ≥2 FALSE_POSITIVE labels per engine.',
              badge: 'Closed-loop',
            },
          ].map(s => (
            <Card key={s.title}>
              <CardContent className="pt-4 pb-4">
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>{s.title}</h3>
                  <Badge variant="indigo" className="text-[9px]">{s.badge}</Badge>
                </div>
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{s.body}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Citation */}
      <div>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Citation</h2>
        <Card>
          <CardContent className="pt-4">
            <pre
              className="text-xs overflow-x-auto whitespace-pre-wrap"
              style={{ color: 'var(--text-muted)', background: 'var(--bg-sidebar)', padding: '12px', borderRadius: '8px' }}
            >{`@article{patel2026anomaly,
  title   = {Don't Trust the Sensors: Regime-Aware Causal Anomaly Triage for Industrial IoT},
  author  = {Patel, Jahnavi},
  year    = {2026},
  school  = {Northeastern University},
  note    = {https://github.com/jan-code26/iot-anomaly-triage}
}`}</pre>
          </CardContent>
        </Card>
      </div>

      {/* Stack */}
      <div>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Tech Stack</h2>
        <div className="flex gap-2 flex-wrap">
          {['FastAPI', 'PostgreSQL (Neon)', 'LangGraph', 'DoWhy', 'scikit-learn', 'Groq / Llama-3', 'Render', 'React + Vite', 'Recharts'].map(t => (
            <Badge key={t} variant="outline">{t}</Badge>
          ))}
        </div>
      </div>
    </div>
  )
}
