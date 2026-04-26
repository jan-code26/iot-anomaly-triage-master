import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, ReferenceLine, Legend,
} from 'recharts'
import {
  ABLATION_FD001, ABLATION_FD002, ABLATION_FD003, ABLATION_FD004,
  REGIME_CENTROIDS, ALPHA_SWEEP, W_SENSITIVITY,
} from '@/lib/mock/fixtures'
import type { AblationRow, Dataset } from '@/lib/types'
import { Card, CardContent } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

const ABLATION_MAP: Record<Dataset, AblationRow[]> = {
  FD001: ABLATION_FD001,
  FD002: ABLATION_FD002,
  FD003: ABLATION_FD003,
  FD004: ABLATION_FD004,
}

export function Methodology() {
  const [alpha, setAlpha] = useState(0.60)
  const [wVal, setWVal] = useState<50 | 100 | 150>(100)
  const [activeDataset, setActiveDataset] = useState<Dataset>('FD001')

  const alphaData = ALPHA_SWEEP.find(d => Math.abs(d.alpha - alpha) < 0.03) ?? ALPHA_SWEEP[12]
  const wData = W_SENSITIVITY.find(d => d.w === wVal)!
  const rows = ABLATION_MAP[activeDataset]

  return (
    <div className="p-5 max-w-5xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
          Methodology & Results
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Interactive companion to the paper. Explore the causal DAG, ablation tables, and parameter sensitivity.
        </p>
      </div>

      {/* ── 1. CMAPSS Dataset ─────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>NASA CMAPSS Benchmark</h2>
        <Card>
          <CardContent className="pt-5 pb-5">
            <p className="text-sm leading-relaxed mb-3" style={{ color: 'var(--text-muted)' }}>
              The Commercial Modular Aero-Propulsion System Simulation (CMAPSS) is a NASA benchmark
              that simulates turbofan engines running to failure under controlled fault conditions.
              Each engine begins healthy and degrades progressively — the task is to detect the onset
              of degradation early, giving operators a lead-time window to act before failure.
            </p>
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              The benchmark provides 21 sensor channels (temperatures, pressures, speeds, ratios)
              plus three operating setting inputs per cycle. Engines are divided across four
              sub-datasets that vary by the number of operating conditions and fault modes.
              FD001 and FD003 operate at a single altitude/speed/throttle combination, making
              them tractable for global statistics. FD002 and FD004 mix six distinct operating
              conditions — the core challenge this project is designed to solve.
            </p>
            <div className="grid sm:grid-cols-2 gap-3 mb-4">
              {[
                { ds: 'FD001', cond: '1 operating condition', fault: 'HPC degradation', engines: 100, color: 'var(--accent)' },
                { ds: 'FD002', cond: '6 operating conditions', fault: 'HPC degradation', engines: 259, color: 'var(--status-warn)' },
                { ds: 'FD003', cond: '1 operating condition', fault: 'Fan + HPC degradation', engines: 100, color: 'var(--accent-cyan)' },
                { ds: 'FD004', cond: '6 operating conditions', fault: 'Fan + HPC degradation', engines: 248, color: 'var(--status-alert)' },
              ].map(d => (
                <div key={d.ds} className="rounded-lg p-3 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-sm font-mono font-bold" style={{ color: d.color }}>{d.ds}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded font-medium" style={{ background: 'var(--border)', color: 'var(--text-muted)' }}>{d.engines} engines</span>
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.cond}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{d.fault}</p>
                </div>
              ))}
            </div>
            <div className="rounded-lg p-3 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
              <p className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-primary)' }}>Why FD002 / FD004 are the hard datasets</p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                Six operating conditions (altitude × Mach × throttle combinations) create a multi-modal sensor
                distribution. A turbofan running at high altitude produces temperature and pressure readings that
                are <em>normal for that regime</em> but anomalous relative to sea-level means — triggering false
                alarms with no relation to engine health. A global z-score baseline trained on the full mixed
                distribution produces 100% false positive rates on both FD002 and FD004 (F1 = 0.000).
                Conditioning scores on the current operating regime cluster is the only approach that produces
                a meaningful triage signal.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── 2. Data Preparation ─────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Data Preparation Pipeline</h2>
        <Card>
          <CardContent className="pt-5 pb-5 space-y-4">
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Raw sensor streams from turbofan engines are noisy, intermittent, and arrive out of order.
              Three preprocessing layers clean and structure the data before any scoring occurs.
            </p>

            {/* Forward-fill */}
            <div className="rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
              <p className="text-xs font-bold mb-1.5" style={{ color: 'var(--text-primary)' }}>1 · Forward-Fill Imputation (5-cycle stale threshold)</p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                Sensors occasionally drop out mid-flight. Rather than discarding partial readings or
                substituting global means (which would corrupt regime-conditioned scores), the system
                carries the last known value forward for up to 5 cycles. If a sensor is still missing
                after 5 cycles it is treated as unavailable and excluded from that reading's score.
                This prevents stale data from silently drifting into anomaly detection.
              </p>
            </div>

            {/* PSI monitoring */}
            <div className="rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
              <p className="text-xs font-bold mb-1.5" style={{ color: 'var(--text-primary)' }}>2 · PSI Distribution Drift Monitor (threshold PSI {'>'} 0.2)</p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                The Population Stability Index (PSI) measures how much a sensor's distribution has shifted
                relative to its training baseline. PSI &lt; 0.10 = stable; 0.10–0.20 = monitor; &gt; 0.20 = action required.
                When a sensor crosses the action threshold, its baseline is automatically reset to the current
                rolling window — the new distribution becomes the reference going forward. This handles fleet
                upgrades, seasonal calibration drift, and sensor replacement without manual intervention.
              </p>
              <div className="mt-2 flex gap-3 text-[10px] font-mono">
                <span className="px-2 py-0.5 rounded" style={{ background: 'var(--status-ok)', color: '#fff' }}>PSI &lt; 0.10 stable</span>
                <span className="px-2 py-0.5 rounded" style={{ background: 'var(--status-warn)', color: '#fff' }}>0.10–0.20 monitor</span>
                <span className="px-2 py-0.5 rounded" style={{ background: 'var(--status-alert)', color: '#fff' }}>&gt; 0.20 auto-reset</span>
              </div>
            </div>

            {/* KMeans */}
            <div className="rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
              <p className="text-xs font-bold mb-1.5" style={{ color: 'var(--text-primary)' }}>3 · KMeans Regime Classification (k = 6)</p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                Before computing any anomaly score, each reading is assigned to one of six operating regime
                clusters. Clustering is performed on the three operating setting inputs — altitude (op_setting_1),
                Mach number (op_setting_2), and Throttle Resolver Angle (op_setting_3). KMeans with k = 6
                was selected by silhouette score optimisation (0.997 at k = 6 vs 0.930 at k = 5 and 0.934 at k = 7).
                Regime assignment takes a single nearest-centroid lookup — sub-millisecond at inference time.
                All downstream scoring — causal residuals and z-scores — uses statistics computed only from
                training data in the same regime cluster.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── 3. Causal DAG ─────────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Causal DAG & Scoring</h2>
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              Instead of flagging sensors that deviate from a global training mean, the system asks:
              <em> given the current operating conditions, what value should this sensor read?</em> The
              difference between the predicted and observed value — the causal residual — is the anomaly signal.
            </p>
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              The causal structure was derived from turbofan thermodynamics and validated using
              DoWhy v0.11 (DAG structure tests). The graph has three root nodes — altitude, Mach, and
              throttle setting — that drive five latent physical variables, each of which determines
              one observed sensor. Per-request scoring uses regime-conditioned LinearRegression with
              pre-computed coefficients for low-latency inference (DoWhy requires ≥ 2 rows; live
              readings arrive one at a time).
            </p>
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              Hover over any node below to see its physical meaning and role in the causal chain.
            </p>
            <CausalDAG />
            <p className="text-xs mt-4" style={{ color: 'var(--text-muted)' }}>
              Root nodes (op settings) drive latent physical variables, which determine observed sensor
              readings. Anomaly scoring computes the per-sensor residual relative to its causally-predicted
              value, conditioned on the current operating regime cluster.
            </p>

            {/* Physics veto */}
            <div className="mt-4 rounded-lg p-4 border" style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
              <p className="text-xs font-bold mb-1.5" style={{ color: 'var(--text-primary)' }}>Physics-Based Veto — G-Test on sensor_11 / sensor_15 Coupling</p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                sensor_11 (HPC outlet temperature) and sensor_15 (bypass ratio) must track each other
                according to the isentropic compression relation. A G-test on their joint distribution
                with χ²(df = 16) critical value of 26.30 distinguishes real engine degradation from
                a faulty sensor reporting a spurious reading. If the coupling is intact, the alert is
                suppressed — the anomaly is instrument noise, not engine health. This physics veto
                is the most important false-positive filter in the pipeline.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── 4. α slider ───────────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Blend Weight α</h2>
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm leading-relaxed mb-3" style={{ color: 'var(--text-muted)' }}>
              The final anomaly score blends the causal residual with a regime-conditioned z-score.
              The blend weight α controls how much each component contributes. At α = 1 the score
              is pure causal; at α = 0 it is pure z-score. The optimal value was determined on a
              held-out validation split of FD001 by sweeping α from 0 to 1 and maximising F1.
            </p>
            <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
              <code className="font-mono px-1.5 py-0.5 rounded text-xs" style={{ background: 'var(--border)' }}>
                combined_score = α × causal_score + (1 − α) × z_score
              </code>
            </p>
            <div className="flex items-center gap-4 mb-4">
              <span className="text-xs w-10" style={{ color: 'var(--text-muted)' }}>α = 0</span>
              <input
                type="range" min={0} max={1} step={0.05}
                value={alpha}
                onChange={e => setAlpha(+e.target.value)}
                className="flex-1 h-1"
              />
              <span className="text-xs w-10 text-right" style={{ color: 'var(--text-muted)' }}>α = 1</span>
            </div>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <p className="text-3xl font-mono font-bold" style={{ color: 'var(--accent)' }}>
                  {alpha.toFixed(2)}
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>α (causal weight)</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-mono font-bold" style={{ color: 'var(--status-ok)' }}>
                  {alphaData.f1.toFixed(3)}
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>F1 (W=100, FD001)</p>
              </div>
              <div className="text-center">
                <p className="text-3xl font-mono font-bold" style={{ color: 'var(--accent-cyan)' }}>
                  {(alphaData.coverage * 100).toFixed(0)}%
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Coverage</p>
              </div>
            </div>
            <p className="text-xs p-3 rounded-lg" style={{ background: 'var(--bg-sidebar)', color: 'var(--text-muted)' }}>
              {alpha === 0 ? 'Pure z-score: 98% coverage but F1 = 0.095 (premature alerts).' :
               alpha === 1 ? 'Pure causal: 64% coverage, F1 = 0.198.' :
               alpha <= 0.4 ? 'z-score dominates: high coverage, lower precision.' :
               alpha <= 0.7 ? 'Balanced: best F1 region. Paper reports α = 0.60 (evaluation-set calibrated).' :
               'Causal dominates: lower recall, higher precision on multi-condition data.'}
            </p>
            <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
              Note: α = 1.00 (pure causal) is used for FD002/FD004 — any z-score contribution collapses precision to near zero on multi-condition data.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* ── 5. W sensitivity ──────────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>Lead-Time Window W</h2>
        <Card>
          <CardContent className="pt-5">
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              An alert is counted as a true positive only if it fires within W cycles before the true
              failure cycle. W = 100 cycles is the paper's primary evaluation window (approximately
              equivalent to 100 flight hours). Smaller W rewards very precise late alerts; larger W
              rewards early warnings that give more lead time for maintenance scheduling.
            </p>
            <div className="flex gap-2 mb-4">
              {([50, 100, 150] as const).map(w => (
                <button
                  key={w}
                  onClick={() => setWVal(w)}
                  className="px-4 py-1.5 rounded-lg text-sm font-mono font-semibold transition-colors cursor-pointer"
                  style={{
                    background: wVal === w ? 'var(--accent)' : 'var(--border)',
                    color: wVal === w ? '#fff' : 'var(--text-muted)',
                  }}
                >
                  W = {w}
                </button>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>F1 comparison at W = {wVal}</p>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={[
                    { name: 'IF Baseline', f1: wData.if_f1 },
                    { name: 'Full Pipeline', f1: wData.pipeline_f1 },
                  ]} barSize={40}>
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 0.5]} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v: unknown) => typeof v === 'number' ? v.toFixed(3) : String(v)} />
                    <Bar dataKey="f1" radius={[6,6,0,0]}>
                      <Cell fill="var(--text-muted)" />
                      <Cell fill="var(--accent)" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-col justify-center gap-3">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>IF F1</p>
                  <p className="text-2xl font-mono font-bold" style={{ color: 'var(--text-muted)' }}>{wData.if_f1.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Full pipeline F1</p>
                  <p className="text-2xl font-mono font-bold" style={{ color: 'var(--accent)' }}>{wData.pipeline_f1.toFixed(3)}</p>
                </div>
                <p className="text-xs p-2 rounded-lg" style={{ background: 'var(--bg-sidebar)', color: 'var(--text-muted)' }}>
                  {wVal === 50 ? 'IF slightly ahead at W=50: its 17 alerts are very close to failure. Pipeline coverage advantage not yet active.' :
                   wVal === 100 ? 'Pipeline leads at W=100 (0.276 vs 0.165): broader coverage of 69 engines dominates.' :
                   'Pipeline F1 rises to 0.374 at W=150 as more of its 69 alerted engines fall inside the window.'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* ── 6. Ablation tables ────────────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-3 mb-3 flex-wrap">
          <h2 className="text-base font-bold" style={{ color: 'var(--text-primary)' }}>Ablation Results</h2>
          <div className="flex gap-1">
            {(['FD001','FD002','FD003','FD004'] as Dataset[]).map(ds => (
              <button
                key={ds}
                onClick={() => setActiveDataset(ds)}
                className="px-2.5 py-1 text-xs font-mono font-semibold rounded-md transition-colors cursor-pointer"
                style={{
                  background: activeDataset === ds ? 'var(--accent)' : 'var(--border)',
                  color: activeDataset === ds ? '#fff' : 'var(--text-muted)',
                }}
              >{ds}</button>
            ))}
          </div>
        </div>
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              Four variants are compared on each dataset. The ablation isolates the contribution of
              each component: replacing the anomaly scorer (Isolation Forest → z-score → causal),
              then adding the full LangGraph pipeline with physics veto and operator feedback.
              Coverage = fraction of engines that received at least one alert before failure.
              Mean lead = average number of cycles between first alert and actual failure cycle.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Variant','Coverage','95% CI','Mean Lead','Median','P','R','F1','95% CI F1','r','Wilcoxon-p','Fisher-p'].map(h => (
                      <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <tr
                      key={row.variant}
                      style={{
                        background: i % 2 === 0 ? 'var(--bg-card)' : 'var(--bg)',
                        borderBottom: '1px solid var(--border)',
                        fontWeight: row.f1 === Math.max(...rows.map(r => r.f1)) ? 700 : undefined,
                      }}
                    >
                      <td className="px-3 py-2 font-medium" style={{ color: 'var(--text-primary)' }}>{row.variant}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--accent)' }}>{(row.coverage * 100).toFixed(0)}%</td>
                      <td className="px-3 py-2 font-mono text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        [{(row.ci_low * 100).toFixed(0)}%–{(row.ci_high * 100).toFixed(0)}%]
                      </td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{row.mean_lead.toFixed(1)}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{row.median_lead.toFixed(1)}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{row.precision.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{row.recall.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono font-bold" style={{ color: row.f1 === Math.max(...rows.map(r => r.f1)) ? 'var(--status-ok)' : 'var(--text-primary)' }}>
                        {row.f1.toFixed(3)}
                      </td>
                      <td className="px-3 py-2 font-mono text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        [{row.f1_ci_low.toFixed(3)}–{row.f1_ci_high.toFixed(3)}]
                      </td>
                      <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{row.r != null ? row.r.toFixed(3) : '—'}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: row.wilcoxon_p === '<0.001' ? 'var(--status-ok)' : 'var(--text-muted)' }}>{row.wilcoxon_p}</td>
                      <td className="px-3 py-2 font-mono" style={{ color: row.fisher_p === '<0.001' ? 'var(--status-ok)' : 'var(--text-muted)' }}>{row.fisher_p}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-[10px] mt-3" style={{ color: 'var(--text-muted)' }}>
              Lead time in cycles. 95% CIs bootstrapped (10 000 resamples). F1 definition: W = 100 cycles actionable window.
              Bold row = highest F1. Green p-values = significant (p &lt; 0.001).
            </p>
          </CardContent>
        </Card>
      </section>

      {/* ── 7. Regime centroids ───────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>FD002 Regime Centroids (KMeans, k = 6)</h2>
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              The six regime clusters learned from FD002 training data correspond to physically
              interpretable operating points. Each cluster has its own set of LinearRegression
              coefficients, one per causal branch. Silhouette score = 0.9971 at k = 6 — the six
              clusters are extremely tight, reflecting that CMAPSS operating conditions are drawn
              from a discrete grid rather than a continuous manifold.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['Cluster','N (train)','Altitude','Mach','TRA','Regime'].map(h => (
                      <th key={h} className="text-left px-4 py-2 font-medium" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {REGIME_CENTROIDS.map((c, i) => {
                    const colors = ['indigo','ok','warn','alert','veto','outline'] as const
                    return (
                      <tr key={c.cluster} style={{ background: i % 2 === 0 ? 'var(--bg-card)' : 'var(--bg)', borderBottom: '1px solid var(--border)' }}>
                        <td className="px-4 py-2 font-mono font-bold" style={{ color: 'var(--text-primary)' }}>{c.cluster}</td>
                        <td className="px-4 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{c.n_train.toLocaleString()}</td>
                        <td className="px-4 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{c.altitude.toFixed(1)}</td>
                        <td className="px-4 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{c.mach.toFixed(2)}</td>
                        <td className="px-4 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{c.tra.toFixed(1)}</td>
                        <td className="px-4 py-2">
                          <Badge variant={colors[i % colors.length]}>{['High-Alt Cruise','Mid-Alt Cruise','Mid-Alt Part-Pwr','Sea-Level','Low-Alt Climb','High-Alt Part-Pwr'][c.cluster]}</Badge>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-[10px] mt-3" style={{ color: 'var(--text-muted)' }}>
              Silhouette score = 0.9971 at k=6 (vs 0.9300 at k=5, 0.9337 at k=7). Within-cluster inertia drops from 1133.9 at k=5 to 0.2 at k=6.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* ── 8. Alpha sweep chart ──────────────────────────────────────────────── */}
      <section>
        <h2 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary)' }}>F1 vs α (FD001)</h2>
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--text-muted)' }}>
              This chart shows the F1 and coverage trade-off as α sweeps from 0 (pure z-score)
              to 1 (pure causal) on FD001. F1 peaks in the 0.55–0.65 range — the blend weight
              used in evaluation is α = 0.60. Coverage is high at low α (z-score catches most engines
              early) but precision falls sharply. The causal component brings precision up at the
              cost of some recall.
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={ALPHA_SWEEP}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="alpha" tick={{ fontSize: 10 }} label={{ value: 'α (causal weight)', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine x={0.60} stroke="var(--accent)" strokeDasharray="4 2" label={{ value: 'α=0.60', fontSize: 9, fill: 'var(--accent)' }} />
                <Line type="monotone" dataKey="f1" stroke="var(--status-ok)" dot={false} strokeWidth={2} name="F1" />
                <Line type="monotone" dataKey="coverage" stroke="var(--accent-cyan)" dot={false} strokeDasharray="4 2" strokeWidth={1.5} name="Coverage" />
                <Legend />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

// ── Interactive Causal DAG SVG ────────────────────────────────────────────────
function CausalDAG() {
  const [hovered, setHovered] = useState<string | null>(null)

  const tooltips: Record<string, string> = {
    op1: 'op_setting_1 — Altitude: sets ambient air density, which controls HPC cooling efficiency.',
    op2: 'op_setting_2 — Mach number: determines rotor tip speed and HPC stage loading.',
    op3: 'op_setting_3 — Throttle Resolver Angle (TRA): controls fuel flow and combustor temperature.',
    s4:  'sensor_4 — LPT Outlet Temp: rises when altitude reduces air density and cooling. Altitude-driven.',
    s11: 'sensor_11 — HPC Pressure Ratio: must track sensor_15 isentropically. Decoupling triggers physics veto.',
    s15: 'sensor_15 — Bypass Ratio: must rise/fall with sensor_11 per isentropic relation.',
    s3:  'sensor_3 — HPC Outlet Temp: responds to fuel flow changes commanded by throttle.',
    s9:  'sensor_9 — Core Speed (N2): governed by combustor temperature — rises when TRA is high.',
  }

  // Layout constants — all coordinates match exactly (no magic numbers)
  const RW = 155, RH = 38, RX = 12          // root nodes
  const LW = 118, LH = 32, LX = 222         // latent nodes, gap=55 from root right
  const SW = 155, SH = 40, SX = 392         // sensor nodes, gap=52 from latent right
  const STEP = 53                             // vertical pitch (matches all three columns)

  // Row centres (top of first node + RH/2 aligns with STEP multiples)
  const rowCY = (row: number) => 64 + row * STEP   // rows 0–4

  const roots = [
    { id: 'op1', row: 0.0, label: 'op_setting_1', sub: 'Altitude' },
    { id: 'op2', row: 2.0, label: 'op_setting_2', sub: 'Mach' },
    { id: 'op3', row: 4.0, label: 'op_setting_3', sub: 'TRA' },
  ]
  const latents = [
    { id: 'l1', row: 0, label: 'AirDensity' },
    { id: 'l2', row: 1, label: 'TipSpeed' },
    { id: 'l3', row: 2, label: 'HPCLoading' },
    { id: 'l4', row: 3, label: 'FuelFlow' },
    { id: 'l5', row: 4, label: 'CombustorT' },
  ]
  const sensors = [
    { id: 's4',  row: 0, label: 'sensor_4',  sub: 'LPT Outlet Temp' },
    { id: 's11', row: 1, label: 'sensor_11', sub: 'HPC Pressure Ratio' },
    { id: 's15', row: 2, label: 'sensor_15', sub: 'Bypass Ratio' },
    { id: 's3',  row: 3, label: 'sensor_3',  sub: 'HPC Outlet Temp' },
    { id: 's9',  row: 4, label: 'sensor_9',  sub: 'Core Speed (N2)' },
  ]

  // op → latent: (right edge of root) → (left edge of latent)
  const opEdges: [number, number, number, number][] = [
    [RX + RW, rowCY(0), LX, rowCY(0)],  // op1 → AirDensity
    [RX + RW, rowCY(2), LX, rowCY(1)],  // op2 → TipSpeed
    [RX + RW, rowCY(2), LX, rowCY(2)],  // op2 → HPCLoading
    [RX + RW, rowCY(4), LX, rowCY(3)],  // op3 → FuelFlow
    [RX + RW, rowCY(4), LX, rowCY(4)],  // op3 → CombustorT
  ]
  // latent → sensor: (right edge of latent) → (left edge of sensor) — 1:1 aligned rows
  const latEdges: [number, number, number, number][] = latents.map(l => [
    LX + LW, rowCY(l.row), SX, rowCY(l.row),
  ])

  const fill = (id: string, type: 'root' | 'sensor' | 'latent') =>
    hovered === id
      ? { fill: type === 'root' ? '#4f46e5' : type === 'sensor' ? '#16a34a' : '#d97706' }
      : { fill: type === 'root' ? '#e0e7ff' : type === 'sensor' ? '#dcfce7' : '#fef3c7' }

  const stroke = (type: 'root' | 'sensor' | 'latent') =>
    type === 'root' ? '#4f46e5' : type === 'sensor' ? '#16a34a' : '#d97706'

  const textFill = (id: string, type: 'root' | 'sensor' | 'latent', muted = false) =>
    hovered === id
      ? (muted ? '#c7d2fe' : '#ffffff')
      : (type === 'root'
          ? (muted ? '#6366f1' : '#3730a3')
          : type === 'sensor'
          ? (muted ? '#4ade80' : '#166534')
          : (muted ? '#b45309' : '#78350f'))

  const VB_W = SX + SW + 12
  const VB_H = rowCY(4) + SH / 2 + 28 + 22   // last node bottom + legend

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full max-w-2xl" style={{ fontFamily: 'system-ui, sans-serif' }}>
        <defs>
          <marker id="dag-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0.5 L0,6.5 L6,3.5 z" fill="#cbd5e1" />
          </marker>
        </defs>

        {/* Edges — op → latent */}
        {opEdges.map(([x1, y1, x2, y2], i) => (
          <line key={`oe${i}`} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke="#cbd5e1" strokeWidth={1.2} markerEnd="url(#dag-arrow)" />
        ))}
        {/* Edges — latent → sensor */}
        {latEdges.map(([x1, y1, x2, y2], i) => (
          <line key={`le${i}`} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke="#cbd5e1" strokeWidth={1.2} markerEnd="url(#dag-arrow)" />
        ))}

        {/* Root nodes */}
        {roots.map(n => {
          const cy = rowCY(n.row), x = RX, y = cy - RH / 2
          return (
            <g key={n.id} onMouseEnter={() => setHovered(n.id)} onMouseLeave={() => setHovered(null)} style={{ cursor: 'pointer' }}>
              <rect x={x} y={y} width={RW} height={RH} rx={9} {...fill(n.id, 'root')} stroke={stroke('root')} strokeWidth={1.5} />
              <text x={x + RW / 2} y={y + 14} textAnchor="middle" fontSize={9.5} fontWeight={700} fill={textFill(n.id, 'root')}>{n.label}</text>
              <text x={x + RW / 2} y={y + 27} textAnchor="middle" fontSize={9} fill={textFill(n.id, 'root', true)}>{n.sub}</text>
            </g>
          )
        })}

        {/* Latent nodes */}
        {latents.map(n => {
          const cy = rowCY(n.row), x = LX, y = cy - LH / 2
          return (
            <g key={n.id}>
              <rect x={x} y={y} width={LW} height={LH} rx={7} {...fill(n.id, 'latent')} stroke={stroke('latent')} strokeWidth={1} />
              <text x={x + LW / 2} y={y + LH / 2 + 4} textAnchor="middle" fontSize={9} fontWeight={600} fill={textFill(n.id, 'latent')}>{n.label}</text>
            </g>
          )
        })}

        {/* Sensor nodes */}
        {sensors.map(n => {
          const cy = rowCY(n.row), x = SX, y = cy - SH / 2
          return (
            <g key={n.id} onMouseEnter={() => setHovered(n.id)} onMouseLeave={() => setHovered(null)} style={{ cursor: 'pointer' }}>
              <rect x={x} y={y} width={SW} height={SH} rx={9} {...fill(n.id, 'sensor')} stroke={stroke('sensor')} strokeWidth={1.5} />
              <text x={x + SW / 2} y={y + 15} textAnchor="middle" fontSize={9.5} fontWeight={700} fill={textFill(n.id, 'sensor')}>{n.label}</text>
              <text x={x + SW / 2} y={y + 29} textAnchor="middle" fontSize={8.5} fill={textFill(n.id, 'sensor', true)}>{n.sub}</text>
            </g>
          )
        })}

        {/* Legend */}
        {[
          { x: 12, fill: '#e0e7ff', stroke: '#4f46e5', label: 'Op settings (root)' },
          { x: 175, fill: '#fef3c7', stroke: '#d97706', label: 'Latent variables' },
          { x: 330, fill: '#dcfce7', stroke: '#16a34a', label: 'Observed sensors' },
        ].map(l => {
          const ly = VB_H - 14
          return (
            <g key={l.label}>
              <rect x={l.x} y={ly} width={11} height={11} rx={3} fill={l.fill} stroke={l.stroke} strokeWidth={1} />
              <text x={l.x + 16} y={ly + 9} fontSize={8.5} fill="var(--text-muted)">{l.label}</text>
            </g>
          )
        })}
      </svg>

      {hovered && tooltips[hovered] && (
        <div
          className="absolute bottom-2 left-2 text-xs p-2.5 rounded-lg shadow-sm max-w-xs leading-snug"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
        >
          {tooltips[hovered]}
        </div>
      )}
    </div>
  )
}
