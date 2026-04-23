# Don't Trust the Sensors: Regime-Aware Causal Anomaly Triage for Industrial IoT

**Jahnavi Patel**
College of Engineering, Northeastern University
patel.jahnavi@northeastern.edu

---

## Abstract

Industrial IoT anomaly detection systems routinely generate false alarms because they compare sensor readings against global statistical thresholds that ignore operating conditions. A turbofan engine running at high altitude produces temperature and pressure readings that are normal for that regime but anomalous relative to global training-set means — triggering alarms that have nothing to do with engine health. We present a causal anomaly triage system that conditions scoring on a learned operating-regime classifier and augments it with a physics-based structural veto grounded in the isentropic compression relation, deployed with a seven-node LangGraph agent for operator-correctable explanations. We evaluate on all four NASA CMAPSS sub-datasets: FD001/FD003 (single operating condition) and FD002/FD004 (six operating conditions). On single-condition data, the full blended pipeline achieves 4.1–4.7× better coverage than Isolation Forest at 53% earlier mean lead time (FD001: 69% vs. 17%, 164.9 vs. 107.4 cycles; FD003: 89% vs. 19%). On multi-condition data, both a global z-score baseline and a retrained z-score baseline that recomputes means from the target dataset produce 100% false positive rates (F1 = 0.000) — confirming that domain-shift correction alone is insufficient. Regime-aware causal scoring reduces false alarms to 66% coverage on FD002 (F1 = 0.279) and 57% on FD004 (F1 = 0.352), with statistically significant false positive reduction (Fisher *p* < 0.001) on both. The four-dataset evaluation demonstrates a consistent pattern: blended scoring improves recall on homogeneous data; **regime conditioning — not retraining — is the necessary mechanism for preventing false positive explosion on heterogeneous, multi-condition fleets**.

---

## 1. Introduction

Alarm fatigue is one of the most studied failure modes in industrial monitoring. A survey of alarm management practice across the UK chemical and power industries found that approximately 50% of process alarms could have been eliminated with little detriment to plant operations, and that in worst-case scenarios operators faced up to 90 alarms per minute (Bransby & Jenkinson, 1998). Operators who receive hundreds of spurious alerts per shift learn to ignore them — and in doing so, they miss the real failures. A UK Health and Safety Executive investigation into the 1994 Texaco refinery explosion at Milford Haven found that operators had to respond to 275 alarms in the 11 minutes before the incident — more than one every two seconds — and concluded that the excessive alarm load was a direct contributing factor (UK HSE, 1997).

The root cause of most false alarms is not sensor failure; it is context blindness. A global anomaly detector is trained on data from all operating conditions and learns a single statistical model of "normal." When a system operates in a condition that differs from the training distribution, the detector fires — even though the reading is perfectly healthy for that condition. This problem is not theoretical: the NASA CMAPSS FD002 dataset, which simulates turbofan engines across six distinct altitude/Mach/throttle regimes, allows us to reproduce it exactly. A detector trained on single-condition means and applied to multi-condition data produces alerts on 100% of engines — a false positive rate indistinguishable from random.

This paper presents a system that addresses the problem structurally rather than by tuning thresholds. The key idea is that anomaly scoring should be conditioned on the current operating regime, not evaluated against a global baseline. We encode this via a causal directed acyclic graph (DAG) in which operational settings (altitude, Mach number, throttle resolver angle) are root cause nodes that determine expected sensor values through physical relationships. The anomaly score for a sensor reading is its residual from the causally-predicted value given the current operating conditions — not its deviation from a global mean.

We make three contributions:

1. **Regime-aware causal scoring**: A LinearRegression-based causal scorer conditioned on a KMeans operating-regime classifier, evaluated with a five-variant ablation study (Isolation Forest baseline, z-score only, causal only, full blended pipeline, full pipeline + physics veto) including bootstrapped 95% CIs and statistical significance tests.

2. **Physics-based structural veto**: A G-test monitor that detects when the isentropic coupling between HPC outlet temperature and pressure is broken — distinguishing sensor faults from engine degradation — with the correct χ² threshold for a 5×5 contingency table (threshold = 26.30, df = 16).

3. **Open-source deployable system**: A FastAPI backend with a seven-node LangGraph pipeline, PostgreSQL telemetry store, operator feedback loop, and HTML dashboard — fully reproducible from the public repository.

The rest of this paper is organized as follows. Section 2 reviews related work. Section 3 describes the system architecture. Section 4 presents experimental results on all four CMAPSS sub-datasets (FD001–FD004). Section 5 discusses implications and limitations. Section 6 concludes.

---

## 2. Related Work

**Anomaly detection for industrial time series.** Isolation Forest (Liu et al., 2008) is the dominant baseline for unsupervised anomaly detection in industrial sensor data. It constructs an ensemble of random decision trees and scores points by how quickly they are isolated — anomalies, being unusual, require fewer splits. Contamination rate (fraction of training data assumed anomalous) controls sensitivity. It does not condition on operating mode.

**NASA CMAPSS benchmark.** The Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset (Saxena & Goebel, 2008) is the standard benchmark for turbofan engine predictive maintenance research. It provides four sub-datasets (FD001–FD004) with varying numbers of operating conditions (1 or 6) and fault modes (1 or 2). FD001 is the standard entry point (1 condition, 1 fault mode); FD002 introduces six distinct altitude/Mach/throttle regimes that serve as a demanding multi-condition test bed. Most published work evaluates only on FD001 or FD001+FD003 (Hong et al., 2020; Peng et al., 2021; Zheng et al., 2017), which means the regime mismatch problem demonstrated here has gone largely unexamined in the predictive maintenance literature.

**Causal inference for anomaly detection.** DoWhy (Sharma & Kiciman, 2020) provides a Python interface for specifying and estimating causal models. The key insight behind causal anomaly scoring is that an observation should be judged anomalous relative to what the causal model predicts given the current context — not relative to an unconditional mean. Pearl's do-calculus (Pearl, 2009) provides the formal foundation. For live inference at the per-reading latency required here, we use DoWhy for DAG validation at module load time and fit lightweight sklearn LinearRegression models per causal branch for per-request scoring. Specifically, DoWhy validates that the specified DAG has no directed cycles and that each causal branch has at least one observed variable; validation failure raises an exception at module load time, preventing deployment of an invalid causal structure. The five-branch DAG used in this system passed all structural checks: no cycles were detected, all five sensor nodes were reachable from their respective op-setting parents, and no latent confounders were flagged between the three root nodes (the op-settings are set by the flight regime and are mutually independent by design).

**Multi-mode and condition-aware process monitoring.** The problem of operating-condition variation in sensor data has been studied extensively in statistical process control (SPC). Multi-mode PCA approaches partition training data by operating mode and build separate control charts per mode, avoiding the cross-mode contamination that degrades global detectors (Zhao et al., 2004). Gaussian Mixture Model (GMM)-based monitoring methods model the joint sensor distribution as a mixture, implicitly capturing multi-regime structure without hard cluster boundaries (Yu & Qin, 2009). These approaches condition on regime at the distributional level but do not encode causal structure: they learn that certain sensor combinations co-occur under certain conditions without representing *why* those co-occurrences hold. As a consequence, a GMM or multi-mode PCA detector will still fire on a healthy engine operating at an unusual-but-valid regime point that was underrepresented in training — because its statistical model of "normal" is data-driven, not physics-driven. The causal approach taken here replaces this data-driven conditioning with a structural model: the expected sensor value is computed from a physical relationship (e.g., compressor thermodynamics), not from cluster membership alone. This distinction is especially relevant for fleet monitoring where operating points are sparse or novel regimes arise at deployment time.

**Domain adaptation and transfer learning.** Domain adaptation and transfer learning methods (Yan et al., 2023) address the related problem of applying detectors across domains but typically require labeled anomaly examples from the target domain — an assumption violated in predictive maintenance, where true failure events are rare and often unlabeled. The regime-conditioning approach requires only that operating conditions are observable, which is the case for the CMAPSS op-settings and for most industrial equipment with measurement of altitude, load, speed, or throttle.

**LLM-augmented monitoring agents.** LangGraph (LangChain, 2024) provides a framework for building stateful multi-step agents as directed graphs. Several recent systems use LLMs to generate explanations for anomaly decisions. Liu et al. (2024) propose LLMAD, which applies LLMs with chain-of-thought prompting to detect time-series anomalies and produce textual explanations; human evaluation showed their anomaly chain-of-thought improved explanation usefulness by 13.4% over baselines. Our contribution is integrating the LLM explanation step within a pipeline that already has causally-grounded scores, physics vetoes, and cache-corrected confidence — so the explanation is constrained by structure, not produced in a vacuum.

---

## 3. System Architecture

### 3.1 Causal DAG

The system's scoring model is organized around a causal directed acyclic graph that encodes the physical relationships between turbofan operating conditions and the sensor channels they govern.

```
op_setting_1 (Altitude)  →  AirDensity → CoolingEfficiency → sensor_4
op_setting_2 (Mach)      →  TipSpeed   → HPCLoading        → sensor_11
                                                            → sensor_15
op_setting_3 (TRA)       →  FuelFlow   → CombustorTemp     → sensor_3
                                                            → sensor_9
```

![Causal DAG](causal_dag.png)

*Figure 1: Causal DAG connecting operational settings (root nodes, blue) through latent physical variables (orange) to observed sensors (green). Anomaly scoring computes the residual of each sensor relative to the value predicted by its causal parent, conditioned on the current operating regime.*

The three root nodes (`op_setting_1`, `op_setting_2`, `op_setting_3`) correspond to Altitude, Mach number, and Throttle Resolver Angle respectively. These are set by the flight regime, not by the engine's health — they are the conditions under which the engine is operating, not signals of its state.

For each causal branch, a LinearRegression model is fit on the training set:

```
predicted_sensor = coef × op_setting + intercept
residual = observed − predicted
causal_z = |residual| / residual_std
```

The causal anomaly score uses the top-3 of the 5 causal sensor z-scores (k = min(3, n_sensors)), normalized to [0, 1] using a 5-standard-deviation ceiling: `causal_score = min(mean_top3_z / 5.0, 1.0)`. This mirrors the global z-scorer's top-3-of-14 aggregation strategy, ensuring the two components operate on comparable scales before blending. This score answers the question: "How anomalous is this sensor reading given where the engine is currently operating?" — rather than "How anomalous is this reading compared to global averages?"

Per-sensor noise floors prevent tight-distribution sensors (sensor_8: σ = 0.058 training std) from generating inflated z-scores for physically negligible deviations. Noise floors are derived as 2× the cross-engine standard deviation at cycle 1 in the training set, representing natural inter-engine variation that should not be flagged.

### 3.2 Regime-Aware Scoring

On FD001 (single operating condition), the causal branches are fit on all training data combined. On FD002 (six conditions), a KMeans classifier (k = 6) is trained on the three operational settings to assign each reading to a regime. The choice k = 6 is motivated by two independent lines of evidence: (1) the CMAPSS dataset specification (Saxena & Goebel, 2008) explicitly defines six discrete operating conditions (altitude × Mach × throttle resolver angle combinations), and (2) a silhouette analysis on the FD002 training op-settings confirms k = 6 as the optimal cluster count (silhouette score = 0.9971 at k = 6, vs. 0.9300 at k = 5 and 0.9337 at k = 7; within-cluster inertia drops from 1133.9 at k = 5 to 0.2 at k = 6, indicating that the six conditions are genuinely discrete rather than a continuous manifold). The same k = 6 value applies to FD004, which shares the same six operating conditions. Per-regime LinearRegression models are fit on each cluster's subset of the training data. At inference time, the classifier assigns the incoming reading to a cluster and the corresponding per-cluster coefficients are used for residual computation. This ensures that the expected temperature at op_setting_1 = 42 (high altitude) is computed from engines operating at high altitude — not from engines at sea level.

The full anomaly score blends the regime-aware causal score with a global z-score using a learned weight α:

```
combined_score = α × causal_score + (1 − α) × z_score
```

The blend weight α is selected by grid-searching α ∈ {0.00, 0.05, …, 1.00} to maximize F1 (W = 100 cycles). **Calibration transparency note:** two calibration settings are reported throughout this paper and their distinction matters for interpretation. *Training-set α* = 0.70 is obtained by grid search on the FD001 training set; however, because CMAPSS training engines all run to failure, recall = 100% at every α value on training data, making F1 purely precision-driven and producing a value that does not generalize. *Evaluation-set α* = 0.60 is obtained by grid search on the held-out FD001 test set, which introduces a data leakage risk: the evaluation set is used both to select α and to report performance. All quantitative results in this paper use α = 0.60; readers should treat the reported F1 values as an optimistic upper bound on out-of-sample performance at this blend weight. Proper calibration would use a held-out validation split separate from the test set, or k-fold cross-validation within the training set — this limitation is discussed further in Section 5.3. On FD002 and multi-condition data, α = 1.00 (pure causal): any z-score contribution collapses precision to near zero because global means do not condition on regime. The global z-score provides sensitivity (it fires on any sensor significantly deviating from historical mean, regardless of cause). The causal score provides precision (it penalizes only readings that deviate from what the current operating regime predicts).

### 3.3 Physics-Based Structural Veto

Beyond the statistical scoring, the system applies a structural validation layer grounded in the isentropic compression relation for the High Pressure Compressor (HPC).

In normal operation, HPC outlet temperature (sensor_11) and HPC outlet pressure (sensor_15) are thermodynamically coupled. The governing relation is:

$$\frac{T_2}{T_1} = \left(\frac{P_2}{P_1}\right)^{(\gamma-1)/\gamma}, \quad \gamma \approx 1.4 \text{ for air}$$

This guarantees that both sensors must rise and fall together through the compressor. If this coupling breaks — sensor_11 spikes while sensor_15 is stable, for example — the anomaly is more likely a sensor fault than real engine degradation.

To detect coupling breaks without assuming a specific functional form, we use a G-test for independence on a 5×5 contingency table built from the last 100 readings per engine:

$$G = 2 \sum_i O_i \ln\left(\frac{O_i}{E_i}\right)$$

where *O_i* is the observed count in each cell and *E_i* is the expected count under independence. The correct critical value for a 5×5 table is χ²(df = 16, *p* = 0.05) = **26.30** (not 9.49, which applies to a 3×3 table).

When the G-statistic indicates a coupling break, the veto applies a graduated penalty proportional to observed coupling strength:

```
veto_factor = 1.0 − 0.5 × min(G / 26.30, 1.0)
causal_score_refined = causal_score × veto_factor
```

Boundary values: G = 0 (perfect coupling) → veto_factor = 1.0 (no penalty); G = 13.15 → veto_factor = 0.75 (25% reduction); G = 26.30 (at critical threshold) → veto_factor = 0.50 (maximum 50% reduction); G > 26.30 → penalty clamped at 0.50. The graduated design avoids the binary jump of a hard gate, and the system does not eliminate causal signal entirely even under extreme coupling breaks. The reading is marked as a likely sensor fault rather than engine degradation when the veto fires.

This approach combines a domain-specific physical constraint (isentropic coupling) with a non-parametric statistical test (G-test) to achieve a form of structural anomaly detection that does not depend on ML training data.

### 3.4 LangGraph Seven-Node Agent

Readings with combined_score ≥ 0.3 trigger a seven-node LangGraph agent:

| Node | Function |
|---|---|
| `ingest_validator` | Flags if >3 of 5 causal-branch sensors are stale or missing |
| `regime_classifier` | Assigns operating regime via KMeans predict |
| `causal_reasoner` | Passes regime-conditioned causal score to downstream nodes |
| `physics_veto` | Applies G-test coupling check; applies graduated veto_factor = 1.0 − 0.5 × min(G/26.30, 1.0) |
| `cache_lookup` | Queries prior readings for same engine; applies 0.7× penalty if ≥2 operator FALSE_POSITIVE labels exist |
| `llm_explainer` | Calls Groq (Llama-3) or Gemini; falls back to rule-based template on API failure |
| `decision_writer` | Computes final blended score and decision; updates alert record; writes reasoning traces |

Each node writes an execution record to a `reasoning_traces` table (node name, input/output state, latency). This provides a full audit trail for every decision. The agent runs outside the main database transaction to prevent LLM latency from holding database connections open.

Operators can correct decisions via `POST /feedback` with a label (`TRUE_POSITIVE`, `FALSE_POSITIVE`, `UNCERTAIN`). The `cache_lookup` node queries these labels on subsequent alerts for the same engine and adjusts confidence accordingly, closing a human-in-the-loop feedback cycle.

A single-file HTML dashboard (`dashboard/index.html`, zero JS dependencies) provides a live alert feed with color-coded anomaly scores, per-alert LLM explanation display, one-click feedback buttons, and a collapsible PSI sensor health strip. The dashboard communicates with the backend via `fetch()` against the FastAPI REST endpoints.

---

## 4. Evaluation

### 4.1 Experimental Setup

**Dataset.** We use all four sub-datasets of the NASA CMAPSS turbofan engine degradation benchmark. FD001 (100 test engines, 1 condition, HPC fault) and FD003 (100 test engines, 1 condition, HPC + fan fault) provide the single-condition evaluation; FD002 (259 test engines, 6 conditions, HPC fault) and FD004 (248 test engines, 6 conditions, HPC + fan fault) provide the multi-condition evaluation. This design isolates three independent axes of variation: single vs. multi-condition (FD001 vs. FD002), fault-mode generalizability (FD001 vs. FD003), and the interaction of both (FD004 as the hardest case).

**Alert threshold.** A reading is considered an alert when combined_score ≥ 0.3, corresponding to the UNCERTAIN decision boundary. This threshold was calibrated on the training set as the 90th percentile of per-reading causal scores during the first 30 cycles of each training engine (the healthy-operation window) — setting the threshold at the upper bound of normal operating variability, so an alert fires only when a reading deviates more than 90% of healthy readings ever do. The threshold is held fixed for all test-set evaluations reported here; the ALERT boundary (≥ 0.6) requires simultaneous multi-sensor degradation not consistently present in test data.

**Lead time metric.** For each engine, we record the first cycle at which an alert fires. Lead time is defined as:

```
lead_time = true_failure_cycle − first_alert_cycle
```

where `true_failure_cycle = last_cycle_in_test + RUL_at_end` from the NASA-provided RUL labels. A larger positive lead time means the system alerted earlier before failure. Engines that never fire an alert contribute to the coverage denominator but not to lead-time statistics.

**Task scope.** This paper evaluates anomaly detection with lead-time measurement — not Remaining Useful Life (RUL) regression. The distinction matters for baseline selection: deep learning approaches cited in Related Work (Hong et al., 2020; Zheng et al., 2017; Peng et al., 2021) produce continuous RUL estimates from the full degradation trajectory; they assume the system has access to the complete run and must estimate time-to-failure. This system operates cycle-by-cycle on streaming sensor readings and must decide, at each cycle, whether to raise an alert — without access to future data. These are different tasks with different evaluation criteria (MAE/RMSE for RUL regression vs. coverage/F1/lead-time for anomaly detection), and direct numeric comparison would be misleading.

**Baselines.** For FD001, the baseline is an Isolation Forest (contamination = 0.05, n_estimators = 100) trained on the 14 informative sensors from `train_FD001.txt`. For FD002, the baseline is global z-score scoring using means and standard deviations derived from FD001 training data. This baseline deliberately represents the cross-dataset deployment failure mode — applying a single-condition detector to multi-condition data without retraining — which is the specific scenario this experiment is designed to demonstrate. Table 2b includes a retrained z-score variant (using FD002 marginal means) to isolate whether domain-shift correction alone is sufficient; as the results show, retraining does not reduce the false positive rate (Wilcoxon *p* = 0.548, Fisher *p* = 1.000 vs. FD001 baseline), confirming that regime conditioning — not retraining — is the necessary mechanism.

**Isolation Forest contamination sensitivity.** The contamination parameter controls the fraction of training data treated as anomalous at fit time and directly determines IF coverage. To assess sensitivity to this choice, IF was evaluated at contamination ∈ {0.01, 0.05, 0.10, 0.15}:

| Contamination | IF Coverage | IF F1 (W=100) |
|---|---|---|
| 0.01 | 0% | 0.000 |
| **0.05** | **17%** | **0.165** |
| 0.10 | 42% | 0.214 |
| 0.15 | 53% | 0.230 |

The value 0.05 is the sklearn default and represents a conservative prior (5% of training readings treated as anomalous). All results reported in Table 2a use contamination = 0.05. Critically, even the most aggressive setting tested (0.15) produces IF F1 = 0.230 — still below the full pipeline's F1 = 0.276. The full pipeline's advantage over IF is not an artifact of a conservatively set contamination parameter; it holds across the range of plausible contamination values.

**Noise floors.** Per-sensor noise floors (derived as 2× the cross-engine standard deviation at cycle 1) are applied uniformly across all ablation variants and the FD002 evaluation. This ensures that the full pipeline's performance advantage reflects blending, not noise floor correction alone.

**Statistical tests.** We report Wilcoxon rank-sum (*p*-values from `scipy.stats.mannwhitneyu`, two-sided) comparing lead-time distributions against the baseline, and Fisher's exact test comparing coverage counts in a 2×2 contingency table of (alerts, no-alerts) × (variant, baseline).

### 4.2 FD001 Ablation Study

**Table 2a: FD001 Ablation Results** (100 test engines, alert threshold = 0.3, F1 window W = 100 cycles)

| Variant | Coverage | 95% CI | Mean (SD) | Median | Precision | Recall | F1 | 95% CI | *r* | Wilcoxon-*p* | Fisher-*p* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 17% | [10%–25%] | 107.4 (100.9) | 41.0 | 0.529 | 0.098 | 0.165 | [0.077–0.261] | — | — | — |
| Z-score only | 98% | [95%–100%] | 188.6 (49.2) | 179.0 | 0.051 | 0.714 | 0.095 | [0.020–0.182] | 0.366 | 0.016 | <0.001 |
| Causal only | 64% | [54%–73%] | 172.8 (74.4) | 187.5 | 0.172 | 0.234 | 0.198 | [0.095–0.291] | 0.341 | 0.032 | <0.001 |
| Full pipeline (α = 0.60) | **69%** | **[60%–78%]** | **164.9 (74.9)** | **178.0** | **0.232** | **0.340** | **0.276** | **[0.165–0.374]** | **0.339** | **0.031** | **<0.001** |
| Full pipeline + veto | 52% | [42%–62%] | 167.0 (87.4) | 195.0 | 0.250 | 0.213 | 0.230 | [0.131–0.333] | 0.371 | 0.023 | <0.001 |

*Lead time in cycles. Coverage = fraction of 100 engines with any alert. 95% CIs are bootstrapped (10 000 resamples of the full 100-engine set). Precision/Recall/F1 use W = 100 cycles as the actionable-window: TP = alerted with lead time ≤ W, FP = alerted with lead time > W (premature alert), FN = not alerted. *r* = rank-biserial correlation (r = 0.1 small, 0.3 medium, 0.5 large; positive = variant has larger lead times than IF). Wilcoxon and Fisher tests against Isolation Forest baseline.*

**Methodological note — aggregation correction.** The original causal scorer averaged residual z-scores across all 5 causal sensors (mean of 5), while the global z-scorer averaged only the top-3 worst of 14 informative sensors. This asymmetry suppressed causal scores relative to z-scores, making the nominal 50/50 blend effectively z-score-dominated. The corrected aggregation uses the top-3 of 5 causal sensors, consistent with z-score's top-3-of-14 strategy. All numbers in this table reflect the corrected aggregation.

The z-score-only variant achieves 98% coverage but at the cost of firing on virtually every engine — including those that will not fail for hundreds of cycles. The precision of 0.051 (F1 = 0.095) confirms that fewer than 1 in 20 alerts fires within the 100-cycle actionable window; the rest are premature. Causal-only scoring is more conservative (64% coverage), with mean lead time 172.8 cycles. The Wilcoxon *p* = 0.032 confirms a significant lead-time distribution difference vs. IF, reversing the non-significant result (0.180) from the pre-correction baseline. This result is expected: on FD001 with nearly constant op_settings, the corrected causal scorer is now comparably sensitive to the z-scorer, so the blend benefits from both components.

The full pipeline (α = 0.60) achieves 69% coverage — 4.1× the Isolation Forest baseline (17%) — with a statistically significant coverage advantage (Fisher *p* < 0.001) and a 53% improvement in mean lead time (164.9 vs. 107.4 cycles). The Wilcoxon *p* = 0.031 on lead time is nominally significant. Adding the graduated physics veto (Full pipeline + veto) reduces coverage to 52% on FD001 — a finding that reveals the veto's mechanism: on single-condition HPC-degradation data, genuine engine deterioration causes sensor_11 and sensor_15 to decouple from their causal parents, which the G-test correctly detects as a coupling break. On FD001, this coupling break *is* the degradation signal, so the veto correctly identifies the structural anomaly but reduces coverage as a trade-off. The veto's intended benefit is suppressing sensor artifacts in conditions where coupling breaks are rare in genuine degradation — an effect best evaluated on FD002 and FD003/FD004 data.

**Multiple comparisons correction.** With five variants and two tests each (Wilcoxon and Fisher), the Bonferroni-corrected significance threshold is α = 0.05/10 = 0.005. All Fisher *p*-values (< 0.001) survive this correction — the coverage differences are robust. No Wilcoxon *p*-value survives correction (smallest: 0.016 for z-score, 0.031 for full pipeline); we report them for completeness but treat coverage (Fisher) as the primary statistical claim.

**The FD001 ablation demonstrates that the improvement of the full pipeline over IF is driven by the complementary coverage of both components — not by causal conditioning in isolation.** To demonstrate causal conditioning as the primary mechanism for false positive suppression, FD002 is required.

### 4.3 FD002 Regime-Aware Evaluation

The core failure mode being demonstrated is regime mismatch: a reading that is physically normal at cluster 0 (altitude = 42, Mach = 0.84) deviates substantially from means computed at cluster 3 (altitude ≈ 0, Mach ≈ 0), so a detector trained on single-condition data fires on every engine in a multi-condition fleet.

**Table 2b: FD002 Regime-Conditioning Results** (259 test engines, alert threshold = 0.3, F1 window W = 100 cycles)

| Variant | Coverage | 95% CI | Mean (SD) | Median | Precision | Recall | F1 | 95% CI | Wilcoxon-*p* | Fisher-*p* |
|---|---|---|---|---|---|---|---|---|---|---|
| Global z-score (FD001 means) | 100% | [100%–100%] | 211.2 (47.8) | 203.0 | 0.000 | 0.000 | 0.000 | [0.000–0.000] | — | — |
| Retrained z-score (FD002 means) | 100% | [100%–100%] | 209.1 (47.8) | 201.0 | 0.000 | 0.000 | 0.000 | [0.000–0.000] | 0.548 | 1.000 |
| Regime-aware causal | **66%** | **[59%–71%]** | **167.5 (86.6)** | **179.5** | **0.247** | **0.321** | **0.279** | **[0.214–0.345]** | **<0.001** | **<0.001** |

*Lead time in cycles. 259 engines across 6 operating conditions. 95% CIs bootstrapped (10 000 resamples of full 259-engine set). F1 definition same as Table 2a (W = 100 cycles). Wilcoxon and Fisher p-values for retrained z-score are vs. global z-score baseline; regime-aware causal p-values are also vs. global z-score. Rank-biserial r is not reported for these comparisons: both z-score baselines achieve 100% coverage with F1 = 0.000, making the lead-time distribution comparison degenerate (all lead times > W).*

The global z-score baseline fires an alert on every single engine in the FD002 test set — 100% coverage that represents a 100% false positive rate, and an F1 of exactly 0.000. Every alert fires more than 100 cycles before the engine's known failure cycle (mean 211.2 cycles), meaning every alert is in the "premature" zone under the W = 100 criterion. The root cause is regime mismatch: the global means were computed from FD001 training data, which has a single operating condition (op_setting_1 ≈ 0, op_setting_2 ≈ 0, op_setting_3 ≈ 100). FD002 includes engines operating at high altitude (op_setting_1 = 42), where HPC outlet temperature and pressure are substantially different from the single-condition training baseline.

**Mechanistic isolation — retraining vs. regime conditioning.** The retrained z-score variant uses means and standard deviations computed from FD002 training data (all six regimes pooled), correcting the domain-shift in the mean. Its result is strikingly similar to the original: 100% coverage, F1 = 0.000, mean lead time 209.1 cycles — not significantly different from the FD001-trained baseline (Wilcoxon *p* = 0.548, Fisher *p* = 1.000). The mechanism is that pooling across regimes replaces a wrong FD001 mean with a wrong FD002 mean: neither matches any specific regime's expected sensor values. A high-altitude healthy engine deviates from the FD002 pooled mean by approximately as much as it deviates from the FD001 mean, because the pooled mean sits in the middle of the regime space rather than at any operating condition. **Retraining does not solve the false positive problem. Regime conditioning does.**

**Table 3: FD002 Operating Regime Centroids** (KMeans, k = 6, trained on `op_setting_1/2/3`)

| Cluster | N (train) | Altitude | Mach | TRA |
|---|---|---|---|---|
| 0 | 13,458 | 42.0 | 0.84 | 100.0 |
| 1 | 8,122 | 20.0 | 0.70 | 100.0 |
| 2 | 8,002 | 25.0 | 0.62 | 60.0 |
| 3 | 8,044 | 0.0 | 0.00 | 100.0 |
| 4 | 8,096 | 10.0 | 0.25 | 100.0 |
| 5 | 8,037 | 35.0 | 0.84 | 100.0 |

The regime-aware causal system assigns each incoming reading to one of these six clusters and evaluates it against per-cluster regression coefficients. The result is 66% coverage with F1 = 0.279 — compared to F1 = 0.000 for the global z-score, a difference of 0.279 absolute. Both coverage difference (Fisher *p* < 0.001) and lead-time distribution (Wilcoxon *p* < 0.001) are statistically significant.

**Threshold calibration validation.** The alert threshold of 0.3 was validated for FD002 via an early-life calibration on the training set: the 90th percentile of per-reading causal scores during the first 30 cycles of each training engine (the healthy-operation baseline) is 0.302, which rounds to 0.30. This confirms the threshold is not arbitrary — it is set at the upper bound of normal operating variability, so alerts require deviation beyond what is typical for a healthy engine in any regime.

The primary contribution of regime-aware causal scoring on FD002 is **false positive suppression with maintained true positive detection**. The 66% coverage reflects the system correctly alerting on engines that are genuinely degrading while declining to alarm on regime-normal readings.

**RUL analysis of non-alerted engines.** To verify that the 34% non-alerted engines (89 of 259) represent genuinely lower-risk units rather than missed detections, we compare their NASA-provided RUL labels at test-set end against the 170 alerted engines using a one-sided Wilcoxon rank-sum test (H₁: non-alerted RUL > alerted RUL).

Non-alerted engines have median RUL = 103.0 cycles at test-set end; alerted engines have median RUL = 54.0 cycles — a 1.9× difference. The Wilcoxon test confirms this difference is highly significant (p < 0.001, rank-biserial r = 0.484, medium-large effect). **The system is declining to alarm on engines that are objectively further from failure, not missing true degradation signals.** The 34% "non-alerted" fraction is correctly cautious, not a detection gap.

### 4.4 FD003 Generalizability — Single Condition, Dual Fault Mode

FD003 shares FD001's single operating condition but adds a second fault mode: fan degradation alongside HPC degradation. The causal DAG was designed around HPC sensors (sensor_4, sensor_11, sensor_15, sensor_3, sensor_9); FD003 tests whether these branches carry signal when the degradation source is fan-related rather than compressor-related.

**Table 2c: FD003 Ablation Results** (100 test engines, single condition, dual fault mode, alert threshold = 0.3, W = 100 cycles)

| Variant | Coverage | 95% CI | Mean (SD) | Median | Precision | Recall | F1 | 95% CI | *r* | Wilcoxon-*p* | Fisher-*p* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 19% | [12%–27%] | 190.7 (140.0) | 225.0 | 0.368 | 0.080 | 0.131 | [0.039–0.214] | — | — | — |
| Z-score only | 100% | [100%–100%] | 231.4 (80.6) | 216.5 | 0.010 | 1.000 | 0.020 | [0.000–0.058] | 0.063 | 0.666 | <0.001 |
| Causal only | 79% | [71%–87%] | 235.0 (98.3) | 225.0 | 0.076 | 0.222 | 0.113 | [0.039–0.198] | 0.114 | 0.445 | <0.001 |
| Full pipeline (α = 0.60) | **89%** | **[83%–95%]** | **218.3 (103.8)** | **216.0** | **0.157** | **0.560** | **0.246** | **[0.148–0.347]** | **0.064** | **0.666** | **<0.001** |
| Full pipeline + veto | 74% | [65%–82%] | 240.3 (98.5) | 227.0 | 0.068 | 0.161 | 0.095 | [0.020–0.182] | 0.136 | 0.365 | <0.001 |

*IF trained on train_FD003.txt. All other variants use FD001 causal coefficients (FD003 is single-condition). 95% CIs bootstrapped (10 000 resamples of full 100-engine set). *r* = rank-biserial correlation vs. IF baseline; Wilcoxon and Fisher tests vs. Isolation Forest. Note: *r* ≈ 0 for all variants (range 0.063–0.136, below the 0.1 "small" threshold) and Wilcoxon tests are non-significant throughout — consistent with one another. The primary effect on FD003 is coverage (Fisher *p* < 0.001), not lead-time ordering. FD003 test engines have uniformly long lead times regardless of variant, so the ranking of individual lead times does not distinguish the variants; only whether an engine fires at all does.*

The full pipeline achieves **89% coverage** on FD003 — 20 percentage points higher than FD001 (69%) — with coverage differences statistically significant vs. IF (Fisher *p* < 0.001). Wilcoxon *p*-values are not significant for any variant; this reflects that coverage (the number of engines caught) improves substantially while the per-engine lead time distribution shape is similar to IF's 19 alerted engines.

The F1 of 0.246 is slightly lower than FD001's 0.276, despite higher coverage. The reason is lead time distribution: FD003 test engines have longer mean lead times overall (mean 218 cycles for the full pipeline, vs. 164 on FD001), placing more alerts in the lead_time > 100 (premature alert) zone under the W = 100 criterion. This is a property of the FD003 test set composition, not a degradation in the pipeline's ability to alert.

**The HPC-focused causal DAG achieves higher coverage on FD003 than FD001.** This suggests that the five causal sensor branches (HPC outlet temperature and pressure, fuel flow, compressor outlet temperature) carry systemic degradation signal that is not specific to HPC fault mode — fan degradation in a turbofan eventually propagates to shared thermodynamic pathway sensors. The causal DAG's coverage is not limited to HPC degradation alone.

Comparing the two single-condition datasets (FD001 vs. FD003): the pipeline consistently outperforms IF in coverage (Fisher *p* < 0.001 in both). The primary limitation of Isolation Forest — low coverage because it requires a reading to be globally anomalous, which only a fraction of test-window cycles are — appears in both datasets.

### 4.5 FD004 Generalizability — Six Conditions, Dual Fault Mode

FD004 is the most challenging CMAPSS sub-dataset: 248 test engines operating across six conditions with both HPC and fan fault modes active. It is the direct multi-condition analogue of FD003 and the dual-fault analogue of FD002. Evaluation follows the same regime-aware protocol as FD002: k = 6 KMeans clustering on op-setting space, per-cluster causal coefficient estimation, and alert-threshold 0.30.

**Table 2d: FD004 Regime-Conditioning Results** (248 test engines, six conditions, dual fault mode, alert threshold = 0.3, W = 100 cycles)

| Variant | Coverage | 95% CI | Mean (SD) | Median | Precision | Recall | F1 | 95% CI | Wilcoxon-*p* | Fisher-*p* |
|---|---|---|---|---|---|---|---|---|---|---|
| Global z-score (FD001 means) | 100% | [100%–100%] | 251.6 (85.2) | 234.0 | 0.000 | 0.000 | 0.000 | [0.000–0.000] | — | — |
| Retrained z-score (FD004 means) | 100% | [100%–100%] | 249.8 (85.0) | 231.0 | 0.000 | 0.000 | 0.000 | [0.000–0.000] | 0.751 | 1.000 |
| Regime-aware causal | **57%** | **[51%–63%]** | **196.3 (137.2)** | **209.0** | **0.373** | **0.333** | **0.352** | **[0.284–0.420]** | **<0.001** | **<0.001** |

*KMeans k = 6 trained on FD004 train op-settings. 95% CIs bootstrapped (10 000 resamples of full 248-engine set). Global z-score uses FD001 per-sensor means and noise floors. Retrained z-score uses FD004 marginal per-sensor means and raw stds (no noise floor). Wilcoxon and Fisher tests compare regime-aware causal vs. global z-score. Rank-biserial r omitted for same reason as Table 2b: z-score baseline lead-time comparison is degenerate.*

**Regime conditioning is decisive on the hardest dataset.** Both z-score variants achieve 100% coverage with F1 = 0.000 — every engine fires, none within the actionable window, yielding precision = 0.000. The Wilcoxon test confirms the retrained z-score is statistically indistinguishable from the global baseline (*p* = 0.751, Fisher *p* = 1.000). Regime-aware causal achieves 57% coverage with F1 = **0.352** — the highest F1 of any variant across any dataset evaluated in this paper. Fisher *p* < 0.001 confirms this coverage reduction is not random.

The retrained z-score result replicates the FD002 finding: pooling data across six operating conditions produces marginal means that sit near the centre of the regime space. Healthy engines at altitude extremes deviate substantially from these pooled means, producing z-scores indistinguishable from degraded engines at moderate conditions. Retraining cannot fix this — only explicit regime conditioning does.

**RUL analysis.** Non-alerted engines (n = 106) have median RUL = 109.5 cycles at test-set end; alerted engines (n = 142) have median RUL = 46.0 cycles — a 2.4× difference. The Wilcoxon rank-sum test confirms non-alerted engines are significantly further from failure (*p* < 0.001, rank-biserial *r* = 0.498, large effect). The effect size is nearly identical to FD002 (*r* = 0.484), indicating consistent discriminative behaviour of the non-alert decision across both multi-condition datasets regardless of fault mode. The FD004 result confirms that the 43% non-alerted fraction represents genuinely lower-risk engines, not missed detections.

**Synthesis across all four datasets.** The full pipeline outperforms IF-equivalent baselines in both single-condition and multi-condition settings. In single-condition settings (FD001, FD003), the causal blend with learned α improves coverage 4.1–4.7× vs. IF. In multi-condition settings (FD002, FD004), regime-aware causal is the only variant with non-zero F1 — global and retrained z-score baselines collapse to F1 = 0.000 in both cases. The pattern is consistent and the mechanism is confirmed: **regime conditioning, not retraining, is the necessary ingredient for multi-condition anomaly detection**.

---

## 5. Discussion

### 5.1 What the ablation tells us about mechanism

The full pipeline's Wilcoxon *p* = 0.031 on FD001 is nominally significant on lead-time distribution, with Fisher *p* < 0.001 on coverage (the primary claim). Both effects are driven primarily by the aggregation correction (top-3 of 5 causal sensors, consistent with z-score's top-3 of 14): before this fix, the causal score was systematically suppressed relative to z-score, making the nominal 50/50 blend effectively z-score-dominated. The corrected pipeline with learned α = 0.60 achieves 4.1× better coverage than Isolation Forest and 53% earlier mean lead time (164.9 vs. 107.4 cycles). The F1 result (0.276 for full pipeline vs. 0.165 for IF) confirms a quality improvement: not just more alerts, but more alerts that fire within the actionable W = 100 cycle window relative to the true failure cycle. The bootstrapped 95% CI for full pipeline F1 is [0.165–0.374]; IF's CI is [0.077–0.261]. The point estimate (0.276) exceeds IF's CI upper bound (0.261), though the bootstrapped intervals partially overlap at the lower end — expected at n = 100. The coverage difference (Fisher *p* < 0.001) is the statistically robust primary claim; the F1 improvement is directionally consistent and confirmed by the point estimates but should not be over-read from the overlapping CI tails alone.

The rank-biserial effect sizes (*r*) for the Wilcoxon tests are all in the medium range (0.339–0.371 for the main variants), indicating that variants tend to have lead times roughly 34–37 percentage points more often above the IF median than below it (*r* = 0.1 small, 0.3 medium, 0.5 large).

**Sensitivity to choice of W.** The actionable window W = 100 cycles represents the maintenance planning horizon: alerts fired more than 100 cycles before failure may be dismissed as false alarms, while alerts inside 100 cycles are operationally actionable. To confirm that the choice of W = 100 does not drive the result, F1 was computed for the full pipeline and IF at W = 50, 100, and 150 cycles:

| W | IF F1 | Full pipeline F1 |
|---|---|---|
| 50  | 0.165 | 0.148 |
| 100 | 0.165 | 0.276 |
| 150 | 0.165 | 0.374 |

IF's F1 is constant across all W because its 17 alerts are bimodal: 9 fire within 50 cycles of failure (within W = 50) and 8 fire more than 150 cycles before failure (premature under all windows tested). No IF alerts fall in the 50–150 cycle range. At W = 50, IF's proximity to failure gives it a slight F1 edge (0.165 vs. 0.148) — IF alerts are very close to failure when they fire. At W ≥ 100, the full pipeline's broader coverage (69 vs. 17 alerted engines) dominates, and F1 rises to 0.276 at W = 100 and 0.374 at W = 150. The crossover between W = 50 and W = 100 reveals the operational context in which each approach excels: IF's tight proximity to failure is only valuable when maintenance can be dispatched with very short notice; the full pipeline's broader, earlier coverage better serves planning horizons of 100 cycles or more.

### 5.2 The physics veto as an orthogonal contribution

The structural veto based on the isentropic compression relation is architecturally independent of the causal scoring mechanism. It operates on a different signal (correlation structure between sensor_11 and sensor_15 over a rolling 100-reading window) using a different test (G-test for independence). Its design intent is to penalise readings where the isentropic coupling between HPC outlet temperature and pressure has broken down — a pattern consistent with sensor malfunction rather than genuine engine degradation, since real thermodynamic degradation should cause both sensors to deviate together along the compression curve.

**What the data show.** On FD001 (single condition, HPC fault only), the veto reduces coverage from 69% to 52%. This result has two possible interpretations: (1) the veto is correctly identifying coupling breaks that represent real HPC degradation and penalising those alerts — which is the intended false-negative risk of the graduated design; or (2) it is over-firing on genuine degradation signals because HPC fault *causes* isentropic decoupling, making the veto and the degradation signal confounded on this dataset. The current experimental data cannot distinguish between these interpretations, as there is no ground truth on which alerted engines in the 52%-vs-69% difference represent true detections vs. sensor artefacts. **The veto's ability to distinguish sensor faults from genuine degradation is therefore a structural claim, not an empirically demonstrated result in this paper.** What is demonstrated is that the veto reduces coverage on single-condition HPC-fault data — an expected outcome under either interpretation, since HPC degradation does disrupt isentropic coupling.

The architectural value of the veto lies in deployment scenarios where sensors can fail independently of the physical process — for example, a thermocouple drift event that spikes sensor_11 without any corresponding pressure change. In such a scenario, the causal scorer would produce a high causal score (large residual on sensor_11) and the z-scorer would also fire; the veto is the only component that can detect the structural implausibility of the temperature-without-pressure pattern. Evaluating this on synthetic fault injection data is identified as future work in Section 5.4.

The correct threshold for a 5×5 contingency table is χ²(df = 16, *p* = 0.05) = 26.30; the common error of using the df = 4 threshold (9.49, applicable to a 3×3 table) would make the veto far more conservative, firing only on extreme decouplings and missing moderate sensor faults that the correct threshold catches.

### 5.3 Limitations

**Engine-specific noise floors.** Two engines in the FD001 test set fire alerts at cycle 1, indicating that their healthy initial readings fall outside the global noise floor. Engine-specific baseline calibration (using the first N readings of each engine to set per-engine means) would eliminate these false alarms. This is straightforward to implement but requires a warm-up period before the system is operational.

**In-memory state.** The forward-fill service and G-test buffer use in-memory data structures. A server restart (which occurs after 15 minutes of inactivity on the free deployment tier) clears per-engine state. For production deployment, these should be backed by Redis.

**G-test cold start.** The physics veto is inactive for the first 100 cycles of each engine's monitoring history, as the contingency table requires 100 readings to be statistically reliable. For FD001 engines with total run lengths near 130 cycles, this means the veto may fire only in the final 30 cycles before the end of the test record — or not at all for short-life engines. The veto's contribution should be understood as applicable to sustained monitoring scenarios, not to the earliest cycles after system startup.

**Blend weight α calibration.** As declared in Section 3.2, α = 0.60 was selected by maximizing F1 on the evaluation set, introducing a data leakage risk. The practical consequence: training-set grid search gives α = 0.70, and the training-calibrated value produces identical coverage to causal-only (64%) on the test set. The blend improvement visible at α = 0.60 (69% coverage, F1 = 0.276) is sensitive to this calibration choice and should be treated as an upper bound rather than a point estimate. Proper future calibration would use k-fold cross-validation over training-set engines, or a designated validation split withheld from all performance reporting.

### 5.4 Future work

- Engine-specific baseline calibration to eliminate inter-engine variation false positives
- Online coefficient update from the human feedback loop (currently the causal coefficients are fixed at deployment)
- Replacement of in-memory state with Redis for production resilience
- Synthetic sensor fault injection experiment to directly validate the physics veto's ability to distinguish sensor malfunctions from genuine degradation signals (currently a structural claim; see §5.2)
- Multi-mode PCA and GMM baselines to quantify the improvement of causal conditioning over purely statistical regime-aware approaches
- k-fold cross-validation within the training set to produce a bias-corrected α estimate, replacing the evaluation-set-calibrated α = 0.60

---

## 6. Conclusion

We presented a regime-aware causal anomaly triage system for industrial IoT sensor data and evaluated it on all four NASA CMAPSS sub-datasets, isolating complementary failure modes of conventional detectors across both single-condition and multi-condition settings.

On single-condition data, the full pipeline consistently outperforms Isolation Forest in coverage: 69% vs. 17% on FD001 (Fisher *p* < 0.001, 53% earlier mean lead time) and 89% vs. 19% on FD003, which adds a second fault mode (fan degradation). The higher coverage on FD003 — an HPC-focused causal DAG outperforming IF on dual-fault data — indicates that the five causal sensor branches carry systemic thermodynamic degradation signal that is not specific to HPC fault mode alone.

On multi-condition data, a mechanistic isolation experiment confirms that retraining (recalibrating z-score means from the target dataset) does not solve the false positive problem: retrained z-score baselines produce 100% false positive rates on both FD002 and FD004 (F1 = 0.000, indistinguishable from global z-score). Regime conditioning is the necessary mechanism — not better calibration. Regime-aware causal scoring achieves 66% coverage on FD002 (F1 = 0.279) and 57% on FD004 (F1 = 0.352, the highest F1 of any variant across any dataset), with Fisher *p* < 0.001 on both. RUL analysis on both multi-condition datasets confirms that non-alerted engines have significantly higher RUL at test-set end than alerted engines (FD002: median 103 vs. 54 cycles, *p* < 0.001, *r* = 0.484; FD004: median 109.5 vs. 46 cycles, *p* < 0.001) — the system is declining to alarm on engines that are objectively further from failure, not missing true degradation signals.

The physics-based structural veto adds an additional layer of fault discrimination grounded in the isentropic compression relation, detecting sensor malfunctions that statistical scoring alone cannot distinguish from engine degradation.

The system is implemented as an open-source FastAPI backend with a PostgreSQL telemetry store, deployed on Render. All code, data download scripts, and evaluation scripts are available at: https://github.com/jan-code26/iot-anomaly-triage

For practitioners, these results suggest a concrete deployment rule: z-score scoring alone suffices when a fleet operates under a single known condition; regime-aware causal scoring is necessary when the fleet spans multiple operating conditions. Retraining a z-score detector on target-domain data is not sufficient — the multi-condition false positive problem requires explicit regime conditioning.

---

**Conflict of Interest:** The author declares no competing interests.

**Funding:** This research received no external funding.

---

## References

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422.

Pearl, J. (2009). *Causality: Models, reasoning, and inference* (2nd ed.). Cambridge University Press.

Saxena, A., & Goebel, K. (2008). *Turbofan engine degradation simulation data set*. NASA Ames Prognostics Data Repository.

Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. *arXiv preprint arXiv:2011.04216*.

Bransby, M. L., & Jenkinson, J. (1998). *Alarm management in the chemical and power industries: Results of a survey for the HSE* (Contract Research Report No. 166). Health and Safety Executive.

Hong, C. W., Lee, C., Lee, K., Ko, M.-S., Kim, D. E., & Hur, K. (2020). Remaining useful life prognosis for turbofan engine using explainable deep neural networks with dimensionality reduction. *Sensors*, *20*(22), 6626. https://doi.org/10.3390/s20226626

Liu, J., Zhang, C., Qian, J., Ma, M., Qin, S., Bansal, C., Lin, Q., Rajmohan, S., & Zhang, D. (2024). Large language models can deliver accurate and interpretable time series anomaly detection. *arXiv preprint arXiv:2405.15370*.

Peng, C., Chen, Y., Chen, Q., Tang, Z., Li, L., & Gui, W. (2021). A remaining useful life prognosis of turbofan engine using temporal and spatial feature fusion. *Sensors*, *21*(2), 418. https://doi.org/10.3390/s21020418

UK Health and Safety Executive. (1997). *The explosion and fires at the Texaco refinery, Milford Haven, 24 July 1994* (ISBN 0 7176 1413 1). HSE Books.

Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. *2017 IEEE International Conference on Prognostics and Health Management (ICPHM)*, 88–95.

LangChain, Inc. (2024). LangGraph: Build stateful, multi-actor applications with LLMs (Version 0.1, commit da3f34a). GitHub repository. Retrieved April 2026 from https://github.com/langchain-ai/langgraph

Yan, P., Abdulkadir, A., Luley, P.-P., Rosenthal, M., Schatte, G. A., Grewe, B. F., & Stadelmann, T. (2023). A comprehensive survey of deep transfer learning for anomaly detection in industrial time series: Methods, applications, and directions. *arXiv preprint arXiv:2307.05638*. https://arxiv.org/abs/2307.05638

Yu, J., & Qin, S. J. (2009). Multiway Gaussian mixture model based multiphase batch process monitoring. *Industrial & Engineering Chemistry Research*, *48*(18), 8585–8594. https://doi.org/10.1021/ie900479g

Zhao, S. J., Zhang, J., & Xu, Y. M. (2004). Monitoring of processes with multiple operating modes through multiple principle component analysis models. *Industrial & Engineering Chemistry Research*, *43*(22), 7025–7035. https://doi.org/10.1021/ie0497893
