# Sentinel AI — Final Report
## Sentinel AI

---

## 1. Executive Summary

Sentinel AI is an end-to-end machine learning pipeline for behavioral anomaly detection in cybersecurity access logs. The system generates realistic synthetic behavioral data for 250 entities across a 45-day window, profiles each entity's baseline behavior using an EMA-based online learning approach, detects anomalies via an ensemble of Isolation Forest and XGBoost+SMOTE, classifies 7 distinct attack types, and delivers SHAP-powered explainable alerts through a production-styled multipage SOC dashboard.

The pipeline is evaluated at a **SOC-realistic top-1% alert budget** — reflecting a real analyst's fixed daily review capacity rather than full-dataset accuracy metrics on the highly imbalanced label set.

---

## 2. Problem Statement

Design an AI/ML system that:
1. Models "normal" access behavior for users, service accounts, and edge devices
2. Detects intrusions or compromised-credential activity
3. Classifies the type of anomaly (brute force, lateral movement, impossible travel, etc.)
4. Provides explainable risk scores for SOC analyst triage

Key challenges addressed:
- **Sequential behavioral data** — access events over time, not static snapshots
- **Extreme class imbalance** — true intrusions are ~2% of total sessions
- **Concept drift** — legitimate behavior evolves and should not be permanently flagged
- **Explainability** — analysts need to know *why* an event was flagged
- **Cold-start problem** — scoring new entities with no behavioral history

---

## 3. Data Generation & Assumptions

### 3.1 Synthetic Data Design

| Parameter | Value |
|:---|:---|
| Total entities | 250 (users, service accounts, edge devices) |
| Simulation window | 45 days |
| Total sessions | 66,282 |
| Anomaly rate | 2.11% (within 0.5–3% target band) |
| Random seed | 42 (fully reproducible) |

Each entity has a unique behavioral profile sampled once:
- **Home city** with lat/lon coordinates (12 global cities)
- **Login hour distribution** (Gaussian mean/std per entity)
- **Resource subset** (4–10 resources from a 28-item catalog)
- **Auth method probability distribution** (Dirichlet-sampled)
- **Session duration** (log-normal per entity)
- **Device fingerprint** (OS, MAC, protocol)

### 3.2 Attack Taxonomy

| Attack Type | Rows | Primary Signal |
|:---|:---|:---|
| Brute Force | 630 | auth_success ≈ 0.5%, ~30 attempts/IP in minutes |
| Credential Stuffing | 296 | Many entity_ids, single IP, auth_success ≈ 5% |
| Lateral Movement | 208 | Short sessions, resources outside entity's known subset |
| Data Exfiltration | 114 | 52% off-hours, sensitive resources, export commands |
| Insider Drift | 63 | Near-normal behavior, only gradual resource expansion |
| Impossible Travel | 32 | >1000 km apart, gap too short to physically travel |
| Device Spoofing | 25 | Same entity, mismatched device fingerprint |

### 3.3 Schema Extension

The field `auth_success` (boolean) was added beyond the suggested schema. Without a pass/fail signal, brute force and credential stuffing attacks are undefined — there is no way to distinguish a burst of failed login attempts from legitimate rapid access. This extension is documented in the data dictionary (`data/raw/data_dictionary.md`).

---

## 4. Baseline Profiling Model

### 4.1 Design Approach

An **Exponential Moving Average (EMA)** profiler builds per-entity statistical profiles online, in chronological order. This design choice addresses two core requirements simultaneously:

- **Concept drift handling**: EMA decay (α = 0.15) smoothly absorbs legitimate behavioral shifts (new work hours, new device) without permanently flagging evolved behavior. A fixed-window approach would anchor to stale history.
- **Cold-start handling**: New entities blend their (limited) personal observations with a **population-level profile** segmented by `entity_type`. The confidence ramp `min(n_obs / K, 1.0)` with K = 15 ensures new entities get reasonable protection from the population baseline while personal evidence accumulates.

### 4.2 Profile Components

Each entity's profile tracks:
- Login hour mean and variance (EMA-updated)
- Session duration mean and variance (EMA-updated)
- Resource access frequency distribution (EMA-updated categorical)
- Authentication method frequency distribution
- Geolocation frequency distribution
- Most recent device fingerprint
- Observation count (confidence score)

### 4.3 Known Limitation

The EMA profiler is not attack-resistant on its own — a sustained attack pattern would slowly pull an entity's baseline toward the malicious behavior. At the injected rate (~2%), this effect is negligible. In production, profile updates should be gated on the detector's risk score (don't update on high-risk sessions).

---

## 5. Feature Engineering

### 5.1 Deviation Features

Ten deviation features are engineered in a **single chronological pass** over all sessions. For every row, the profile snapshot is read BEFORE that row updates any state — ensuring every feature represents "as of just before this session," matching real-time detector behavior. **No future leakage.**

| Feature | Description |
|:---|:---|
| `geo_velocity_kmh` | Implied travel speed vs. entity's previous session (Haversine) |
| `time_since_last_hr` | Hours since entity's previous session (-1 = cold start) |
| `hour_zscore` | Login time deviation from entity's profiled mean/std |
| `resource_novelty` | 1 if resource is outside entity's known top-resource set |
| `fingerprint_mismatch` | 1 if device fingerprint differs from last known |
| `trailing_auth_failures` | Failed logins from this source IP in the last 5 minutes |
| `transition_novelty` | 1 - P(resource | previous resource) from per-entity Markov chain |
| `profile_confidence` | Cold-start indicator (0 = new entity, 1 = established) |
| `auth_success` | Raw authentication outcome (0 = failed, 1 = success) |
| `session_duration` | Raw session length in minutes |

### 5.2 Sequence Modeling: Markov Transition Approach

Rather than deploying a heavyweight LSTM/GRU/Transformer, the system captures sequential structure via **per-entity Markov transition matrices** built from historical resource-access sequences. The transition novelty feature measures how likely the current resource access is given the entity's previous action — an out-of-pattern next-action raises the anomaly signal.

This is explicitly a lightweight stand-in for a full sequence model, chosen for the hackathon timeline. A production system could layer a recurrent or attention-based model on top of these features for stronger sequential anomaly detection.

### 5.3 Temporal Safeguards

- Gaps under 2 minutes are treated as the same physical presence (login burst), preventing jitter-induced false geo-velocity spikes
- Markov counts are updated AFTER feature extraction to prevent self-reinforcing transitions
- The profiler is updated AFTER features are computed for a given row

---

## 6. Anomaly Detection

### 6.1 Two-Model Approach

| Model | Type | Training | Purpose |
|:---|:---|:---|:---|
| **Isolation Forest** | Unsupervised | No labels used | Baseline — works without labeled attack data |
| **XGBoost + SMOTE** | Supervised | SMOTE on training split only | Comparison — leverages labels when available |

Both models are trained on the same feature set. SMOTE oversampling is applied **only to the training split** (80/20 stratified random) — never to the test set — to prevent synthetic leakage into evaluation.

### 6.2 Evaluation at SOC-Realistic Alert Budget

Traditional metrics (accuracy, F1 on the full dataset) are misleading on a 98/2 imbalanced set. Instead, evaluation is performed at a **top-1% alert budget**: the top 1% of sessions ranked by anomaly score enter the analyst review queue.

This directly reflects a real SOC's fixed daily alert capacity. Precision at this budget measures analyst noise; recall measures how many true threats are captured.

### 6.3 Model Selection

The primary model is selected by **F1 score at the top-1% budget**. XGBoost+SMOTE was selected as primary due to higher precision-recall balance at the operating budget.

### 6.4 Budget Sensitivity

The dashboard allows analysts to adjust the alert budget (0.1%–10%) in real time, dynamically recalculating precision, recall, false positive rate, and the alert queue across all pages.

---

## 7. Anomaly Classification

### 7.1 Approach

A **multiclass XGBoost classifier** is trained exclusively on the anomalous subset (sessions labeled with one of the 7 attack types). It uses the same deviation feature vectors as the detector.

### 7.2 Imbalance Handling

Per-class counts range from 630 (brute force) down to 25 (device spoofing). SMOTE is unsafe on classes this small (too few real neighbors to interpolate between), so **balanced class weighting** is used instead — reweighting the loss function rather than fabricating synthetic minority rows.

### 7.3 Results

Per-class precision/recall reports and a confusion matrix are generated and saved to `reports/figures/`. The classifier successfully distinguishes attack types with their distinct behavioral signatures.

---

## 8. Explainability Layer

### 8.1 SHAP TreeExplainer

SHAP's TreeExplainer runs against the XGBoost detector to produce per-alert feature attributions. Only features that **push toward anomaly** (positive SHAP values) are included in explanations — features that reduce risk are suppressed from the analyst-facing summary.

### 8.2 Natural-Language Templates

Raw SHAP values are not analyst-actionable. The top-3 contributing features are mapped through a **natural-language template layer**:

| Feature | Template |
|:---|:---|
| `geo_velocity_kmh` | "implausible travel speed (X km/h)" |
| `trailing_auth_failures` | "N failed logins from this source in the last 5 min" |
| `hour_zscore` | "login time Xσ from usual pattern" |
| `resource_novelty` | "accessed a resource never touched before" |
| `fingerprint_mismatch` | "device fingerprint mismatch vs known history" |
| `profile_confidence` | "limited history for this entity (cold-start)" |

**Example output**: *"flagged due to implausible travel speed (12,841 km/h) + device fingerprint mismatch + accessed a resource never touched before"*

### 8.3 MITRE ATT&CK Mapping

Each predicted attack type is mapped to a MITRE ATT&CK technique ID:

| Attack Type | MITRE ID | Technique |
|:---|:---|:---|
| Brute Force | T1110 | Brute Force |
| Impossible Travel | T1078 | Valid Accounts |
| Data Exfiltration | T1041 | Exfiltration Over C2 Channel |
| Credential Stuffing | T1110.004 | Credential Stuffing |
| Lateral Movement | T1021 | Remote Services |
| Privilege Escalation | T1068 | Exploitation for Privilege Escalation |

### 8.4 Sanity Checks

- Brute force alerts consistently mention auth failures in their explanations
- Impossible travel alerts consistently reference travel speed
- These validations confirm the SHAP attributions align with the known injected attack signatures

---

## 9. Analyst-Facing Dashboard

### 9.1 Architecture

A **Streamlit multipage application** with a custom dark-themed design system, featuring:
- Inter font family, glassmorphism effects, hover animations
- Persistent sidebar with live alert budget slider
- Session state persistence across all 7 pages

### 9.2 Pages

| Page | Function |
|:---|:---|
| **Dashboard** | SOC status overview with KPI cards, threat trend chart, severity distribution |
| **Live Alerts** | Filterable alert queue ranked by risk score, severity-aware routing |
| **Investigation** | SHAP waterfall charts, entity timeline, risk gauges, MITRE context, analyst actions |
| **Entity Search** | Entity profile lookup, session history, active alerts with investigation handoff |
| **Analytics** | ROC curve, confusion matrix, attack distribution, false positive analysis |
| **Model Health** | Pipeline component health: data, IF, classifier, SHAP, inference |
| **Settings** | Review threshold, replay mode, theme, CSV export |

### 9.3 Key Features

- **Alert Budget Slider**: Adjustable from 0.1% to 10%, dynamically recalculates the entire alert pipeline
- **Severity Routing**: One-click sidebar filters for Critical/High/Medium/Low alerts
- **Investigation Handoff**: Select any alert and open a full investigation workspace
- **CSV Export**: Download the active alert queue for external tooling

---

## 10. System Design & Scalability

### 10.1 Current Architecture

The current system runs as a batch pipeline: data generation → profiling → feature engineering → detection → classification → explainability → dashboard. All components are in-process Python with cached model artifacts.

### 10.2 Production Streaming Path

For real-time deployment, the pipeline would need:
- **Ingestion**: Apache Kafka or AWS Kinesis for log stream ingestion
- **Processing**: Apache Flink or Spark Structured Streaming for feature engineering
- **Serving**: Model serving via FastAPI or TensorFlow Serving
- **Storage**: Time-series database (InfluxDB/TimescaleDB) for entity profiles
- **Dashboard**: WebSocket-backed real-time updates

The EMA profiler design already supports O(1) memory per entity update, making it inherently compatible with streaming architectures. The feature engineering pass is already chronological and single-pass — it does not require random access to future data.

---

## 11. Known Limitations

1. **Synthetic data**: Does not capture real-world log noise (packet loss, malformed fields, clock skew, adversarial evasion techniques)
2. **Sequence modeling**: Markov-transition features are a lightweight stand-in for LSTM/GRU/Transformer approaches that could capture longer-range sequential dependencies
3. **Command sequence vocabulary**: Only 8 actions — real privileged-session logs would have far richer command vocabularies
4. **Attack-resistant profiling**: The EMA profiler would be gradually pulled by sustained attacks; production systems should gate updates on risk score
5. **Incident calibration**: Attack injection counts use heuristic ratios rather than calibrated real-world base rates
6. **Insider drift**: Intentionally the weakest signal (by design) — near-normal behavior, intended for false-positive tuning rather than reliable detection

---

## 12. Conclusion

Sentinel AI demonstrates that a lightweight, classical ML pipeline — EMA profiling, tree-based detection, and template-driven SHAP explainability — can deliver production-quality behavioral anomaly detection without the overhead of deep sequence models. By evaluating at a SOC-realistic alert budget and translating model outputs into analyst-readable natural-language evidence, the system bridges the gap between ML research and operational security.

---

## Appendix A: Tech Stack

| Component | Library |
|:---|:---|
| Data Generation | Pandas, NumPy, Faker |
| ML / Detection | XGBoost, Scikit-learn, Isolation Forest |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Profiling | Custom EMA-based online profiler |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit (multipage app) |
| MITRE Mapping | ATT&CK technique IDs |

## Appendix B: Reproducibility

All random seeds are fixed (seed=42). The full pipeline can be reproduced from scratch:

```bash
python src/data_generator.py     # Phase 1
python src/eda.py                # Phase 2
python src/profiling.py          # Phase 3
python src/features.py           # Phase 4.1
python src/detection_model.py    # Phase 4.2
python src/classifier.py         # Phase 5
python src/explain.py            # Phase 6
streamlit run app/dashboard.py   # Phase 7
```
