# 🛡️ AI-Powered Behavioral Anomaly Detection for Cybersecurity
### Hackathon Problem Statement | Sequence-Aware Intrusion Detection & Explainable Risk Scoring

> **How to use this file:**
> - `[ ]` = Not started &nbsp;|&nbsp; `[~]` = In Progress &nbsp;|&nbsp; `[x]` = Done
> - Update **Status** and **Notes** as you go — this is your single source of truth for what's left.
> - The **💡 Stretch** tag marks upgrades to attempt only if you're ahead of schedule.

---

## ⚠️ Time Reality Check

**Deadline: 26/07/2026, 11:59 PM** — roughly a day and a half of working time from now (25/07/2026), less if you sleep.

The problem statement lists **7 deliverables + a report + a presentation**. Built naively (deep sequence models, graph neural nets, full streaming infra) this is a **1–2 week** scope. To actually finish and submit:

- Every phase below has a **classical/statistical MVP path** (fast, uses your existing sklearn/XGBoost stack) marked as the default.
- Deep learning (LSTM/GRU/Transformer) and graph-based approaches are marked **💡 Stretch** — only attempt if the MVP is done with hours to spare.
- **Target submission time: 9:00–10:00 PM on 26/07**, not 11:59 PM. Leave a buffer for upload errors (the portal only accepts PDF/ZIP — see Phase 8).

---

## 🗂️ Project Overview

| Field | Details |
|---|---|
| **Project Title** | AI-Powered Behavioral Anomaly Detection for Cybersecurity |
| **Type** | Hackathon submission (synthetic data, no real dataset provided) |
| **Core Pillars** | Sequential Behavior Modeling · Imbalanced Anomaly Detection · Explainable Risk Scoring · Cold-Start & Drift Handling |
| **Deliverables** | Data generator · Baseline profiler · Detection model · Anomaly classifier · Explainability layer · Analyst dashboard · Report · Presentation |
| **Deadline** | **26/07/2026, 11:59 PM** |
| **Submission format** | Other deliverables: PDF or ZIP (convert to PDF if ZIP fails) · Presentation: must use the provided SIH Idea Submission template, max 6 content slides, exported as PDF only — no PPT/Word accepted |
| **Status** | 🟢 Phase 7 Complete : Streamlit analyst SOC dashboard built |

---

## 📁 Folder Structure

```
behavioral-anomaly-detection/
├── data/
│   ├── raw/                    ← generated synthetic access logs
│   ├── labels/                 ← ground-truth labels (held out from models)
│   └── processed/              ← feature-engineered dataset
├── notebooks/
│   ├── 01_Data_Generation.ipynb
│   ├── 02_EDA_Synthetic_Data.ipynb
│   ├── 03_Baseline_Profiling.ipynb
│   ├── 04_Detection_Model.ipynb
│   ├── 05_Anomaly_Classification.ipynb
│   ├── 06_Explainability.ipynb
│   └── 07_Evaluation_Report.ipynb
├── src/
│   ├── data_generator.py       ← synthetic log + attack injection
│   ├── profiling.py            ← per-entity baseline profile builder
│   ├── features.py             ← deviation feature engineering
│   ├── detection_model.py      ← anomaly scoring model
│   ├── classifier.py           ← anomaly-type classifier
│   └── explain.py              ← SHAP wrapper + NL templating
├── app/
│   └── dashboard.py            ← Streamlit analyst dashboard
├── reports/
│   ├── figures/
│   └── final_report.pdf
├── presentation/
│   └── slides.pptx             ← from hackathon-provided template
└── README.md
```

---

## 🧰 Tech Stack

```
pandas, numpy, faker,                # synthetic data
scikit-learn, xgboost, imbalanced-learn (SMOTE),
shap,                                 # explainability
matplotlib, seaborn, plotly,
streamlit                             # dashboard

💡 Stretch only: torch or tensorflow (LSTM/GRU), networkx (graph features)
```

---

## 🗺️ Phase-wise Roadmap

---

### Phase 0 — Setup & Spec Lock-in
*Today (25/07), ~1 hr*

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Finalize PROJECT_OBJECTIVE.md (problem summary, requirements, assumptions, open questions) | `[x]` | Already authored |
| 0.2 | Set up repo/folder structure above | `[x]` | |
| 0.3 | Install tech stack | `[x]` | |
| 0.4 | Source the presentation template referenced in the problem statement | `[x]` | Received: SIH Idea Submission template (`IDEA_Presentation_Format.pptx`) — 6 content slides max, PDF export only, template's given bullet headers can't be altered |
| 0.5 | Lock MVP scope decision: statistical/tree-based path primary, deep learning explicitly stretch-only | `[x]` | Prevents scope creep given the deadline |
| 0.6 | Gather title-slide details: Problem Statement ID, exact PS Title, Theme, PS Category (Software/Hardware), your registered name & student ID | `[x]` | Not present in the problem-statement file provided here — pull from the SIH portal listing where this problem was assigned |

---

### Phase 1 — Synthetic Data Generator
*Today, ~3–4 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | Build entity roster: `entity_id`, `entity_type` (user/service_account/edge_device) via Faker | `[x]` | |
| 1.2 | Build per-entity behavioral profile: login-hour mean/std, home geo, typical resource subset, auth method, session-duration distribution | `[x]` | This *is* the "normal" baseline used again in Phase 3 |
| 1.3 | Generate normal sessions per entity by sampling profile + noise | `[x]` | |
| 1.4 | Write injection function: brute force | `[x]` | Rapid failed-auth from one source, short window |
| 1.5 | Write injection function: impossible travel | `[x]` | Distant geo pair, implausible time gap |
| 1.6 | Write injection function: credential stuffing | `[x]` | Many entity_ids, few source_ips, high failure rate |
| 1.7 | Write injection function: lateral movement | `[x]` | Unusual breadth/sequence of resources for that entity |
| 1.8 | Write injection function: device spoofing | `[x]` | Same device_id, mismatched fingerprint |
| 1.9 | Write injection function: low-and-slow exfiltration | `[x]` | Gradual off-hours access building over days/weeks |
| 1.10 | Write injection function: insider drift (edge case) | `[x]` | Ambiguous — legit entity slowly expanding footprint, used for FP tuning, not a hard anomaly |
| 1.11 | Inject at controlled rate (0.5–3% of sessions), keep `label` in a separate held-out file | `[x]` | Label must be strippable at inference time |
| 1.12 | Write `data_dictionary.md`: document every assumption made | `[x]` | Directly feeds Phase 8 report |

---

### Phase 2 — EDA on Synthetic Data
*Today evening, ~1–1.5 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Check class imbalance % (should be small, matching real intrusion rarity) | `[x]` | |
| 2.2 | Plot a few entities' normal-session profiles to sanity-check realism | `[x]` | |
| 2.3 | Confirm each injected attack type is visually distinguishable from baseline | `[x]` | If not, injection logic needs adjusting before modeling |
| 2.4 | Check schema completeness against the suggested schema table | `[x]` | |

---

### Phase 3 — Baseline Profiling Model
*Tonight (25/07), ~2 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | Compute rolling per-entity statistical profile (mean/std login hour, geo set, resource set, auth distribution) | `[x]` | This is the "what does normal look like" reference model |
| 3.2 | Cold-start fallback: population profile segmented by `entity_type`, blended in as individual evidence accumulates | `[x]` | Directly answers the cold-start requirement |
| 3.3 | Use exponential moving average (not a fixed historical window) when updating profiles | `[x]` | This is how concept drift gets handled without permanently flagging evolved-but-legitimate behavior |
| 3.4 | Save profile table (one row per entity, versioned by update time) | `[x]` | |
| 3.5 | 💡 Stretch: autoencoder or One-Class SVM trained on normal session vectors, reconstruction error as anomaly signal | `[ ]` | Only if MVP profile table is done early |

---

### Phase 4 — Detection Model (sequence-aware)
*Day 2 morning (26/07), ~3–4 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 4.1 | Engineer deviation features: geo-velocity, time-since-last-session, hour-of-day z-score vs. profile, resource novelty flag, fingerprint mismatch flag, trailing auth-failure count | `[x]` | |
| 4.2 | Build per-entity Markov transition matrix over resource sequences → "next-action novelty" feature | `[x]` | This is the lightweight stand-in for a full sequence model |
| 4.3 | Train Isolation Forest (unsupervised) on deviation features | `[x]` | Doesn't need labels — good for the "true intrusions are rare" reality |
| 4.4 | Train XGBoost (supervised, SMOTE on training set only) as a comparison | `[x]` | |
| 4.5 | Evaluate both at a realistic alert budget (top 1% of events by score) | `[x]` | Matches the hackathon's stated evaluation criterion directly |
| 4.6 | Pick primary model based on top-1% precision/recall trade-off | `[x]` | |
| 4.7 | 💡 Stretch: LSTM/GRU next-resource predictor, high prediction error = anomaly score | `[ ]` | Only attempt if Phase 4 MVP is done with time to spare |
| 4.8 | 💡 Stretch: graph-based entity-resource ego-network shift detection for lateral movement | `[ ]` | Same condition as above |

---

### Phase 5 — Anomaly Classification
*Day 2 late morning, ~2 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 5.1 | Train multi-class classifier (XGBoost/Random Forest) on flagged sessions using ground-truth attack_type | `[x]` | Categories: brute_force / impossible_travel / credential_stuffing / lateral_movement / device_spoofing / exfiltration / insider_drift |
| 5.2 | Handle imbalance across attack types (SMOTE or `class_weight='balanced'`) | `[x]` | |
| 5.3 | Plot confusion matrix across all attack types | `[x]` | |
| 5.4 | Report per-class precision/recall | `[x]` | |

---

### Phase 6 — Explainability Layer
*Day 2 early afternoon, ~2 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 6.1 | SHAP TreeExplainer on the Phase 4/5 model | `[x]` | |
| 6.2 | Extract top-3 contributing features per alert | `[x]` | |
| 6.3 | Build a natural-language template layer (e.g. "flagged due to geo-velocity + new device fingerprint") | `[x]` | This is what a SOC analyst actually reads — not raw SHAP values |
| 6.4 | Sanity-check explanations against known injected attacks | `[x]` | If the explanation for a known brute-force injection doesn't mention auth failures, the feature set needs work |

---

### Phase 7 — Analyst-Facing Dashboard
*Day 2 afternoon, ~2–3 hrs*

| # | Task | Status | Notes |
|---|---|---|---|
| 7.1 | Streamlit app: ranked alert queue sorted by risk score | `[x]` | |
| 7.2 | Row detail view: entity history timeline vs. the flagged event | `[x]` | |
| 7.3 | SHAP contribution mini-chart per selected alert | `[x]` | |
| 7.4 | Alert-budget slider (top X% of events) so false-positive rate is visible live | `[x]` | Directly demoes the evaluation criterion |
| 7.5 | Screenshot/record for the report and slides (deployment optional given the timeline) | `[x]` | Don't burn hours deploying if local screenshots suffice |

---

### 📽️ Presentation Template Reference (SIH Idea Submission Format)

Constraints from the template: **max 6 content slides total**, points/diagrams over paragraphs, don't alter the template's given bullet headers, delete the instructions slide before upload, **export as PDF — no PPT/Word accepted**.

| Slide | Section | What goes there |
|---|---|---|
| 1 | *(Important Instructions)* | Delete before upload — not a content slide |
| 2 | Title Page | Problem Statement ID, PS Title, Theme, PS Category, your name & student ID |
| 3 | Idea Title | Proposed solution, how it addresses the problem, novelty/uniqueness |
| 4 | Technical Approach | Tech stack + methodology (flowchart of the pipeline) |
| 5 | Feasibility & Viability | Feasibility analysis, risks, mitigation strategies |
| 6 | Artifacts | Embedded code snippet, solution image, dashboard screenshots |
| 7 | Research & References | Sources/links used while designing the approach |

---

### Phase 8 — Report, Presentation & Submission Packaging
*Day 2 evening, before 11:59 PM — target 9–10 PM*

| # | Task | Status | Notes |
|---|---|---|---|
| 8.1 | Write report: assumptions, injection rates, feature rationale, metrics at top-1% budget, drift/cold-start handling, known limitations | `[ ]` | |
| 8.2 | Fill Slide 2 — Title Page | `[ ]` | Needs Phase 0.6 details first |
| 8.3 | Fill Slide 3 — Idea Title / Proposed Solution | `[ ]` | Pull from the Implementation section below; lead with novelty (e.g. lightweight Markov-transition + EMA-adaptive baseline + NL-templated SHAP explanations as a fast alternative to a full sequence model) |
| 8.4 | Fill Slide 4 — Technical Approach | `[ ]` | Tech stack list + a simple pipeline flowchart: data gen → profiling → detection → classification → explainability → dashboard |
| 8.5 | Fill Slide 5 — Feasibility & Viability | `[ ]` | Pull straight from the report's limitations (8.1); state mitigations — analyst-in-the-loop review, EMA drift updates, cold-start blending |
| 8.6 | Fill Slide 6 — Artifacts | `[ ]` | One key code snippet, an architecture/solution image, dashboard screenshots from Phase 7.5 |
| 8.7 | Fill Slide 7 — Research & References | `[ ]` | Any papers/sources referenced while designing the approach |
| 8.8 | Delete Slide 1 (Important Instructions) | `[ ]` | Template explicitly requires this before upload |
| 8.9 | Export the filled deck to PDF | `[ ]` | PDF only — PPT/Word rejected |
| 8.10 | Export notebooks/report to PDF | `[ ]` | |
| 8.11 | Zip remaining deliverables; if ZIP upload fails, print everything to PDF instead | `[ ]` | Explicit fallback stated in the problem statement |
| 8.12 | Submit with buffer before 11:59 PM | `[ ]` | |

---

## 🔧 Implementation

*What each component actually does, and how it does it.*

### 1. Synthetic Data Generator
**What it does:** Produces a realistic access-log dataset (per the suggested schema) with a small, controlled percentage of sessions replaced by injected attack patterns, and keeps the ground-truth label separate so it can be hidden from the models at inference time.

**How:** For every entity, a "behavioral profile" is sampled once — a login-hour mean/std, a home geo-location, a subset of the resource catalog it normally touches, an auth-method distribution, and a session-duration distribution. Normal sessions are then generated by drawing from that entity's own profile with added Gaussian/log-normal noise, so no two normal sessions are identical but they stay statistically consistent per entity. Each attack pattern gets its own injection function that takes a normal session (or pair of sessions) and mutates the specific fields the pattern is defined by — e.g. impossible travel duplicates a session at a geographically distant location with a timestamp gap too short to have physically traveled; device spoofing reuses a `device_id` but swaps the `device_fingerprint`. These functions are run at a controlled rate (0.5–3% of total sessions) over the normal data, and the resulting `label` (normal / attack_type) is written to a separate file so unsupervised parts of the pipeline never see it directly.

### 2. Baseline Profiling Model
**What it does:** Defines "normal" for each entity — the reference every later component compares against — and provides a fallback for entities with no history.
**How:** A rolling statistical profile per entity (mean/std of login hour, the set of geo-locations seen, the set of resources touched, auth-method frequencies, session-duration stats), updated with an exponential moving average rather than a fixed window — new legitimate behavior gradually shifts the baseline instead of being permanently flagged, which is how concept drift is handled. For an entity with no history (cold-start), the model falls back to a population-level profile segmented by `entity_type`, and blends in the entity's own observations with increasing weight as more sessions accumulate.

### 3. Detection Model
**What it does:** Scores each session against the entity's baseline and recent sequence, producing a continuous risk score.
**How:** Each session is converted into a deviation feature vector: geo-velocity (implied travel speed versus the previous session), time since last session, how many standard deviations the login hour is from the entity's profile, whether the resource accessed is novel for that entity, whether the device fingerprint matches history, and a trailing count of auth failures. Sequence structure is captured with a per-entity Markov transition matrix built from historical resource-access sequences — how often has resource B followed resource A for this entity — so an out-of-pattern next-action lowers the transition likelihood and raises the score. These features feed an Isolation Forest (unsupervised — doesn't require labeled attacks, which matters since real intrusions are rare) and, as a comparison, an XGBoost classifier trained with SMOTE oversampling on the synthetic labels. Both are evaluated at a realistic analyst alert budget (top 1% of events by score) rather than at an arbitrary threshold, and the better trade-off is kept as primary.

### 4. Anomaly Classification
**What it does:** For sessions the detection model flags, predicts which attack category it most resembles.
**How:** A multi-class XGBoost/Random Forest classifier trained on the same deviation feature vectors, using the ground-truth `attack_type` labels (available only in the synthetic training data), predicting one of the seven injected categories. Class imbalance across attack types is handled with SMOTE or `class_weight='balanced'`, since normal sessions vastly outnumber any single attack type.

### 5. Explainability Layer
**What it does:** Turns a risk score into a reason an analyst can act on.
**How:** SHAP's TreeExplainer runs against the classification/detection model to get per-alert feature attributions, and the top 2–3 contributing features are mapped through a natural-language template (e.g. "flagged due to geo-velocity 18× normal + new device fingerprint") rather than showing raw SHAP values, since that's what an analyst actually needs to triage quickly.

### 6. Analyst-Facing Dashboard
**What it does:** Gives an analyst a ranked, explorable queue of alerts.
**How:** A Streamlit app displays alerts sorted by risk score, with columns for entity, predicted attack type, score, and top contributing features. Selecting a row expands into that entity's session history plotted against the flagged event, plus its SHAP attribution chart. A slider controls the alert budget (top X%) so the resulting false-positive rate is visible live — directly demonstrating the hackathon's stated evaluation criterion.

### 7. Report
**What it does:** Documents the reasoning behind every design choice, the metrics achieved, and where the system falls short.
**How:** Written up from the data dictionary (Phase 1.12), the feature rationale (Phase 4), the drift/cold-start handling (Phase 3), and the top-1%-budget metrics (Phase 4.5) — covering assumptions made in the synthetic data, why each model was chosen over alternatives, and known limitations (e.g. synthetic data won't capture real-world log noise, the Markov/feature-based sequence model is a lighter stand-in for a full LSTM/Transformer, and true real-time streaming would need additional infrastructure work beyond this scope).

---

## 📋 Evaluation Criteria → Where It's Addressed

| Hackathon Criterion | Addressed In |
|---|---|
| Detection accuracy on imbalanced labels | Phase 4 (Isolation Forest / SMOTE-XGBoost) |
| Correct anomaly-type classification | Phase 5 |
| False positive rate at realistic alert budget (top 1%) | Phase 4.5, demoed live in Phase 7.4 |
| Explainability / analyst usability | Phase 6, Phase 7 |
| Cold-start & concept drift handling | Phase 3.2, Phase 3.3 |
| System design & scalability (streaming feasibility) | Report — Phase 8.1, discuss as design commentary since full streaming infra is out of scope for the timeline |
| Report clarity | Phase 8.1 |

---

## 📈 Key Metrics Dashboard *(fill in once models are trained)*

| Metric | Target | Achieved | Notes |
|---|---|---|---|
| Precision @ top 1% alert budget | — | | |
| Recall @ top 1% alert budget | — | | |
| Anomaly-type classification F1 (macro) | — | | |
| Cold-start entity handling | Qualitative | | Describe fallback behavior observed |
| Concept drift handling | Qualitative | | Describe EMA adaptation observed |

---

## 🏁 Final Checklist Before Submission

- [ ] All notebooks run top-to-bottom without errors
- [ ] `src/` contains reusable functions, not notebook-only logic
- [ ] Ground-truth labels were never leaked into unsupervised training
- [ ] Report covers assumptions, metrics, and limitations
- [ ] Presentation follows the SIH template exactly (bullet headers unaltered, ≤6 content slides, instructions slide deleted, exported as PDF)
- [ ] Everything converted to PDF or zipped per submission rules
- [ ] Submitted with buffer time before 26/07/2026, 11:59 PM

---

*Tracker created: 25 July 2026 | Deadline: 26 July 2026, 11:59 PM*
