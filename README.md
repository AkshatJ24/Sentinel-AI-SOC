<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.22+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1.7+-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🛡️ Sentinel AI — Behavioral Anomaly Detection for Cybersecurity

> An end-to-end machine learning pipeline that generates realistic behavioral access logs, profiles entity baselines with EMA-based online learning, detects anomalies using Isolation Forest and XGBoost+SMOTE, classifies attack types, and delivers SHAP-powered explainable alerts through a production-styled Security Operations Center (SOC) dashboard built with Streamlit.

<p align="center">
  <a href="https://akshat-sentinel-ai-soc.streamlit.app/">🌐 Live Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-pipeline-stages">Pipeline</a> •
  <a href="#-getting-started">Setup</a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Pipeline Stages](#-pipeline-stages)
- [Dashboard Pages](#-dashboard-pages)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Resume Bullet Points](#-resume-bullet-points)

---

## 📖 Overview

Security Operations Centers (SOCs) face a critical challenge: identifying genuine threats buried within massive volumes of access logs while minimizing analyst fatigue from false positives. **Sentinel AI** addresses this by building a complete behavioral anomaly detection pipeline — from synthetic data generation to explainable alert delivery.

The system profiles 250 entities (users, service accounts, edge devices) across a 45-day simulation window, engineers 10 deviation features via a single chronological pass (zero future leakage), and evaluates detection at a **SOC-realistic top-1% alert budget** — reflecting a real analyst's fixed daily review capacity rather than full-dataset accuracy metrics.

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Sentinel AI Pipeline                         │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────┤
│  Phase 1     │  Phase 2-3   │  Phase 4     │  Phase 5     │ Phase 6  │
│  Synthetic   │  EDA &       │  Anomaly     │  Attack      │ SHAP     │
│  Data Gen    │  Profiling   │  Detection   │  Classifier  │ Explain  │
│              │              │              │              │          │
│  250 entities│  EMA-based   │  Isolation   │  XGBoost     │ Per-alert│
│  7 attack    │  online      │  Forest +    │  multi-class │ NL       │
│  types       │  profiling   │  XGBoost     │  (balanced   │ templates│
│  ~98/2 split │  cold-start  │  + SMOTE     │  weighting)  │ top-3    │
│              │  blending    │              │              │ features │
├──────────────┴──────────────┴──────────────┴──────────────┴──────────┤
│                     Phase 7 — SOC Dashboard (Streamlit)              │
│  Dashboard │ Live Alerts │ Investigation │ Entity Search │ Analytics │
│                       Model Health │ Settings                        │          
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technologies |
|:---|:---|
| **Language** | Python 3.10+ |
| **ML / Detection** | XGBoost, Scikit-learn, Isolation Forest |
| **Imbalance Handling** | SMOTE (imbalanced-learn) |
| **Explainability** | SHAP (TreeExplainer) |
| **Profiling** | Custom EMA-based online entity profiler |
| **Data Generation** | Faker, NumPy, Pandas |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit (multipage app) |
| **MITRE Mapping** | ATT&CK technique IDs for attack types |

---

## 🔬 Pipeline Stages

### Phase 1 — Synthetic Data Generation (`src/data_generator.py`)
- Generates behavioral access logs for **250 entities** (users, service accounts, edge devices) across a **45-day** simulation window
- Injects **7 attack types** at a realistic ~2% anomaly rate:
  - **Brute Force** — rapid failed authentication bursts from a single IP
  - **Impossible Travel** — geographically implausible logins within minutes
  - **Credential Stuffing** — one IP spraying credentials across many accounts
  - **Device Spoofing** — mismatched device fingerprints on edge devices
  - **Lateral Movement** — accessing many novel resources in a short window
  - **Data Exfiltration** — off-hours access to sensitive resources over days
  - **Insider Drift** — gradual resource footprint expansion (ambiguous, for FP tuning)
- Each entity has a unique behavioral profile: home city, login hour distribution, resource preferences, auth method probabilities, and device fingerprint

### Phase 2 — Exploratory Data Analysis (`src/eda.py`)
- Class imbalance analysis and per-entity profile sanity checks
- Attack-type separability visualizations (auth success rate, session duration, off-hours rate, burst signatures)
- Schema completeness validation against the problem-statement requirements

### Phase 3 — Baseline Profiling (`src/profiling.py`)
- **Online EMA (Exponential Moving Average)** profiler that updates per-entity statistics in chronological order
- **Cold-start blending**: new entities fall back to population-level profiles with a confidence ramp (`min(n_obs / K, 1.0)`)
- **Concept drift adaptation**: EMA decay smoothly absorbs legitimate behavioral shifts without permanent flagging
- Produces per-entity profiles: hour distribution, resource frequencies, auth patterns, device fingerprints

### Phase 4 — Anomaly Detection (`src/features.py`, `src/detection_model.py`)
- **10 deviation features** engineered in a single chronological pass (no future leakage):
  - `geo_velocity_kmh` — implied travel speed vs. previous session
  - `time_since_last_hr` — gap since the entity's last session
  - `hour_zscore` — login time deviation from the entity's profiled pattern
  - `resource_novelty` — access to a never-before-touched resource
  - `fingerprint_mismatch` — device fingerprint differs from known history
  - `trailing_auth_failures` — failed logins from this source IP in the last 5 minutes
  - `transition_novelty` — Markov-based action sequence unlikelihood
  - `profile_confidence` — cold-start indicator (low = limited entity history)
  - `auth_success`, `session_duration` — raw behavioral signals
- **Two detection models** evaluated at a top-1% alert budget:
  - **Isolation Forest** — unsupervised baseline (no labels needed)
  - **XGBoost + SMOTE** — supervised, with synthetic oversampling applied only to the training split
- Primary model selected by F1 score at the SOC-realistic alert budget

### Phase 5 — Attack Classification (`src/classifier.py`)
- Multiclass XGBoost classifier trained on the anomalous subset only
- Uses balanced class weighting (not SMOTE) due to extreme per-class imbalance in the minority set
- Predicts the attack category (brute force, lateral movement, etc.) for each flagged session
- Generates per-class precision/recall reports and a confusion matrix

### Phase 6 — Explainability (`src/explain.py`)
- **SHAP TreeExplainer** on the XGBoost detector for per-alert feature attributions
- Top-3 contributing features mapped through **natural-language templates** — translating raw SHAP values into analyst-readable sentences
- Sanity checks: brute force alerts mention auth failures, impossible travel alerts mention travel speed

### Phase 7 — SOC Dashboard (`app/`)
- Production-styled **Streamlit multipage application** with a custom dark-themed design system
- 6 dashboard pages with sidebar navigation, alert budget controls, and session state persistence

---

## 📊 Dashboard Pages

| Page | Description |
|:---|:---|
| **🏠&nbsp;Dashboard** | SOC status overview — KPI cards, threat trend chart, severity distribution, recent alerts |
| **🚨&nbsp;Live&nbsp;Alerts** | Filterable alert queue ranked by risk score with severity-aware routing and investigation handoff |
| **🔍&nbsp;Investigation** | Deep-dive into a selected alert — SHAP waterfall, entity timeline, risk gauges, MITRE ATT&CK mapping, and analyst action recommendations |
| **👤&nbsp;Entity&nbsp;Search** | Entity profile lookup — session history timeline, resource access patterns, active alerts with investigation handoff |
| **📊&nbsp;Analytics** | Model performance metrics — ROC curve, confusion matrix, attack distribution, false positive analysis at active budget |
| **🧠&nbsp;Model&nbsp;Health** | Pipeline component health status — data pipeline, Isolation Forest, classifier, SHAP, and inference readiness |
| **⚙&nbsp;Settings** | Analyst workspace configuration — review threshold, replay mode, theme selection, CSV export of the active alert queue |

---

## 📁 Project Structure

```
Sentinel-AI-SOC/
├── app/
│   ├── dashboard.py                # Main dashboard entry point
│   ├── soc_shared.py               # Shared UI components, data loading, navigation
│   └── pages/
│       ├── 1_Live_Alerts.py
│       ├── 2_Investigation.py
│       ├── 3_Entity_Search.py
│       ├── 4_Analytics.py
│       ├── 5_Model_Health.py
│       └── 6_Settings.py
├── src/
│   ├── data_generator.py           # Phase 1: Synthetic behavioral data generation
│   ├── eda.py                      # Phase 2: Exploratory data analysis
│   ├── profiling.py                # Phase 3: Online EMA entity profiling
│   ├── features.py                 # Phase 4.1: Deviation feature engineering
│   ├── detection_model.py          # Phase 4.2: Isolation Forest + XGBoost detection
│   ├── classifier.py               # Phase 5: Multiclass attack classification
│   └── explain.py                  # Phase 6: SHAP explainability layer
├── data/
│   ├── raw/
│   │   ├── access_logs.csv         # Generated behavioral access logs (~13 MB)
│   │   └── data_dictionary.md      # Schema documentation
│   ├── labels/
│   │   └── labels.csv              # Ground truth session labels
│   └── processed/
│       ├── deviation_features.csv  # Engineered feature vectors
│       ├── entity_profiles.csv     # Final entity profile table
│       ├── detection_models.pkl    # Trained detection models (IF + XGB)
│       ├── classifier.pkl          # Trained attack classifier
│       └── explained_alerts.csv    # SHAP-explained top-1% alerts
├── reports/
│   └── figures/                    # EDA and classification visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AkshatJ24/Sentinel-AI-SOC.git
   cd Sentinel-AI-SOC
   
2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Reproducing the Pipeline (Optional)

The processed data and model artifacts are included in the repository. To regenerate from scratch:

```bash
# Phase 1 — Generate synthetic data
python src/data_generator.py

# Phase 2 — Run EDA
python src/eda.py

# Phase 3 — Build entity profiles
python src/profiling.py

# Phase 4 — Engineer features and train detection models
python src/features.py
python src/detection_model.py

# Phase 5 — Train attack classifier
python src/classifier.py

# Phase 6 — Generate SHAP explanations
python src/explain.py
```

---

## 💻 Usage

### Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

The application will open at `http://localhost:8501`. Use the sidebar to navigate between pages. The **Alert Budget** slider (in the sidebar) controls the percentage of sessions that enter the active analyst queue — adjusting it dynamically recalculates precision, recall, and the alert queue across all pages.

---

## 🌐 Deployment

🔗 **Live Demo**: [https://akshat-sentinel-ai-soc.streamlit.app/](https://akshat-sentinel-ai-soc.streamlit.app/)

---


