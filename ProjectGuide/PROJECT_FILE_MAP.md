# 🛡️ Sentinel AI — Project File Map

> Concise reference mapping each important file to its responsibility and relationships.
> Paths are repo-relative. For deeper explanations see `PROJECT_MASTER_GUIDE.md` §3 and the Appendix.

## Execution order (mental model)
`data_generator → eda → profiling → features → detection_model → classifier → explain → (dashboard)`
Everything in `src/` is **offline/batch**; everything in `app/` is **online/serving** and only *reads* artifacts.

## `src/` — the ML pipeline

| File | Responsibility | Key functions / classes | Reads | Writes | Called by |
|:---|:---|:---|:---|:---|:---|
| `data_generator.py` | Phase 1 — synthetic logs | `build_entities`, `gen_normal_sessions`, `inject_brute_force`/`_impossible_travel`/`_credential_stuffing`/`_device_spoofing`/`_lateral_movement`/`_low_and_slow_exfil`/`_insider_drift`, `generate_dataset` | — | `data/raw/access_logs.csv`, `data/labels/labels.csv` ⚠️ hardcoded Linux paths in `__main__` | run manually |
| `eda.py` | Phase 2 — validation + figures | flat script (runs on import) | raw logs + labels | `reports/figures/entity_profile_sanity.png`, `attack_separability.png` | run manually |
| `profiling.py` | Phase 3 — online EMA profiler | `EntityProfiler` (`get_effective_profile`, `update`, `confidence`, `final_table`), `build_population_profiles`, `_ema_dict_update`, `demo_cold_start`, `demo_drift` | raw logs + labels | `data/processed/entity_profiles.csv`, `entity_profiler.pkl` (not committed) | `features.py` imports `EntityProfiler`, `build_population_profiles` |
| `features.py` | Phase 4.1 — 10 deviation features (leakage-free) | `engineer_features`, `haversine_km` | `access_logs.csv`, `labels.csv` | `data/processed/deviation_features.csv` | run manually; output used by detection + classifier |
| `detection_model.py` | Phase 4.2–4.6 — detectors + budget eval | `load_data`, `eval_at_budget`, `main` | `deviation_features.csv` | `data/processed/detection_models.pkl` | run manually; artifact used by `explain.py` + dashboard |
| `classifier.py` | Phase 5 — multiclass attack classifier | `main` (LabelEncoder, balanced sample weights, multiclass XGBoost) | `deviation_features.csv` | `data/processed/classifier.pkl`, `reports/figures/classification_confusion_matrix.png`, `classification_per_class_report.csv` | run manually; artifact used by dashboard |
| `explain.py` | Phase 6 — SHAP → NL explanations | `build_explanation`, `describe_feature`, `FEATURE_LABELS` | `detection_models.pkl`, `deviation_features.csv` | `data/processed/explained_alerts.csv` | **imported by `app/soc_shared.py`**; also run manually |

## `app/` — the Streamlit SOC dashboard

| File | Responsibility | Key functions | Relationship |
|:---|:---|:---|:---|
| `soc_shared.py` | The engine: theme, data load, workspace build, all render/chart helpers, navigation | `configure_page`/`inject_css`, `load_artifacts` (cached), `build_workspace`/`get_workspace` (cached, budget-keyed), `current_alert_budget`/`_save_budget`, `render_sidebar`, `enrich_alert_presentation`, `build_shap_chart`, `build_entity_timeline`, `build_roc_figure`, `build_confusion_figure`, `build_gauge`, `render_investigation_panel`, `render_entity_search`, `render_settings`, `severity_from_score`, `mitre_for_attack`, `MITRE_MAPPING` | Imported by **every** page; loads `detection_models.pkl` + `classifier.pkl` + `deviation_features.csv` + `access_logs.csv`; imports `build_explanation`, `FEATURE_LABELS` from `src/explain.py` |
| `dashboard.py` | Entry point / Home — KPI cards, threat trend, severity overview, recent alerts | — | `streamlit run app/dashboard.py` |
| `pages/1_Live_Alerts.py` | Filterable ranked alert queue + investigation handoff | `render_alert_queue` | thin wrapper over `soc_shared` |
| `pages/2_Investigation.py` | Selected-alert deep dive: SHAP chart, timeline, gauges, MITRE, actions | `get_selected_context`, `build_shap_chart`, `render_recommendations` | thin wrapper |
| `pages/3_Entity_Search.py` | Entity profile lookup + active alerts | `render_entity_search` | thin wrapper |
| `pages/4_Analytics.py` | ROC, confusion matrix, attack distribution, FP table | `build_roc_figure`, `build_confusion_figure`, `build_attack_distribution` | thin wrapper |
| `pages/5_Model_Health.py` | Component health + inference KPIs | `render_model_health` | thin wrapper |
| `pages/6_Settings.py` | Review threshold, replay mode, theme, CSV export | `render_settings` | thin wrapper |

## `data/`, `reports/`, root

| Path | Contents | Notes |
|:---|:---|:---|
| `data/raw/access_logs.csv` | 66,282 sessions, features only | ~13 MB; schema in `data_dictionary.md` |
| `data/raw/data_dictionary.md` | Schema + assumptions + attack taxonomy | documents the `auth_success` extension |
| `data/labels/labels.csv` | `session_id → label` (held out) | joined only for supervised training + eval |
| `data/processed/*.pkl` | `detection_models.pkl`, `classifier.pkl` | consumed by dashboard; version-coupled |
| `data/processed/*.csv` | `deviation_features.csv`, `entity_profiles.csv`, `explained_alerts.csv` | pipeline outputs |
| `reports/final_report.md` | Full narrative report (12 sections + appendices) | git-ignored (`.gitignore`) |
| `reports/figures/` | EDA + classification figures, per-class CSV | evaluation evidence |
| `requirements.txt` | 11 deps, `>=` pins | version drift → pickle warnings |
| `README.md` | Project overview + setup + live demo link | — |
| `Files/` | Problem statement, tracker, `.docx`/`.pptx` | git-ignored background material |

## Dependency graph (imports)
- `features.py` → imports `profiling.EntityProfiler`, `profiling.build_population_profiles`
- `app/soc_shared.py` → imports `src/explain.build_explanation`, `src/explain.FEATURE_LABELS` (via `sys.path` insert)
- every `app/pages/*.py` → imports from `app/soc_shared.py`
- `dashboard.py` → imports from `soc_shared`

## Flags to remember
- ⚠️ `data_generator.py __main__` writes to `/home/claude/...` (edit to relative paths to regen on Windows).
- ⚠️ `MITRE_MAPPING` keys `data_exfiltration`/`privilege_escalation` ≠ actual labels (`exfiltration`; none for priv-esc).
- ⚠️ `entity_profiler.pkl` is written by `profiling.py` but not committed / not loaded.
