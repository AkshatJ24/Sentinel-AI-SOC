# 🛡️ Sentinel AI — Interview Quick Revision

> Read this 10–15 minutes before the interview. Full detail lives in `PROJECT_MASTER_GUIDE.md`.

## The pitch (30s)
Sentinel AI is an end-to-end behavioral anomaly-detection pipeline for cybersecurity access logs. It
generates seeded synthetic logs for 250 entities over 45 days, profiles each entity's normal behavior
online with an EMA, engineers 10 leakage-free deviation features, and scores every session with XGBoost.
It's evaluated at a **top-1% alert budget** (an analyst's real review capacity) — where it hits **100%
precision / 48% recall / 0.65 F1**. A second model classifies the attack type, SHAP explains each alert
in plain English, and a Streamlit SOC dashboard ties it together.

## Architecture (say this)
Offline `src/` pipeline → committed `.pkl`/`.csv` artifacts → online `app/` Streamlit dashboard that only
*loads* artifacts (never retrains). Seven phases: **generate → EDA → profile → features → detect → classify → explain → dashboard.**

```
data_generator → eda
              ↘ profiling → features → detection_model ┐
                                     ↘ classifier      ├→ data/processed/*.pkl → soc_shared.build_workspace → pages/*
                           detection → explain ────────┘
```

## Key technologies & why
| Tech | Why |
|:---|:---|
| XGBoost (+SMOTE) | Strong tabular detector; SMOTE (train-only) fixes 98/2 imbalance |
| Isolation Forest | Unsupervised baseline — works with no labels (production reality) |
| Custom EMA profiler | O(1)/entity online baseline; handles cold start + concept drift |
| SHAP TreeExplainer | Exact per-alert attributions → natural-language reasons |
| Markov transitions | Lightweight, explainable sequential signal (vs. heavy LSTM) |
| Streamlit | Fastest Python → interactive analyst UI |

## Verified numbers (safe to quote)
- Data: **66,282 sessions**, 250 entities, 45 days, **2.11%** anomaly rate.
- Detection @ top-1% (13,419 test sessions, 280 anomalies, 134 alerts): **XGBoost P 1.000 / R 0.479 / F1 0.647**; Isolation Forest **0.784 / 0.375 / 0.507**; primary = XGBoost.
- Classifier per-class F1: brute_force 0.990, credential_stuffing 0.982, impossible_travel 1.000, lateral_movement 0.964, insider_drift 0.941, device_spoofing 0.923, exfiltration 0.906.

## The 10 features
geo_velocity_kmh · time_since_last_hr · hour_zscore · resource_novelty · fingerprint_mismatch ·
trailing_auth_failures · transition_novelty · profile_confidence · auth_success · session_duration
(+ one-hot entity_type). **All computed from the profile snapshot BEFORE the row updates state → no leakage.**

## Top 25 questions — one-line triggers
1. 2-min overview → §2B. 2. Why budget metric → fixed analyst capacity. 3. Leakage prevention → profile-before-update, SMOTE/scaler train-only. 4. Actual numbers → above. 5. Synthetic data → validates methodology not real perf. 6. EMA → recency-weighted, α=0.15, cold-start blend. 7. Cold start → c·personal+(1−c)·population, c=min(n/15,1). 8. Drift → EMA decays stale behavior. 9. EMA weakness → attack poisoning; fix = risk-gated updates. 10. XGB vs IF → higher top-1% F1. 11. SMOTE train-only → avoid test leakage. 12. Class weights not SMOTE → classes too small (6–25). 13. SHAP → top-3 positive features → templates. 14. The 10 features → above. 15. Markov vs LSTM → lightweight+explainable. 16. Dashboard flow → cached build_workspace. 17. Budget slider consistency → dedicated session-state key. 18. Per-class results → above. 19. Stratified vs temporal → fair for MVP, temporal stricter. 20. insider_drift → weak by design (FP tuning). 21. Mine vs library → orchestration/profiler/features/eval/explanations/dashboard mine; estimators library. 22. Productionize → Kafka→Flink→FastAPI→TSDB→WebSocket (not built). 23. Explanation trust → exact SHAP + sanity checks. 24. MITRE bug → `data_exfiltration` key ≠ `exfiltration` label. 25. Limitations → synthetic, poisoning, Markov memory, no tests, version pins.

## Five design decisions to defend
1. Evaluate at a **top-1% alert budget**, not accuracy. 2. **SMOTE on the training split only.** 3. **Class
weights** (not SMOTE) in the classifier. 4. **Markov** sequential signal over an LSTM. 5. **Stratified random**
split (fair here; temporal for production).

## Five honest limitations
Synthetic data (no real noise) · EMA profiler poisoning · Markov one-step memory · no tests/DB/CI ·
pickle↔library version coupling (sklearn/XGBoost warnings). **Bonus (shows rigor):** the `MITRE_MAPPING`
label mismatch means exfiltration alerts show a generic technique.

## Never claim without evidence
Real-world deployment or impact · measured analyst time savings · that 100% precision survives on real data.

## "Why should we care?"
"It takes the unglamorous-but-real problems — imbalance, leakage, cold start, drift, explainability —
seriously, and measures success the way a SOC actually operates: at a fixed alert budget."
