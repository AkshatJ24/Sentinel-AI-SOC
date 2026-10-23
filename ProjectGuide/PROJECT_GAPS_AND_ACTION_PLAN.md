# 🛡️ Sentinel AI — Gaps & Action Plan

> Honest inventory of unverified assumptions, missing tests, implementation gaps, and recommended next
> steps. Grounded in actual code inspection. Nothing here was changed in the repo — this is a to-do list.

---

## A. Confirmed implementation gaps (found in code)

| # | Gap | Where | Impact | Fix | Effort |
|:--|:---|:---|:---|:---|:---:|
| 1 | **MITRE label mismatch** — `MITRE_MAPPING` keys `data_exfiltration`, `privilege_escalation` don't match actual labels (`exfiltration`; no priv-esc type) | `app/soc_shared.py` | Exfiltration alerts show generic `T1078 / Behavioral anomaly` instead of `T1041` | Rename key to `exfiltration`; drop or back the unused `privilege_escalation` entry | Low |
| 2 | **Hardcoded Linux output paths** — `/home/claude/behavioral-anomaly-detection/...` | `src/data_generator.py` `__main__` | Regen step fails on Windows/other machines | Change to relative `data/raw/` and `data/labels/` paths (as other scripts use) | Low |
| 3 | **Unused artifact** — `entity_profiler.pkl` written but not committed and not loaded | `src/profiling.py` | Dead output; mild confusion | Either load it in `features.py` or stop writing it | Low |
| 4 | **Loose dependency pins** — `requirements.txt` uses `>=` | root | sklearn 1.8→1.9 + XGBoost unpickle warnings | Pin exact versions or re-export models with `Booster.save_model` | Low |
| 5 | **No automated tests** — no `tests/`, no pytest, no CI | repo-wide | Regressions (e.g. a leakage bug) would go unnoticed | Add a pytest suite (see §C) + a CI workflow | Medium |

---

## B. Unverified assumptions (need testing before you claim them)

1. **100% top-1% precision generalizes.** Verified only on the seeded synthetic test set. *Test:* re-run with
   several seeds; report variance. Expect a drop on any real/noisy data.
2. **Stratified-random split is representative.** Assumed because incidents are spread across 45 days. *Test:*
   add a temporal split and compare F1 at the same budget.
3. **α=0.15 and K=15 are good defaults.** Chosen, not swept. *Test:* grid-sweep α and K; measure precision/recall@budget.
4. **~2% injection rate is realistic.** Scaled off `n_normal // k` heuristics (documented), not a calibrated base rate.
5. **The live Streamlit demo is up.** The README links a Community Cloud URL; the repo can't confirm it's currently live. *Check manually.*

---

## C. Missing tests — prioritized plan

| Priority | Test | Sample input | Expected behavior |
|:---:|:---|:---|:---|
| P0 | **Leakage regression** | a tiny 3-session entity | a feature row never reflects its own session's profile update (profile read precedes `profiler.update`) |
| P0 | **Generator invariants** | `generate_dataset(n_entities=20)` | anomaly rate within 0.5–3%; all 7 labels present; no NaN in required non-geo fields |
| P1 | **Budget monotonicity** | cached `xgb_scores`, `y_test` | recall is non-decreasing as budget goes 0.5% → 1% → 2% → 5% |
| P1 | **Explanation coverage** | top-k alert SHAP rows | every alert yields a non-empty explanation string |
| P1 | **MITRE resolution** | each generator label | resolves to a non-fallback technique (currently **fails** for `exfiltration`) |
| P2 | **Scaler/feature-column contract** | `detection_models.pkl` | `feature_cols` matches the columns the dashboard builds (one-hot `etype_*` included) |

---

## D. Recommended next steps (ordered)

### Quick wins (hours)
1. Fix the MITRE key mismatch (gap #1) and verify T1041 shows for exfiltration in Investigation.
2. Relative-path the generator (gap #2) so regen works cross-platform.
3. Pin exact dependency versions (gap #4) to silence unpickle warnings and lock reproducibility.

### Medium (days)
4. Add the P0/P1 pytest suite (§C) and a minimal GitHub Actions CI.
5. Add a temporal-split evaluation path to `detection_model.py`; report both metrics.
6. Parameterize α/K per entity type; add a small sweep script.

### Advanced (weeks)
7. **Risk-gated profile updates** — don't update an entity's EMA on high-risk sessions (attack-resistant profiling).
8. **FastAPI scoring service** — load the pickle once, POST a session → risk score + explanation.
9. **Streaming path** — Kafka/Kinesis → Flink/Spark features → model serving → time-series DB → WebSocket dashboard
   (sketched in `reports/final_report.md` §10, not built).
10. Consider a sequence model (GRU/Transformer) layered on the deviation features if longer-range patterns prove valuable.

---

## E. Questions only the author can answer
- Timeline, solo vs. team, and your specific role on the project.
- Whether this was a graded/submitted hackathon entry and any feedback received.
- Whether the hosted Streamlit demo is currently live.
- Intended resolution for the MITRE mapping and the hardcoded paths (left unmodified here per the safety rule).

---

## F. What is **not** broken (so you don't over-apologize)
- The core leakage discipline is correct (profile read before update).
- SMOTE is correctly confined to the training split.
- The budget-based evaluation is sound and the cited metrics are reproducible from committed artifacts.
- The dashboard correctly avoids retraining via caching.
No full rewrite is warranted — these are targeted, mostly low-effort improvements.
