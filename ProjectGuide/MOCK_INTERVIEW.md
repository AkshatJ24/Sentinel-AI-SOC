# 🛡️ Sentinel AI — Mock Interview

> A realistic progression from warm-up to hard follow-ups, with model answers and a scoring rubric.
> Practice speaking the answers aloud; keep them concise and specific to the repo.

---

## Round 1 — Warm-up (5 min)

**Q1. Tell me about this project.**
*Model answer:* "Sentinel AI is an end-to-end behavioral anomaly-detection pipeline for security access
logs. Since real labelled intrusion data is hard to get, I built a seeded synthetic generator — 66,000
sessions, 250 entities, 45 days, seven attack types at a 2% rate. I profile each entity's normal behavior
online with an EMA, engineer ten leakage-free deviation features, and score sessions with XGBoost. The key
idea is evaluation at a top-1% alert budget — what an analyst can actually review — where the detector gets
100% precision and 48% recall. A second model classifies the attack type, SHAP explains each alert in plain
English, and a Streamlit dashboard serves it to an analyst."

**Q2. What was your role / what did you build yourself?**
*Model answer:* "I designed and wrote the whole pipeline: the synthetic generator, the EMA profiler, the
leakage-free feature pass, the budget-based evaluation harness, the explanation-template layer, and the
dashboard logic. The estimators — XGBoost, Isolation Forest, SMOTE, SHAP — are libraries I wired together;
I didn't reimplement them." *(Personalize the team/solo and timeline honestly.)*

**Q3. Who's the user and what do they do with it?**
*Model answer:* "A SOC analyst. They open the dashboard, see a ranked, explained alert queue, pick an alert,
and the Investigation page shows a SHAP breakdown, the entity timeline, a MITRE technique, and recommended
actions like lock account or require MFA."

---

## Round 2 — Core technical (15 min)

**Q4. Why not just measure accuracy?**
*Model answer:* "The set is 98% normal, so a model that predicts 'normal' for everything scores 98% accuracy
and catches zero attacks. I evaluate at a top-1% alert budget instead: mark the top 1% of sessions by risk
as alerts, then measure precision and recall there. That mirrors the real constraint — an analyst reviews a
fixed slice per day."

**Q5. Walk me through your features and how you avoid leakage.**
*Model answer:* "Ten deviation features — geo velocity, time since last session, hour z-score, resource and
transition novelty, fingerprint mismatch, trailing auth failures, profile confidence, auth success, and
duration. I build them in one chronological pass. The crucial part: for each row I read the entity's profile
snapshot and the Markov transition probability *before* that row updates any state, then update afterward.
So no feature ever sees information from its own session or the future."

**Q6. Explain the EMA profiler and the cold-start handling.**
*Model answer:* "Each entity has an exponential moving average of its behavior — login hour, duration,
resource/auth/geo frequencies — with α=0.15, so recent behavior is weighted more and legitimate drift is
absorbed. For a new entity with little history I blend: effective = c·personal + (1−c)·population, where
c = min(n_obs/15, 1). A brand-new entity leans entirely on the population profile for its type; confidence
ramps to 1 as evidence accumulates."

**Q7. Isolation Forest vs XGBoost+SMOTE — which won and why keep both?**
*Model answer:* "XGBoost+SMOTE is primary: at top-1% it gets F1 0.647 versus Isolation Forest's 0.507.
But I keep Isolation Forest because it needs no labels, which matches production reality where true
intrusions are rare and unlabelled. SMOTE is applied only to the training split — never the test set — so
synthetic rows can't leak into evaluation."

**Q8. How does explainability work?**
*Model answer:* "SHAP's TreeExplainer runs on the XGBoost detector to get per-alert feature attributions.
I take the top-3 features that *push toward* anomaly — positive SHAP only — and map them through
natural-language templates, so the analyst reads 'flagged due to implausible travel speed + device
fingerprint mismatch + a resource never touched before' instead of raw numbers. I added sanity checks:
brute-force alerts should mention failed logins, impossible-travel alerts should mention travel speed."

---

## Round 3 — Design & trade-offs (10 min)

**Q9. Why Markov transitions instead of an LSTM?**
*Model answer:* "The data is tabular with a modest anomaly set, I needed explainability, and it was a
hackathon timeline. Per-entity Markov transition novelty — 1 minus the probability of the current resource
given the previous one — gives a lightweight, explainable sequential signal. An LSTM would add training
cost and opacity for marginal gain here. I'd revisit it if longer-range sequential patterns mattered."

**Q10. Why class weights in the classifier but SMOTE in the detector?**
*Model answer:* "In the detector it's binary with enough minority examples for SMOTE to interpolate safely.
In the classifier, per-class counts drop to as low as 6 test rows for device spoofing — too few real
neighbors to synthesize meaningfully — so I use balanced class weights, which reweight the loss instead of
fabricating minority rows."

**Q11. Stratified random vs temporal split — defend it.**
*Model answer:* "I used a stratified random 80/20. The injected incidents are spread evenly across the full
45-day window, so a stratified split gives a clean apples-to-apples comparison for this MVP. A temporal split
is stricter and closer to production — I'd switch to it and report both before claiming real-world readiness."

---

## Round 4 — Hard follow-ups & honesty (10 min)

**Q12. Your detector shows 100% precision. Isn't that suspicious?**
*Model answer:* "On this seeded synthetic test set at a 1% budget, yes — all 134 flagged sessions are true
anomalies. But I'm clear that's synthetic, noise-free data. On real logs I'd expect precision to fall. The
number validates the ranking and methodology, not real-world performance."

**Q13. What's the weakest part of your system?**
*Model answer:* "Two things. The EMA profiler isn't attack-resistant — a slow, sustained attack would
gradually pull an entity's own baseline toward the malicious pattern. At a 2% injection rate the effect is
negligible, but in production I'd gate profile updates on the detector's risk score. And there's no automated
test suite — validation is EDA checks plus the SHAP sanity assertions, which I'd formalize with pytest."

**Q14. I looked at your code — the MITRE mapping has a bug. Explain.**
*Model answer:* "Good catch. `MITRE_MAPPING` has keys `data_exfiltration` and `privilege_escalation`, but my
generator emits the label `exfiltration` and has no privilege-escalation type. So exfiltration alerts don't
match a key and fall through to the generic fallback technique instead of T1041. The fix is to rename the
key to `exfiltration` (and drop the unused priv-esc entry or add a matching attack type)."

**Q15. What breaks at 100× the data?**
*Model answer:* "The in-memory Pandas DataFrames and the single-threaded Python feature loop. The algorithm
itself is single-pass O(N) with O(1) memory per entity, so it's streaming-friendly in principle — I'd move
it onto Kafka for ingestion and Flink or Spark for the feature pass, serve the model behind FastAPI, and put
entity profiles in a time-series database. That streaming path is sketched in my report but not implemented."

**Q16. What did you learn here that a tutorial wouldn't teach?**
*Model answer:* "Designing evaluation around an operational constraint instead of defaulting to accuracy,
and the discipline of profile-before-update to guarantee no leakage in a stateful online pass. Those are the
parts I had to reason through myself." *(Personalize.)*

---

## Scoring rubric (25 points)

| Dimension | Weight | What "strong" looks like |
|:---|:---:|:---|
| **Clarity of overview** | 4 | 2-min pitch hits problem → approach → budget metric → result without rambling |
| **Leakage & evaluation rigor** | 5 | Explains profile-before-update, SMOTE-train-only, and why the budget metric beats accuracy |
| **Depth on EMA / cold start / drift** | 4 | Gives α, the blend formula, and the poisoning weakness unprompted |
| **Model trade-offs** | 4 | Defends XGB-vs-IF, class-weights-vs-SMOTE, Markov-vs-LSTM with reasons |
| **Explainability** | 3 | Describes SHAP→templates, positive-only filtering, sanity checks |
| **Honesty on limitations** | 3 | Volunteers synthetic-data caveat, no-tests, the MITRE bug; never overclaims |
| **Production/scaling thinking** | 2 | Names a concrete streaming path and what they'd measure |

**Interpretation:** 21–25 excellent · 16–20 solid · 11–15 needs more depth on rigor/trade-offs · ≤10 revise fundamentals.

### Red flags to avoid
- Quoting accuracy as the headline metric.
- Claiming real-world deployment, impact, or that 100% precision holds on real data.
- Saying there are "no limitations."
- Claiming you implemented XGBoost/SHAP/SMOTE themselves.
