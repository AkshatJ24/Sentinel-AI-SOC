# Data Dictionary — Synthetic Behavioral Access Logs

## Files
- `data/raw/access_logs.csv` — 66,282 sessions, features only (no `label`)
- `data/labels/labels.csv` — `session_id` → `label`, held out separately. Never joined into unsupervised training; only used for supervised comparison models and final evaluation.

## Schema

| Field | Type | Notes |
|---|---|---|
| session_id | str | unique row id |
| entity_id | str | `usr_/svc_/edg_` + hash, ties back to entity type |
| entity_type | categorical | user / service_account / edge_device |
| timestamp | datetime | session start |
| source_ip | str | attacker sessions use random IPs outside entity's home `/24`-ish prefix |
| geo_location | str | city name; `"unknown"` for spoofed/brute-force traffic with no resolvable geo |
| lat / lon | float | for geo-velocity feature engineering (Phase 4) |
| resource_accessed | str | one of 28 catalog items (API endpoints, file shares, ports, device functions) |
| auth_method | categorical | password / token / certificate / biometric |
| **auth_success** | bool | **extension beyond the suggested schema** — required to make brute force / credential stuffing meaningful. Documented here per Phase 1.12. |
| session_duration | float (min) | log-normal per entity |
| command_sequence | list[str] | only populated for ~15% of normal sessions (privileged/service actions) and attack types that plausibly involve multi-step action (lateral movement, exfiltration, spoofing) |
| device_fingerprint | str | `OS\|MAC\|protocol` |
| label | categorical | ground truth, held-out file only |

## Assumptions
- 250 entities, 45-day simulation window, seeded (reproducible: seed=42).
- Each entity has one stable "home" city/IP prefix/device fingerprint/resource subset (4–10 resources) sampled once — this **is** the ground-truth baseline that Phase 3's profiler is expected to re-derive statistically.
- `edge_device` entities have wider login-hour spread (always-on) and higher session frequency than `user`/`service_account`.
- Attack incidents are injected as discrete events (not per-row), then exploded into rows — e.g. one brute-force incident = 15–40 rows.
- Overall injection rate: **2.11%** of final dataset (within the 0.5–3% target band).
- `insider_drift` is intentionally the weakest signal (normal duration, normal hours) — it is not meant to be reliably caught by the detector; it exists to measure/tune false-positive behavior, per the problem statement.

## Attack taxonomy → distinguishing signal (validated, see EDA)
| Type | Primary signal | Rows |
|---|---|---|
| brute_force | auth_success ≈0.5%, ~30 attempts/IP in minutes | 630 |
| credential_stuffing | many entity_ids, one IP, auth_success ≈5% | 296 |
| lateral_movement | short sessions (6.5min vs 90min baseline), resources outside entity's subset | 208 |
| exfiltration | 52% off-hours (vs 27% baseline), 61% carry `export` command, sensitive resources | 114 |
| insider_drift | near-normal duration/hours; only new-resource drift signals it — ambiguous by design | 63 |
| impossible_travel | paired sessions, >1000km apart, gap too short to travel | 32 |
| device_spoofing | same entity/device_id, fingerprint mismatch vs history | 25 |

## Known limitations (feeds Phase 8 report)
- Synthetic data has no real network/log noise (packet loss, malformed fields, clock skew).
- `command_sequence` vocabulary is small (8 actions) — real privileged-session logs would be far richer.
- Incident counts scaled off `n_normal // k` heuristics rather than a calibrated real-world base rate.
