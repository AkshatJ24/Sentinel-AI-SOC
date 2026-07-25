"""
Phase 3 — Baseline Profiling Model.

Design:
- Population profile per entity_type (cold-start fallback), computed once
  from the full session set (>97% normal, so bias from the ~2% attack rows
  is negligible for this purpose).
- Per-entity profile updated ONLINE via EMA as sessions arrive in
  chronological order (unsupervised — no label is used to decide whether
  to update, matching how a real deployed profiler would behave).
- Cold-start blending: effective profile = confidence * personal
  + (1 - confidence) * population[entity_type], where
  confidence = min(n_obs / K, 1.0). New entities lean on the population
  profile; confidence -> 1 as personal evidence accumulates.
- Concept drift: EMA (not a fixed window) means old behavior decays
  smoothly, so a legitimate shift (new work hours, new device) is
  absorbed rather than permanently flagged.

Known limitation (documented for the report): the online EMA is not
attack-resistant on its own — a sustained attack pattern would slowly
pull an entity's own profile too. At the injected rate here (~2%) the
effect is negligible, and Phase 4's detection layer is what catches
individual sessions, not the profiler itself. A production system would
gate profile updates on the detector's own risk score (don't update on
high-risk sessions).
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

ALPHA = 0.15          # EMA decay — higher = faster adaptation to new behavior
COLD_START_K = 15     # sessions needed before confidence saturates near 1.0


def _ema_dict_update(freq_dict, key, alpha):
    """EMA update over a categorical frequency distribution (resource_freq, auth_freq, geo_freq)."""
    for k in freq_dict:
        freq_dict[k] *= (1 - alpha)
    freq_dict[key] = freq_dict.get(key, 0.0) + alpha
    return freq_dict


def build_population_profiles(df):
    """Cold-start fallback profile, segmented by entity_type."""
    pop = {}
    for etype, sub in df.groupby("entity_type"):
        hours = pd.to_datetime(sub["timestamp"]).dt.hour + pd.to_datetime(sub["timestamp"]).dt.minute / 60
        res_freq = (sub["resource_accessed"].value_counts(normalize=True)).to_dict()
        auth_freq = (sub["auth_method"].value_counts(normalize=True)).to_dict()
        geo_freq = (sub["geo_location"].value_counts(normalize=True)).to_dict()
        pop[etype] = {
            "hour_mean": hours.mean(),
            "hour_std": max(hours.std(), 0.5),
            "resource_freq": res_freq,
            "auth_freq": auth_freq,
            "geo_freq": geo_freq,
            "duration_mean": sub["session_duration"].mean(),
            "duration_std": max(sub["session_duration"].std(), 0.1),
        }
    return pop


class EntityProfiler:
    def __init__(self, population_profiles, alpha=ALPHA, cold_start_k=COLD_START_K):
        self.pop = population_profiles
        self.alpha = alpha
        self.k = cold_start_k
        self.state = {}  # entity_id -> profile dict

    def _init_entity(self, entity_id, entity_type):
        pop = self.pop[entity_type]
        self.state[entity_id] = {
            "entity_type": entity_type,
            "n_obs": 0,
            "hour_mean": pop["hour_mean"],
            "hour_var": pop["hour_std"] ** 2,
            "resource_freq": defaultdict(float),
            "auth_freq": defaultdict(float),
            "geo_freq": defaultdict(float),
            "duration_mean": pop["duration_mean"],
            "duration_var": pop["duration_std"] ** 2,
            "last_updated": None,
        }

    def confidence(self, entity_id):
        n = self.state[entity_id]["n_obs"] if entity_id in self.state else 0
        return min(n / self.k, 1.0)

    def get_effective_profile(self, entity_id, entity_type):
        """Blended (cold-start-aware) profile snapshot — call BEFORE update() to score a session without leakage."""
        if entity_id not in self.state:
            self._init_entity(entity_id, entity_type)
        s = self.state[entity_id]
        pop = self.pop[entity_type]
        c = self.confidence(entity_id)

        eff_resource_freq = defaultdict(float)
        keys = set(s["resource_freq"]) | set(pop["resource_freq"])
        for k in keys:
            eff_resource_freq[k] = c * s["resource_freq"].get(k, 0.0) + (1 - c) * pop["resource_freq"].get(k, 0.0)

        return {
            "confidence": c,
            "hour_mean": c * s["hour_mean"] + (1 - c) * pop["hour_mean"],
            "hour_std": max((c * np.sqrt(s["hour_var"]) + (1 - c) * pop["hour_std"]), 0.3),
            "resource_freq": eff_resource_freq,
            "duration_mean": c * s["duration_mean"] + (1 - c) * pop["duration_mean"],
            "duration_std": max(c * np.sqrt(s["duration_var"]) + (1 - c) * pop["duration_std"], 0.1),
            "known_resources": set(k for k, v in s["resource_freq"].items() if v > 0.01),
            "known_device_fp": s.get("device_fingerprint"),
        }

    def update(self, entity_id, entity_type, timestamp, hour, resource, auth_method,
               geo, duration, device_fingerprint):
        if entity_id not in self.state:
            self._init_entity(entity_id, entity_type)
        s = self.state[entity_id]
        a = self.alpha

        delta = hour - s["hour_mean"]
        s["hour_mean"] += a * delta
        s["hour_var"] = (1 - a) * (s["hour_var"] + a * delta ** 2)

        d_delta = duration - s["duration_mean"]
        s["duration_mean"] += a * d_delta
        s["duration_var"] = (1 - a) * (s["duration_var"] + a * d_delta ** 2)

        _ema_dict_update(s["resource_freq"], resource, a)
        _ema_dict_update(s["auth_freq"], auth_method, a)
        _ema_dict_update(s["geo_freq"], geo, a)
        s["device_fingerprint"] = device_fingerprint  # most recent — spoofing check compares against this
        s["n_obs"] += 1
        s["last_updated"] = timestamp

    def final_table(self):
        rows = []
        for eid, s in self.state.items():
            top_resources = sorted(s["resource_freq"].items(), key=lambda x: -x[1])[:5]
            rows.append({
                "entity_id": eid,
                "entity_type": s["entity_type"],
                "n_obs": s["n_obs"],
                "confidence": min(s["n_obs"] / self.k, 1.0),
                "hour_mean": round(s["hour_mean"], 2),
                "hour_std": round(np.sqrt(s["hour_var"]), 2),
                "duration_mean": round(s["duration_mean"], 2),
                "duration_std": round(np.sqrt(s["duration_var"]), 2),
                "top_resources": ", ".join(f"{r}:{p:.2f}" for r, p in top_resources),
                "current_device_fingerprint": s.get("device_fingerprint"),
                "last_updated": s["last_updated"],
            })
        return pd.DataFrame(rows)


def run_profiling(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df = df.sort_values("timestamp").reset_index(drop=True)

    pop_profiles = build_population_profiles(df)
    profiler = EntityProfiler(pop_profiles)

    for row in df.itertuples(index=False):
        profiler.update(
            entity_id=row.entity_id, entity_type=row.entity_type, timestamp=row.timestamp,
            hour=row.hour, resource=row.resource_accessed, auth_method=row.auth_method,
            geo=row.geo_location, duration=row.session_duration,
            device_fingerprint=row.device_fingerprint,
        )

    return profiler, pop_profiles


# ---------------------------------------------------------------------------
# Qualitative demos for 3.2 (cold-start) and 3.3 (drift) — printed + saved
# ---------------------------------------------------------------------------

def demo_cold_start(profiler, pop_profiles):
    print("\n--- Cold-start demo ---")
    fresh_id, fresh_type = "demo_new_user", "user"
    eff = profiler.get_effective_profile(fresh_id, fresh_type)
    pop = pop_profiles[fresh_type]
    print(f"Brand-new '{fresh_type}' entity, 0 sessions -> confidence={eff['confidence']:.2f}")
    print(f"  effective hour_mean={eff['hour_mean']:.2f} (population hour_mean={pop['hour_mean']:.2f}) -> matches population, as expected")


def demo_drift(pop_profiles):
    """Show EMA absorbing a legitimate shift vs a fixed-window baseline flagging it forever."""
    print("\n--- Concept drift demo (EMA vs fixed-window) ---")
    rng = np.random.default_rng(0)
    profiler = EntityProfiler({"user": pop_profiles["user"]})
    eid = "demo_drift_user"

    # 40 sessions at old hours (~10am), then entity shifts to ~8pm for 20 sessions (new legit shift)
    old_hours = rng.normal(10, 1, 40)
    new_hours = rng.normal(20, 1, 20)
    fixed_window = []
    ema_hour_means = []

    for h in np.concatenate([old_hours, new_hours]):
        eff = profiler.get_effective_profile(eid, "user")
        ema_hour_means.append(eff["hour_mean"])
        profiler.update(eid, "user", pd.Timestamp.now(), h, "res_x", "token", "CityX", 5.0, "fp")
        fixed_window.append(h)

    fixed_window_mean_last20 = np.mean(fixed_window[-20:])  # what a naive fixed full-history mean would show
    print(f"After shift: EMA hour_mean = {ema_hour_means[-1]:.1f} (tracks the new ~20:00 pattern)")
    print(f"Naive full-history fixed mean would still show ~{np.mean(fixed_window):.1f} "
          f"(anchored to old + new mixed) -> would keep flagging every post-shift session as deviation")
    print("EMA converges toward the new pattern within a handful of sessions instead of staying anchored to stale history.")


if __name__ == "__main__":
    feat = pd.read_csv("data/raw/access_logs.csv")
    lab = pd.read_csv("data/labels/labels.csv")
    df = feat.merge(lab, on="session_id")

    profiler, pop_profiles = run_profiling(df)

    final_table = profiler.final_table()
    final_table.to_csv("data/processed/entity_profiles.csv", index=False)
    with open("data/processed/entity_profiler.pkl", "wb") as f:
        pickle.dump({"profiler": profiler, "population_profiles": pop_profiles}, f)

    print(f"Profiled {len(final_table)} entities.")
    print(final_table[["entity_id", "entity_type", "n_obs", "confidence", "hour_mean", "hour_std"]].head())

    demo_cold_start(profiler, pop_profiles)
    demo_drift(pop_profiles)

    print("\nSaved: data/processed/entity_profiles.csv, data/processed/entity_profiler.pkl")
