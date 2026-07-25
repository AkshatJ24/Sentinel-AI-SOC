"""
Phase 4.1-4.2 — Deviation feature engineering.

Single chronological pass over all sessions. For every row, the profile
snapshot (from profiling.EntityProfiler) and the Markov transition
probability are read BEFORE that row updates any state — so every
feature is "as of just before this session," matching how a real-time
detector would see it. No future leakage.

Features produced:
- geo_velocity_kmh      : implied travel speed vs entity's previous session
- time_since_last_hr    : hours since entity's previous session
- hour_zscore           : |hour - profile.hour_mean| / profile.hour_std
- resource_novelty      : 1 if resource outside entity's known top-resource set
- fingerprint_mismatch  : 1 if device_fingerprint differs from entity's last seen one
- trailing_auth_failures: count of failed auths from this source_ip in the last 5 min
- transition_novelty    : 1 - P(resource | previous resource) from per-entity Markov counts
- profile_confidence    : cold-start confidence at time of session (0=new entity, 1=established)
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter, deque

from profiling import EntityProfiler, build_population_profiles

FAILURE_WINDOW_MIN = 5


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def engineer_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df = df.sort_values("timestamp").reset_index(drop=True)

    pop_profiles = build_population_profiles(df)
    profiler = EntityProfiler(pop_profiles)

    last_session = {}                              # entity_id -> (ts, lat, lon)
    markov_counts = defaultdict(lambda: defaultdict(Counter))  # entity -> prev_res -> Counter(next_res)
    prev_resource = {}                              # entity_id -> last resource
    ip_failure_times = defaultdict(deque)           # source_ip -> deque[timestamps of failures]

    out_rows = []

    for row in df.itertuples(index=False):
        entity = row.entity_id
        ts = row.timestamp

        # --- geo velocity / time since last session ---
        # Gaps under 2 minutes are treated as the same physical presence (login
        # burst / multiple resources in one sitting) rather than travel — otherwise
        # small geo jitter over a near-zero time delta produces absurd velocities.
        MIN_GAP_HOURS = 2 / 60
        geo_velocity, time_since_last = 0.0, -1.0  # -1 sentinel = no prior session (cold start)
        if entity in last_session:
            prev_ts, prev_lat, prev_lon = last_session[entity]
            dt_hours = (ts - prev_ts).total_seconds() / 3600
            time_since_last = max(dt_hours, 0.0)
            if dt_hours >= MIN_GAP_HOURS and not (
                np.isnan(prev_lat) or np.isnan(prev_lon) or np.isnan(row.lat) or np.isnan(row.lon)
            ):
                dist = haversine_km(prev_lat, prev_lon, row.lat, row.lon)
                geo_velocity = dist / dt_hours
        last_session[entity] = (ts, row.lat, row.lon)

        # --- profile snapshot (pre-update, cold-start-blended) ---
        eff = profiler.get_effective_profile(entity, row.entity_type)
        hour_zscore = abs(row.hour - eff["hour_mean"]) / eff["hour_std"]
        resource_novelty = 0 if row.resource_accessed in eff["known_resources"] else 1
        known_fp = eff["known_device_fp"]
        fingerprint_mismatch = 1 if (known_fp is not None and row.device_fingerprint != known_fp) else 0

        # --- Markov transition novelty ---
        prev_res = prev_resource.get(entity)
        if prev_res is not None:
            counts = markov_counts[entity][prev_res]
            total = sum(counts.values())
            prob = counts.get(row.resource_accessed, 0) / total if total > 0 else 0.0
            transition_novelty = 1 - prob
        else:
            transition_novelty = 0.5  # no prior action to compare against
        if prev_res is not None:
            markov_counts[entity][prev_res][row.resource_accessed] += 1
        prev_resource[entity] = row.resource_accessed

        # --- trailing auth failures for this source_ip ---
        dq = ip_failure_times[row.source_ip]
        window_start = ts - pd.Timedelta(minutes=FAILURE_WINDOW_MIN)
        while dq and dq[0] < window_start:
            dq.popleft()
        trailing_auth_failures = len(dq)
        if not row.auth_success:
            dq.append(ts)

        out_rows.append({
            "session_id": row.session_id,
            "entity_id": entity,
            "entity_type": row.entity_type,
            "timestamp": ts,
            "geo_velocity_kmh": geo_velocity,
            "time_since_last_hr": time_since_last,
            "hour_zscore": hour_zscore,
            "resource_novelty": resource_novelty,
            "fingerprint_mismatch": fingerprint_mismatch,
            "trailing_auth_failures": trailing_auth_failures,
            "transition_novelty": transition_novelty,
            "profile_confidence": eff["confidence"],
            "auth_success": int(row.auth_success),
            "session_duration": row.session_duration,
        })

        # --- update profile state AFTER features computed (no leakage) ---
        profiler.update(
            entity_id=entity, entity_type=row.entity_type, timestamp=ts, hour=row.hour,
            resource=row.resource_accessed, auth_method=row.auth_method,
            geo=row.geo_location, duration=row.session_duration,
            device_fingerprint=row.device_fingerprint,
        )

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    feat = pd.read_csv("data/raw/access_logs.csv")
    lab = pd.read_csv("data/labels/labels.csv")
    df = feat.merge(lab, on="session_id")

    dev_features = engineer_features(df)
    dev_features = dev_features.merge(lab, on="session_id")
    dev_features.to_csv("data/processed/deviation_features.csv", index=False)

    print(f"Built {len(dev_features)} feature rows, {dev_features.shape[1]} columns")
    print("\nMean feature values by label (sanity check — attack rows should stand out):")
    numeric_cols = ["geo_velocity_kmh", "time_since_last_hr", "hour_zscore",
                     "resource_novelty", "fingerprint_mismatch", "trailing_auth_failures",
                     "transition_novelty"]
    print(dev_features.groupby("label")[numeric_cols].mean().round(2))
