"""
Synthetic behavioral access-log generator.
Schema: entity_id, entity_type, timestamp, source_ip, geo_location,
        resource_accessed, auth_method, auth_success, session_duration,
        command_sequence, device_fingerprint, label

NOTE: `auth_success` is added beyond the suggested schema — brute force /
credential stuffing are undefined without a pass/fail signal. Documented
in data_dictionary.md.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

SIM_START = datetime(2026, 6, 1)
SIM_DAYS = 45

# ---------------------------------------------------------------------------
# Reference catalogs
# ---------------------------------------------------------------------------

CITIES = [
    ("Mumbai", 19.0760, 72.8777), ("Delhi", 28.7041, 77.1025),
    ("Bengaluru", 12.9716, 77.5946), ("Pune", 18.5204, 73.8567),
    ("Hyderabad", 17.3850, 78.4867), ("Chennai", 13.0827, 80.2707),
    ("Singapore", 1.3521, 103.8198), ("London", 51.5072, -0.1276),
    ("New York", 40.7128, -74.0060), ("Frankfurt", 50.1109, 8.6821),
    ("Tokyo", 35.6762, 139.6503), ("Sydney", -33.8688, 151.2093),
]

RESOURCE_CATALOG = (
    [f"/api/v1/{r}" for r in
     ["users", "billing", "reports", "orders", "inventory", "auth",
      "config", "logs", "payments", "customers", "analytics", "admin"]]
    + [f"file://share/{f}" for f in
       ["finance", "hr", "engineering", "legal", "exec", "public"]]
    + [f"port:{p}" for p in [22, 443, 3389, 502, 1883, 8080]]
    + [f"device_fn:{d}" for d in ["read_sensor", "write_config",
                                   "firmware_update", "reboot"]]
)

AUTH_METHODS = ["password", "token", "certificate", "biometric"]

OS_LIST = ["Windows 11", "Ubuntu 22.04", "macOS 14", "iOS 17",
           "Android 14", "FirmwareOS 3.2"]

ACTION_VOCAB = ["list", "read", "write", "download", "delete",
                "modify_permissions", "export", "execute"]

ENTITY_TYPES = ["user", "user", "user", "service_account", "edge_device"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def mac_addr():
    return ":".join(f"{rng.integers(0, 256):02x}" for _ in range(6))


# ---------------------------------------------------------------------------
# Phase 1.1 / 1.2 — entity roster + behavioral profile
# ---------------------------------------------------------------------------

def build_entities(n=250):
    entities = []
    for _ in range(n):
        etype = random.choice(ENTITY_TYPES)
        eid = f"{etype[:3]}_{uuid.uuid4().hex[:8]}"
        home_city = random.choice(CITIES)
        n_resources = rng.integers(4, 10)
        resource_subset = list(rng.choice(RESOURCE_CATALOG, size=n_resources, replace=False))

        auth_pref = rng.dirichlet(np.ones(len(AUTH_METHODS)) * 2)

        profile = {
            "entity_id": eid,
            "entity_type": etype,
            "home_city": home_city[0],
            "home_lat": home_city[1],
            "home_lon": home_city[2],
            "login_hour_mean": rng.uniform(7, 20) if etype != "edge_device" else rng.uniform(0, 24),
            "login_hour_std": rng.uniform(0.7, 2.5) if etype != "edge_device" else rng.uniform(3, 8),
            "resource_subset": resource_subset,
            "auth_methods": AUTH_METHODS,
            "auth_probs": auth_pref,
            "session_dur_mu": rng.uniform(2.5, 5.5),   # lognormal params, minutes
            "session_dur_sigma": rng.uniform(0.3, 0.8),
            "device_fingerprint": f"{random.choice(OS_LIST)}|{mac_addr()}|{'TLS1.3' if etype!='edge_device' else 'MQTT'}",
            "home_ip_prefix": f"{rng.integers(10,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}",
            "sessions_per_day_lambda": rng.uniform(0.5, 4) if etype != "edge_device" else rng.uniform(5, 40),
        }
        entities.append(profile)
    return entities


# ---------------------------------------------------------------------------
# Phase 1.3 — normal session generation
# ---------------------------------------------------------------------------

def gen_command_sequence(is_privileged):
    if not is_privileged or rng.random() > 0.4:
        return []
    length = rng.integers(2, 6)
    return list(rng.choice(ACTION_VOCAB, size=length, replace=True))


def sample_geo(profile, day_city):
    """Normal sessions use the day's assigned city (home, or a rare whole-day trip)
    with small jitter — avoids physically-impossible within-day city flips."""
    city_name, city_lat, city_lon = day_city
    return city_name, city_lat + rng.normal(0, 0.05), city_lon + rng.normal(0, 0.05)


def gen_normal_sessions(profile, travel_day_prob=0.02):
    rows = []
    for day in range(SIM_DAYS):
        # Decide travel ONCE per day, not per session — a legit trip holds for the
        # whole day instead of flip-flopping between home/remote city within minutes.
        if rng.random() < travel_day_prob:
            day_city = random.choice(CITIES)
        else:
            day_city = (profile["home_city"], profile["home_lat"], profile["home_lon"])

        n_sessions = rng.poisson(profile["sessions_per_day_lambda"])
        for _ in range(n_sessions):
            hour = np.clip(rng.normal(profile["login_hour_mean"], profile["login_hour_std"]), 0, 23.98)
            ts = SIM_START + timedelta(days=day, hours=float(hour))
            city, lat, lon = sample_geo(profile, day_city)
            resource = random.choice(profile["resource_subset"])
            auth_method = rng.choice(profile["auth_methods"], p=profile["auth_probs"])
            duration = float(rng.lognormal(profile["session_dur_mu"], profile["session_dur_sigma"]))
            is_priv = profile["entity_type"] in ("service_account",) or rng.random() < 0.15
            rows.append({
                "entity_id": profile["entity_id"],
                "entity_type": profile["entity_type"],
                "timestamp": ts,
                "source_ip": f"{profile['home_ip_prefix']}.{rng.integers(1,255)}",
                "geo_location": city,
                "lat": lat, "lon": lon,
                "resource_accessed": resource,
                "auth_method": auth_method,
                "auth_success": True,
                "session_duration": round(duration, 2),
                "command_sequence": gen_command_sequence(is_priv),
                "device_fingerprint": profile["device_fingerprint"],
                "label": "normal",
            })
    return rows


# ---------------------------------------------------------------------------
# Phase 1.4–1.10 — attack injection functions
# Each takes (profiles, existing_df) -> list[dict] of new rows (label set)
# ---------------------------------------------------------------------------

def inject_brute_force(profiles, n_incidents=6):
    rows = []
    for _ in range(n_incidents):
        target = random.choice(profiles)
        attacker_ip = f"{rng.integers(1,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(1,255)}"
        day = rng.integers(0, SIM_DAYS)
        start = SIM_START + timedelta(days=int(day), hours=float(rng.uniform(0, 23)))
        n_attempts = rng.integers(15, 40)
        for i in range(n_attempts):
            ts = start + timedelta(seconds=int(i * rng.uniform(2, 8)))
            success = (i == n_attempts - 1) and rng.random() < 0.3
            rows.append({
                "entity_id": target["entity_id"], "entity_type": target["entity_type"],
                "timestamp": ts, "source_ip": attacker_ip,
                "geo_location": "unknown", "lat": np.nan, "lon": np.nan,
                "resource_accessed": "/api/v1/auth",
                "auth_method": "password", "auth_success": bool(success),
                "session_duration": round(float(rng.uniform(0.05, 0.3)), 2),
                "command_sequence": [],
                "device_fingerprint": "unknown|unknown|unknown",
                "label": "brute_force",
            })
    return rows


def inject_impossible_travel(profiles, n_incidents=8):
    rows = []
    for _ in range(n_incidents):
        target = random.choice(profiles)
        day = rng.integers(0, SIM_DAYS)
        t1 = SIM_START + timedelta(days=int(day), hours=float(rng.uniform(6, 18)))
        far_city = max(CITIES, key=lambda c: haversine_km(target["home_lat"], target["home_lon"], c[1], c[2]))
        gap_minutes = rng.uniform(10, 90)  # not enough time to physically travel
        t2 = t1 + timedelta(minutes=float(gap_minutes))

        rows.append({
            "entity_id": target["entity_id"], "entity_type": target["entity_type"],
            "timestamp": t1, "source_ip": f"{target['home_ip_prefix']}.{rng.integers(1,255)}",
            "geo_location": target["home_city"], "lat": target["home_lat"], "lon": target["home_lon"],
            "resource_accessed": random.choice(target["resource_subset"]),
            "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
            "auth_success": True, "session_duration": round(float(rng.lognormal(3, 0.5)), 2),
            "command_sequence": [], "device_fingerprint": target["device_fingerprint"],
            "label": "normal",
        })
        rows.append({
            "entity_id": target["entity_id"], "entity_type": target["entity_type"],
            "timestamp": t2, "source_ip": f"{rng.integers(1,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(1,255)}",
            "geo_location": far_city[0], "lat": far_city[1], "lon": far_city[2],
            "resource_accessed": random.choice(target["resource_subset"]),
            "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
            "auth_success": True, "session_duration": round(float(rng.lognormal(3, 0.5)), 2),
            "command_sequence": [], "device_fingerprint": target["device_fingerprint"],
            "label": "impossible_travel",
        })
    return rows


def inject_credential_stuffing(profiles, n_incidents=4):
    rows = []
    for _ in range(n_incidents):
        attacker_ip = f"{rng.integers(1,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(1,255)}"
        targets = list(rng.choice(profiles, size=int(rng.integers(20, 45)), replace=False))
        day = rng.integers(0, SIM_DAYS)
        start = SIM_START + timedelta(days=int(day), hours=float(rng.uniform(0, 23)))
        for i, target in enumerate(targets):
            ts = start + timedelta(seconds=int(i * rng.uniform(1, 4)))
            success = rng.random() < 0.05
            rows.append({
                "entity_id": target["entity_id"], "entity_type": target["entity_type"],
                "timestamp": ts, "source_ip": attacker_ip,
                "geo_location": "unknown", "lat": np.nan, "lon": np.nan,
                "resource_accessed": "/api/v1/auth",
                "auth_method": "password", "auth_success": bool(success),
                "session_duration": round(float(rng.uniform(0.05, 0.2)), 2),
                "command_sequence": [], "device_fingerprint": "unknown|unknown|unknown",
                "label": "credential_stuffing",
            })
    return rows


def inject_device_spoofing(profiles, n_incidents=6):
    rows = []
    device_profiles = [p for p in profiles if p["entity_type"] == "edge_device"] or profiles
    for _ in range(n_incidents):
        target = random.choice(device_profiles)
        day = rng.integers(0, SIM_DAYS)
        ts = SIM_START + timedelta(days=int(day), hours=float(rng.uniform(0, 23)))
        spoofed_fp = f"{random.choice(OS_LIST)}|{mac_addr()}|{'TLS1.3'}"
        rows.append({
            "entity_id": target["entity_id"], "entity_type": target["entity_type"],
            "timestamp": ts, "source_ip": f"{rng.integers(1,224)}.{rng.integers(0,256)}.{rng.integers(0,256)}.{rng.integers(1,255)}",
            "geo_location": random.choice(CITIES)[0], "lat": np.nan, "lon": np.nan,
            "resource_accessed": random.choice(target["resource_subset"]),
            "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
            "auth_success": True, "session_duration": round(float(rng.lognormal(2.5, 0.4)), 2),
            "command_sequence": gen_command_sequence(True),
            "device_fingerprint": spoofed_fp,
            "label": "device_spoofing",
        })
    return rows


def inject_lateral_movement(profiles, n_incidents=6):
    rows = []
    for _ in range(n_incidents):
        target = random.choice(profiles)
        unseen = [r for r in RESOURCE_CATALOG if r not in target["resource_subset"]]
        breadth = int(rng.integers(6, 12))
        touched = list(rng.choice(unseen, size=min(breadth, len(unseen)), replace=False))
        day = rng.integers(0, SIM_DAYS)
        start = SIM_START + timedelta(days=int(day), hours=float(rng.uniform(0, 23)))
        for i, res in enumerate(touched):
            ts = start + timedelta(minutes=int(i * rng.uniform(1, 5)))
            rows.append({
                "entity_id": target["entity_id"], "entity_type": target["entity_type"],
                "timestamp": ts, "source_ip": f"{target['home_ip_prefix']}.{rng.integers(1,255)}",
                "geo_location": target["home_city"], "lat": target["home_lat"], "lon": target["home_lon"],
                "resource_accessed": res,
                "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
                "auth_success": True, "session_duration": round(float(rng.lognormal(1.8, 0.4)), 2),
                "command_sequence": gen_command_sequence(True),
                "device_fingerprint": target["device_fingerprint"],
                "label": "lateral_movement",
            })
    return rows


def inject_low_and_slow_exfil(profiles, n_incidents=5):
    """Small off-hours sessions on sensitive-ish resources, spread over days."""
    rows = []
    sensitive = [r for r in RESOURCE_CATALOG if "finance" in r or "hr" in r or "exec" in r or "billing" in r]
    for _ in range(n_incidents):
        target = random.choice(profiles)
        start_day = int(rng.integers(0, SIM_DAYS - 10))
        n_days_active = int(rng.integers(6, 10))
        off_hour = (target["login_hour_mean"] + 12) % 24  # opposite of normal pattern
        for d in range(n_days_active):
            if rng.random() < 0.7:  # not every day — "low and slow"
                ts = SIM_START + timedelta(days=start_day + d, hours=float(off_hour + rng.normal(0, 1)))
                res = random.choice(sensitive) if sensitive else random.choice(RESOURCE_CATALOG)
                rows.append({
                    "entity_id": target["entity_id"], "entity_type": target["entity_type"],
                    "timestamp": ts, "source_ip": f"{target['home_ip_prefix']}.{rng.integers(1,255)}",
                    "geo_location": target["home_city"], "lat": target["home_lat"], "lon": target["home_lon"],
                    "resource_accessed": res,
                    "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
                    "auth_success": True, "session_duration": round(float(rng.lognormal(2.0, 0.3)), 2),
                    "command_sequence": ["read", "export"] if rng.random() < 0.6 else ["read"],
                    "device_fingerprint": target["device_fingerprint"],
                    "label": "exfiltration",
                })
    return rows


def inject_insider_drift(profiles, n_incidents=5):
    """Edge case: legit entity slowly widening its resource footprint,
    during normal hours, no failures — ambiguous, used for FP tuning."""
    rows = []
    for _ in range(n_incidents):
        target = random.choice(profiles)
        extra_pool = [r for r in RESOURCE_CATALOG if r not in target["resource_subset"]]
        n_new = min(3, len(extra_pool))
        new_resources = list(rng.choice(extra_pool, size=n_new, replace=False))
        start_day = int(rng.integers(0, SIM_DAYS - 15))
        for i, res in enumerate(new_resources):
            # each new resource shows up a bit later than the last -> gradual
            day = start_day + i * int(rng.integers(3, 6))
            hour = np.clip(rng.normal(target["login_hour_mean"], target["login_hour_std"]), 0, 23.98)
            ts = SIM_START + timedelta(days=day, hours=float(hour))
            rows.append({
                "entity_id": target["entity_id"], "entity_type": target["entity_type"],
                "timestamp": ts, "source_ip": f"{target['home_ip_prefix']}.{rng.integers(1,255)}",
                "geo_location": target["home_city"], "lat": target["home_lat"], "lon": target["home_lon"],
                "resource_accessed": res,
                "auth_method": rng.choice(target["auth_methods"], p=target["auth_probs"]),
                "auth_success": True, "session_duration": round(float(rng.lognormal(target["session_dur_mu"], target["session_dur_sigma"])), 2),
                "command_sequence": gen_command_sequence(False),
                "device_fingerprint": target["device_fingerprint"],
                "label": "insider_drift",
            })
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(n_entities=250):
    profiles = build_entities(n_entities)

    normal_rows = []
    for p in profiles:
        normal_rows.extend(gen_normal_sessions(p))
    normal_df = pd.DataFrame(normal_rows)
    n_normal = len(normal_df)
    print(f"Normal sessions: {n_normal}")

    # Target overall injected rate ~1.5% of final dataset -> scale incident counts
    attack_rows = []
    attack_rows += inject_brute_force(profiles, n_incidents=max(4, n_normal // 3000))
    attack_rows += inject_impossible_travel(profiles, n_incidents=max(6, n_normal // 2000))
    attack_rows += inject_credential_stuffing(profiles, n_incidents=max(3, n_normal // 6000))
    attack_rows += inject_device_spoofing(profiles, n_incidents=max(5, n_normal // 2500))
    attack_rows += inject_lateral_movement(profiles, n_incidents=max(5, n_normal // 2500))
    attack_rows += inject_low_and_slow_exfil(profiles, n_incidents=max(4, n_normal // 3000))
    attack_rows += inject_insider_drift(profiles, n_incidents=max(4, n_normal // 3000))

    attack_df = pd.DataFrame(attack_rows)
    print(f"Attack rows: {len(attack_df)}  ({len(attack_df)/(n_normal+len(attack_df))*100:.2f}% of total)")
    print(attack_df["label"].value_counts())

    full_df = pd.concat([normal_df, attack_df], ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df["session_id"] = [f"s_{i:07d}" for i in range(len(full_df))]

    labels_df = full_df[["session_id", "label"]].copy()
    features_df = full_df.drop(columns=["label"])

    return features_df, labels_df, profiles


if __name__ == "__main__":
    features_df, labels_df, profiles = generate_dataset(n_entities=250)
    features_df.to_csv("/home/claude/behavioral-anomaly-detection/data/raw/access_logs.csv", index=False)
    labels_df.to_csv("/home/claude/behavioral-anomaly-detection/data/labels/labels.csv", index=False)
    print("\nSaved:")
    print(" - data/raw/access_logs.csv", features_df.shape)
    print(" - data/labels/labels.csv", labels_df.shape)
