"""Phase 2 — EDA on synthetic data. Saves figures to reports/figures/, prints checks."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "reports/figures"

feat = pd.read_csv("data/raw/access_logs.csv")
lab = pd.read_csv("data/labels/labels.csv")
df = feat.merge(lab, on="session_id")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60

# ---------------------------------------------------------------------
# 2.1 — class imbalance
# ---------------------------------------------------------------------
print("=" * 60)
print("2.1 CLASS IMBALANCE")
print("=" * 60)
counts = df["label"].value_counts()
pct = (counts / len(df) * 100).round(3)
print(pd.DataFrame({"count": counts, "pct": pct}))
print(f"\nTotal anomalous: {(df.label != 'normal').sum()} / {len(df)} = "
      f"{(df.label != 'normal').mean()*100:.2f}%")

# ---------------------------------------------------------------------
# 2.2 — per-entity normal-session profile sanity check
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("2.2 PER-ENTITY PROFILE SANITY CHECK (3 sample entities)")
print("=" * 60)
normal_df = df[df.label == "normal"]
sample_entities = normal_df["entity_id"].drop_duplicates().sample(3, random_state=1)

fig, axes = plt.subplots(3, 2, figsize=(11, 10))
for i, eid in enumerate(sample_entities):
    sub = normal_df[normal_df.entity_id == eid]
    axes[i, 0].hist(sub["hour"], bins=24, color="#3b6ea5")
    axes[i, 0].set_title(f"{eid} — login hour distribution (n={len(sub)})")
    axes[i, 0].set_xlabel("hour of day")

    res_counts = sub["resource_accessed"].value_counts()
    axes[i, 1].barh(res_counts.index[:8][::-1], res_counts.values[:8][::-1], color="#5a9367")
    axes[i, 1].set_title(f"{eid} — top resources touched")
    print(f"{eid}: {len(sub)} normal sessions, "
          f"login_hour mean={sub['hour'].mean():.1f} std={sub['hour'].std():.1f}, "
          f"{sub['resource_accessed'].nunique()} unique resources, "
          f"{sub['device_fingerprint'].nunique()} device fingerprint(s)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entity_profile_sanity.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 2.3 — attack type visual separability vs baseline
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("2.3 ATTACK TYPE SEPARABILITY VS BASELINE")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# auth_success rate — brute force / cred stuffing
ax = axes[0, 0]
auth_rate = df.groupby("label")["auth_success"].mean().sort_values()
ax.barh(auth_rate.index, auth_rate.values, color="#c0504d")
ax.set_title("auth_success rate by label\n(brute_force / credential_stuffing near 0)")
ax.set_xlim(0, 1)

# session_duration distribution — normal vs lateral_movement vs exfiltration
ax = axes[0, 1]
for lbl, color in [("normal", "gray"), ("lateral_movement", "#c0504d"),
                    ("exfiltration", "#e8a33d"), ("insider_drift", "#3b6ea5")]:
    vals = df[df.label == lbl]["session_duration"]
    ax.hist(np.log1p(vals), bins=30, alpha=0.5, label=lbl, density=True, color=color)
ax.set_title("log(session_duration) — normal vs 3 attack types")
ax.set_xlabel("log(1 + minutes)")
ax.legend(fontsize=8)

# off-hours rate
ax = axes[1, 0]
df["off_hours"] = df["hour"].apply(lambda h: h < 6 or h >= 22)
off_rate = df.groupby("label")["off_hours"].mean().sort_values()
ax.barh(off_rate.index, off_rate.values, color="#e8a33d")
ax.set_title("off-hours session rate by label\n(exfiltration should lead)")

# rows-per-source_ip for brute force / credential stuffing (burst signature)
ax = axes[1, 1]
burst = df[df.label.isin(["brute_force", "credential_stuffing", "normal"])]
burst_counts = burst.groupby(["label", "source_ip"]).size().reset_index(name="n")
sns.boxplot(data=burst_counts, x="label", y="n", ax=ax, hue="label",
            palette=["#c0504d", "#e8a33d", "#4f81bd"], legend=False)
ax.set_yscale("log")
ax.set_title("rows per source_ip (log scale)\nburst signature vs normal")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attack_separability.png", dpi=120)
plt.close()

separability_summary = df.groupby("label").agg(
    auth_success_rate=("auth_success", "mean"),
    median_duration_min=("session_duration", "median"),
    off_hours_rate=("off_hours", "mean"),
    n=("label", "size"),
).round(3)
print(separability_summary)

# ---------------------------------------------------------------------
# 2.4 — schema completeness check
# ---------------------------------------------------------------------
print("\n" + "=" * 60)
print("2.4 SCHEMA COMPLETENESS CHECK vs problem-statement schema")
print("=" * 60)
required = ["entity_id", "entity_type", "timestamp", "source_ip", "geo_location",
            "resource_accessed", "auth_method", "session_duration",
            "command_sequence", "device_fingerprint"]
present = list(feat.columns)
missing = [c for c in required if c not in present]
extra = [c for c in present if c not in required]
print("Required fields present:", [c for c in required if c in present])
print("Missing:", missing if missing else "None")
print("Extra fields (documented in data_dictionary.md):", extra)

null_report = feat.isnull().mean().round(3)
print("\nNull rate per column (lat/lon nulls expected for spoofed/brute-force IPs):")
print(null_report[null_report > 0])

print("\nSaved figures: reports/figures/entity_profile_sanity.png, "
      "reports/figures/attack_separability.png")
