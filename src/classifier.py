"""
Phase 5 — Anomaly Classification.

Given a session already flagged as anomalous (Phase 4), predict which of
the 7 injected categories it resembles. Trained only on the anomalous
subset, using the same deviation feature vectors from Phase 4.

Imbalance note: per-class counts range from ~600 (brute_force) down to
~25 (device_spoofing) — too small/uneven for safe SMOTE on the rarest
classes (few real neighbors to interpolate between), so this uses
class_weight='balanced' instead, which reweights the loss rather than
fabricating synthetic minority rows.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pickle

FEATURE_COLS = [
    "geo_velocity_kmh", "time_since_last_hr", "hour_zscore", "resource_novelty",
    "fingerprint_mismatch", "trailing_auth_failures", "transition_novelty",
    "profile_confidence", "auth_success", "session_duration",
]


def main():
    df = pd.read_csv("data/processed/deviation_features.csv")
    df = pd.get_dummies(df, columns=["entity_type"], prefix="etype")
    etype_cols = [c for c in df.columns if c.startswith("etype_")]
    feature_cols = FEATURE_COLS + etype_cols

    anomalies = df[df["label"] != "normal"].copy()
    print(f"Anomalous sessions: {len(anomalies)}")
    print(anomalies["label"].value_counts(), "\n")

    X = anomalies[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(anomalies["label"].values)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # per-sample weights (class_weight='balanced' equivalent for multi:softprob)
    class_counts = np.bincount(y_train)
    weights = {c: len(y_train) / (len(class_counts) * cnt) for c, cnt in enumerate(class_counts)}
    sample_weight = np.array([weights[c] for c in y_train])

    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        objective="multi:softprob", num_class=len(class_names),
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test_s)

    print("=== Classification report (test split) ===")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=3, zero_division=0)
    print(report)

    # 5.3 confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Anomaly-type classification — confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("reports/figures/classification_confusion_matrix.png", dpi=120)
    plt.close()

    # 5.4 per-class precision/recall table
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    per_class = pd.DataFrame({
        "class": class_names, "precision": prec.round(3), "recall": rec.round(3),
        "f1": f1.round(3), "support": support,
    }).sort_values("support", ascending=False)
    print("\n=== Per-class precision/recall ===")
    print(per_class.to_string(index=False))
    per_class.to_csv("reports/figures/classification_per_class_report.csv", index=False)

    with open("data/processed/classifier.pkl", "wb") as f:
        pickle.dump({
            "model": clf, "scaler": scaler, "label_encoder": le,
            "feature_cols": feature_cols, "class_names": list(class_names),
        }, f)
    print("\nSaved: data/processed/classifier.pkl, reports/figures/classification_confusion_matrix.png, "
          "reports/figures/classification_per_class_report.csv")


if __name__ == "__main__":
    main()
