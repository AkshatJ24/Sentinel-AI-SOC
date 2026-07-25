"""
Phase 4.3-4.6 — Detection models + evaluation at a realistic alert budget.

Two candidates:
  A) Isolation Forest — unsupervised, doesn't need labels (matches the
     reality that true intrusions are rare/unlabeled in production).
  B) XGBoost — supervised, SMOTE-oversampled on the training split only
     (never on test — avoids synthetic leakage into evaluation).

Both scored, both evaluated at the SOC-realistic budget: top 1% of
sessions by risk score. Precision/recall at that budget is what actually
matters to an analyst (fixed daily alert capacity), not accuracy on the
full imbalanced set.

Split: stratified random 80/20 (documented assumption — a temporal split
was considered, but injected incidents are scattered evenly across the
full 45-day window, so stratified random gives a cleaner apples-to-apples
comparison for this MVP timeline).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

FEATURE_COLS = [
    "geo_velocity_kmh", "time_since_last_hr", "hour_zscore", "resource_novelty",
    "fingerprint_mismatch", "trailing_auth_failures", "transition_novelty",
    "profile_confidence", "auth_success", "session_duration",
]
ALERT_BUDGET = 0.01  # top 1% of sessions, per hackathon's stated evaluation criterion


def load_data():
    df = pd.read_csv("data/processed/deviation_features.csv")
    df["y"] = (df["label"] != "normal").astype(int)
    # entity_type as one-hot — cheap extra signal (e.g. edge_device baseline differs from user)
    df = pd.get_dummies(df, columns=["entity_type"], prefix="etype")
    etype_cols = [c for c in df.columns if c.startswith("etype_")]
    return df, FEATURE_COLS + etype_cols


def eval_at_budget(y_true, scores, budget=ALERT_BUDGET, label=""):
    n_alerts = max(1, int(len(scores) * budget))
    top_idx = np.argsort(scores)[::-1][:n_alerts]
    y_pred = np.zeros(len(scores), dtype=int)
    y_pred[top_idx] = 1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_true_caught = y_true[top_idx].sum()
    print(f"{label:22s} | alerts={n_alerts:4d} | precision={precision:.3f} | "
          f"recall={recall:.3f} | f1={f1:.3f} | anomalies caught={n_true_caught}/{y_true.sum()}")
    return {"precision": precision, "recall": recall, "f1": f1, "n_alerts": n_alerts}


def main():
    df, feature_cols = load_data()
    X = df[feature_cols].fillna(0).values
    y = df["y"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} anomalies) | "
          f"Test: {len(X_test)} ({y_test.sum()} anomalies)\n")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # ---------------- Model A: Isolation Forest (unsupervised) ----------------
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)  # no y used
    iso_scores = -iso.decision_function(X_test_s)  # higher = riskier

    # ---------------- Model B: XGBoost + SMOTE (train split only) -------------
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_s, y_train)
    print(f"After SMOTE: {len(X_train_sm)} rows ({y_train_sm.sum()} anomalies, balanced)\n")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train_sm, y_train_sm)
    xgb_scores = xgb.predict_proba(X_test_s)[:, 1]

    # ---------------- Evaluation at top-1% alert budget ----------------
    print("=== Evaluation @ top-1% alert budget ===")
    iso_result = eval_at_budget(y_test, iso_scores, label="IsolationForest")
    xgb_result = eval_at_budget(y_test, xgb_scores, label="XGBoost+SMOTE")

    # sweep a couple more budgets for context
    print("\n=== Sensitivity across budgets ===")
    for b in [0.005, 0.01, 0.02, 0.05]:
        print(f"-- budget={b} --")
        eval_at_budget(y_test, iso_scores, budget=b, label="  IsolationForest")
        eval_at_budget(y_test, xgb_scores, budget=b, label="  XGBoost+SMOTE")

    # ---------------- Pick primary ----------------
    primary = "xgboost" if xgb_result["f1"] >= iso_result["f1"] else "isolation_forest"
    print(f"\nPrimary model selected: {primary} "
          f"(top-1% F1: IF={iso_result['f1']:.3f} vs XGB={xgb_result['f1']:.3f})")

    # save artifacts for Phase 5/6/7 reuse
    with open("data/processed/detection_models.pkl", "wb") as f:
        pickle.dump({
            "scaler": scaler, "isolation_forest": iso, "xgboost": xgb,
            "feature_cols": feature_cols, "primary": primary,
            "test_idx": idx_test, "iso_scores": iso_scores, "xgb_scores": xgb_scores,
            "y_test": y_test,
        }, f)
    print("\nSaved: data/processed/detection_models.pkl")


if __name__ == "__main__":
    main()
