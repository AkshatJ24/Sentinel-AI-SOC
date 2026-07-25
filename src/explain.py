"""
Phase 6 — Explainability Layer.

SHAP TreeExplainer runs against the Phase 4 XGBoost detector (the primary
model) to get per-alert feature attributions. Top-3 contributing features
are mapped through a natural-language template — that's what a SOC
analyst actually reads, not raw SHAP values.
"""

import pandas as pd
import numpy as np
import pickle
import shap

FEATURE_LABELS = {
    "geo_velocity_kmh": lambda v: f"implausible travel speed ({v:,.0f} km/h)",
    "time_since_last_hr": lambda v: f"unusual gap since last session ({v:.1f}h)",
    "hour_zscore": lambda v: f"login time {v:.1f}σ from usual pattern",
    "resource_novelty": lambda v: "accessed a resource never touched before",
    "fingerprint_mismatch": lambda v: "device fingerprint mismatch vs known history",
    "trailing_auth_failures": lambda v: f"{int(v)} failed logins from this source in the last 5 min",
    "transition_novelty": lambda v: "unexpected action sequence for this entity",
    "profile_confidence": lambda v: "limited history for this entity (cold-start)" if v < 0.3 else None,
    "auth_success": lambda v: "authentication failed" if v == 0 else None,
    "session_duration": lambda v: f"unusually short session ({v:.2f} min)" if v < 1 else None,
}


def describe_feature(fname, value):
    if fname.startswith("etype_"):
        return None
    fn = FEATURE_LABELS.get(fname)
    if fn is None:
        return f"{fname}={value:.2f}"
    return fn(value)


def build_explanation(shap_row, feature_names, X_row, top_n=3):
    order = np.argsort(-np.abs(shap_row))
    phrases = []
    for i in order:
        if shap_row[i] <= 0:  # only features PUSHING toward anomaly, not away
            continue
        desc = describe_feature(feature_names[i], X_row[i])
        if desc:
            phrases.append(desc)
        if len(phrases) == top_n:
            break
    if not phrases:
        return "flagged by combined weak signals across several features"
    return "flagged due to " + " + ".join(phrases)


def main():
    with open("data/processed/detection_models.pkl", "rb") as f:
        det = pickle.load(f)
    df = pd.read_csv("data/processed/deviation_features.csv")
    df = pd.get_dummies(df, columns=["entity_type"], prefix="etype")

    feature_cols = det["feature_cols"]
    xgb_model = det["xgboost"]
    scaler = det["scaler"]
    test_idx = det["test_idx"]
    xgb_scores = det["xgb_scores"]

    test_df = df.loc[test_idx].reset_index(drop=True)
    X_test = test_df[feature_cols].fillna(0).values
    X_test_s = scaler.transform(X_test)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_s)
    if isinstance(shap_values, list):  # binary classifier sometimes returns [class0, class1]
        shap_values = shap_values[1]

    # top-1% alerts (same budget as Phase 4)
    n_alerts = max(1, int(len(xgb_scores) * 0.01))
    top_idx = np.argsort(xgb_scores)[::-1][:n_alerts]

    alerts = []
    for i in top_idx:
        explanation = build_explanation(shap_values[i], feature_cols, X_test[i])
        alerts.append({
            "session_id": test_df.loc[i, "session_id"],
            "entity_id": test_df.loc[i, "entity_id"],
            "true_label": test_df.loc[i, "label"],
            "risk_score": round(float(xgb_scores[i]), 4),
            "explanation": explanation,
        })
    alerts_df = pd.DataFrame(alerts).sort_values("risk_score", ascending=False)
    alerts_df.to_csv("data/processed/explained_alerts.csv", index=False)

    print(f"Generated explanations for top {n_alerts} alerts.\n")
    print("=== Sample alerts ===")
    for _, row in alerts_df.head(8).iterrows():
        print(f"[{row.risk_score:.3f}] {row.entity_id} (true: {row.true_label}) -> {row.explanation}")

    # 6.4 — sanity check: do brute_force alerts actually mention auth failures?
    print("\n=== Sanity check: brute_force alerts should mention auth failures ===")
    bf_alerts = alerts_df[alerts_df.true_label == "brute_force"]
    if len(bf_alerts):
        mentions = bf_alerts["explanation"].str.contains("failed logins").mean()
        print(f"{len(bf_alerts)} brute_force alerts in top-1%, "
              f"{mentions*100:.0f}% mention failed logins in explanation")
        for _, row in bf_alerts.head(3).iterrows():
            print(f"  -> {row.explanation}")
    else:
        print("No brute_force sessions landed in top-1% test alerts this split (check lower budgets).")

    print("\n=== Sanity check: impossible_travel alerts should mention travel speed ===")
    it_alerts = alerts_df[alerts_df.true_label == "impossible_travel"]
    if len(it_alerts):
        mentions = it_alerts["explanation"].str.contains("travel speed").mean()
        print(f"{len(it_alerts)} impossible_travel alerts, {mentions*100:.0f}% mention travel speed")
        for _, row in it_alerts.head(3).iterrows():
            print(f"  -> {row.explanation}")

    print("\nSaved: data/processed/explained_alerts.csv")


if __name__ == "__main__":
    main()
