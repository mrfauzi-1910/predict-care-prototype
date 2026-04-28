"""
Churn prediction model + feature explainability (SHAP-like).

Trains a simple Logistic Regression on the synthetic data.
Returns per-mitra risk score, tier, and top-3 driver features
with their relative contribution.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix


# Features used by the model
NUMERIC_FEATURES = [
    "tenure_days",
    "distance_to_hub_km",
    "avg_working_hours",
    "shift_completion_rate",
    "late_arrival_7d",
    "no_show_30d",
    "earnings_4w_jt",
    "earnings_trend_pct",
    "earnings_volatility",
    "complaints_received",
    "complaints_filed",
    "app_login_freq_7d",
    "notification_response_rate",
    "captain_assigned",
    "prior_gig_experience",
]
# Categorical features → one-hot
CATEGORICAL_FEATURES = ["role", "source_channel"]

# Friendly names for display
FEATURE_DISPLAY = {
    "tenure_days": "Tenure",
    "distance_to_hub_km": "Distance to Hub",
    "avg_working_hours": "Working hours",
    "shift_completion_rate": "Shift completion",
    "late_arrival_7d": "Late arrivals (7d)",
    "no_show_30d": "No-shows (30d)",
    "earnings_4w_jt": "Earnings (4w)",
    "earnings_trend_pct": "Earnings trend",
    "earnings_volatility": "Earnings volatility",
    "complaints_received": "Complaints received",
    "complaints_filed": "Complaints filed",
    "app_login_freq_7d": "App login freq",
    "notification_response_rate": "Notif response rate",
    "captain_assigned": "Captain assigned",
    "prior_gig_experience": "Prior gig experience",
    "role_Picker": "Role: Picker",
    "source_channel_walk-in": "Source: walk-in",
    "source_channel_ads": "Source: ads",
    "source_channel_agency": "Source: agency",
}


def prepare_features(df: pd.DataFrame):
    """Return X (features) and feature_names."""
    X_num = df[NUMERIC_FEATURES].copy()
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True).astype(int)
    X = pd.concat([X_num, X_cat], axis=1)
    return X, X.columns.tolist()


def train_model(df: pd.DataFrame, random_state: int = 42):
    """Train logistic regression on synthetic data.
    Returns trained model, scaler, feature names, and metrics."""
    X, feature_names = prepare_features(df)
    y = df["churned"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, random_state=random_state
    )
    model.fit(X_train, y_train)

    probs_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    preds_test = (probs_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds_test)

    metrics = {
        "auc": auc,
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": _coef_importance(model, feature_names),
    }

    return model, scaler, feature_names, metrics


def _coef_importance(model, feature_names):
    """Global feature importance from logistic regression coefficients."""
    coefs = model.coef_[0]
    fi = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    fi["display_name"] = fi["feature"].map(lambda f: FEATURE_DISPLAY.get(f, f))
    return fi


def score_mitra(df: pd.DataFrame, model, scaler, feature_names):
    """Score every mitra. Returns df with added columns:
    - churn_prob_pred (0..1)
    - risk_tier (Green/Yellow/Red)
    - top_drivers (list of dicts: {feature, contribution_z, value})
    """
    X, _ = prepare_features(df)
    X = X[feature_names]  # ensure column order matches training
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]

    out = df.copy()
    out["churn_prob_pred"] = probs
    out["risk_tier"] = pd.cut(
        probs,
        bins=[-0.01, 0.40, 0.65, 1.01],
        labels=["Green", "Yellow", "Red"],
    )

    # Per-mitra top drivers — coef * standardized value (logistic contribution)
    coefs = model.coef_[0]
    contribs = Xs * coefs  # shape (n, n_features)

    top_drivers_list = []
    for i in range(len(out)):
        row_contribs = contribs[i]
        # Top 3 by absolute positive contribution to churn (push toward 1)
        idx_sorted = np.argsort(-row_contribs)  # descending
        top3 = []
        for idx in idx_sorted[:3]:
            if row_contribs[idx] <= 0:
                break
            fname = feature_names[idx]
            raw_val = X.iloc[i][fname]
            top3.append({
                "feature": fname,
                "display_name": FEATURE_DISPLAY.get(fname, fname),
                "contribution": float(row_contribs[idx]),
                "value": float(raw_val) if pd.notna(raw_val) else None,
            })
        top_drivers_list.append(top3)

    out["top_drivers"] = top_drivers_list
    return out


def confidence_score(probs: np.ndarray) -> np.ndarray:
    """How confident is the model about each prediction?
    Confidence is high when prob is near 0 or near 1 (low entropy)."""
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    # entropy max = 1 when p = 0.5; min = 0 when p ∈ {0, 1}
    return 1 - entropy


if __name__ == "__main__":
    df = pd.read_csv("data/mitra.csv")
    model, scaler, feature_names, metrics = train_model(df)
    print(f"Trained on {metrics['n_train']} mitra, tested on {metrics['n_test']}")
    print(f"Test AUC: {metrics['auc']:.3f}")
    print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
    print("\nTop 8 features by importance:")
    print(metrics["feature_importance"].head(8)[["display_name", "coef"]].to_string(index=False))

    scored = score_mitra(df, model, scaler, feature_names)
    print(f"\nRisk tier distribution:")
    print(scored["risk_tier"].value_counts())
