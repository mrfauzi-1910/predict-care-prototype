"""
Synthetic mitra dataset generator.
Generates realistic mitra data with planted churn patterns for demo.

Run: python data_generator.py
Output: data/mitra.csv
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 250  # total mitra

HUBS = ["Tebet", "Kemang", "PIK", "Bintaro", "Senayan"]
ROLES = ["Driver", "Picker"]
SOURCES = ["referral", "walk-in", "ads", "agency"]

# Indonesian first/last names — realistic but synthetic
FIRST_NAMES = [
    "Budi", "Rina", "Andi", "Dewi", "Sari", "Joko", "Ayu", "Reza",
    "Putri", "Bayu", "Maya", "Doni", "Indah", "Fajar", "Lila",
    "Eko", "Nadia", "Riko", "Citra", "Hadi", "Wati", "Yusuf",
    "Mira", "Bagas", "Sinta", "Adi", "Lestari", "Iwan", "Tika",
    "Galih", "Vina", "Hari", "Ratna", "Dimas", "Rosa", "Anto",
    "Wulan", "Beni", "Salsa", "Rama"
]
LAST_INITIALS = ["S.", "P.", "W.", "L.", "R.", "M.", "A.", "K.", "T.", "H."]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_dataset():
    rows = []
    for i in range(N):
        mid = f"M-{1000 + i:04d}"
        name = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_INITIALS)}"
        role = np.random.choice(ROLES, p=[0.30, 0.70])  # picker > driver
        hub = np.random.choice(HUBS)

        # Tenure: skewed — many new joiners, some tenured
        tenure_days = int(np.random.choice(
            [np.random.randint(1, 30),       # new (40%)
             np.random.randint(30, 180),     # mid (35%)
             np.random.randint(180, 720)],   # tenured (25%)
            p=[0.40, 0.35, 0.25]
        ))

        source = np.random.choice(SOURCES, p=[0.30, 0.35, 0.25, 0.10])
        distance = float(np.clip(np.random.exponential(4) + 1, 0.5, 25))
        prior_gig = int(np.random.random() < 0.35)

        # Behavioral signals (correlated with risk drivers)
        # — base values, then apply pattern injection later
        base_hours = np.random.normal(8.5, 1.5)
        avg_working_hours = float(np.clip(base_hours, 4, 14))
        shift_completion_rate = float(np.clip(np.random.beta(8, 1.5), 0.5, 1.0))
        late_arrival_7d = int(np.random.poisson(1.0))
        no_show_30d = int(np.random.poisson(0.6))

        # Earnings (Rp juta last 4w)
        if role == "Driver":
            base_earnings = np.random.normal(4.5, 1.2)
        else:
            base_earnings = np.random.normal(3.2, 0.9)
        earnings_4w = float(np.clip(base_earnings, 1.0, 8.5))
        # trend pct (vs prev 4w) — many stable, some declining
        earnings_trend_pct = float(np.clip(np.random.normal(-2, 18), -60, 30))
        earnings_volatility = float(np.clip(np.random.beta(2, 6), 0.05, 0.6))

        # Complaints
        complaints_received = int(np.random.poisson(0.4))
        complaints_filed = int(np.random.poisson(0.2))

        # Engagement
        app_login_freq_7d = int(np.clip(np.random.poisson(15), 1, 35))
        notification_response_rate = float(np.clip(np.random.beta(6, 2), 0.2, 1.0))

        # Onboarding (for new joiners only)
        onboarding_score = float(np.clip(np.random.beta(7, 3), 0.4, 1.0)) if tenure_days < 30 else None

        # Captain assignment — more likely for new + at-risk
        captain_assigned = int(
            (tenure_days < 30 and np.random.random() < 0.55) or
            (np.random.random() < 0.20)
        )

        # Vehicle (driver only)
        vehicle_type = np.random.choice(["Motor", "Motor", "Motor", "Mobil"]) if role == "Driver" else None

        # ============ Plant churn patterns ============
        # Higher risk if: high hours, earnings drop, complaints, new + high distance, etc.
        risk_signal = 0.0
        risk_signal += 0.40 * max(0, avg_working_hours - 9.5) / 2.0      # burnout
        risk_signal += 0.55 * (earnings_trend_pct < -25)                  # earnings drop
        risk_signal += 0.35 * (late_arrival_7d >= 3)
        risk_signal += 0.45 * (no_show_30d >= 2)
        risk_signal += 0.50 * (complaints_received >= 2)
        risk_signal += 0.60 * (tenure_days < 14 and source == "walk-in")  # cold-start risk
        risk_signal += 0.35 * (distance > 8.0)
        risk_signal += 0.30 * (notification_response_rate < 0.4)
        risk_signal += 0.40 * (shift_completion_rate < 0.7)
        risk_signal -= 0.55 * captain_assigned                            # protective
        risk_signal -= 0.25 * (tenure_days > 365)                         # tenured = sticky
        risk_signal -= 0.30 * (source == "referral")                      # referral retention
        risk_signal += np.random.normal(0, 0.25)                          # noise

        churn_prob_true = float(sigmoid(risk_signal))
        churned = int(np.random.random() < churn_prob_true)

        # Inject some severe-case patterns for demo storytelling
        # - 3-5 mitra dengan working_hours very high + earnings drop (Budi-style case)
        # - 5-7 mitra new joiner cold-start (Rina-style)
        # We'll just let the natural distribution produce these.

        rows.append({
            "mitra_id": mid,
            "name": name,
            "role": role,
            "hub": hub,
            "tenure_days": tenure_days,
            "source_channel": source,
            "distance_to_hub_km": round(distance, 1),
            "prior_gig_experience": prior_gig,
            "vehicle_type": vehicle_type,
            "avg_working_hours": round(avg_working_hours, 1),
            "shift_completion_rate": round(shift_completion_rate, 3),
            "late_arrival_7d": late_arrival_7d,
            "no_show_30d": no_show_30d,
            "earnings_4w_jt": round(earnings_4w, 2),
            "earnings_trend_pct": round(earnings_trend_pct, 1),
            "earnings_volatility": round(earnings_volatility, 3),
            "complaints_received": complaints_received,
            "complaints_filed": complaints_filed,
            "app_login_freq_7d": app_login_freq_7d,
            "notification_response_rate": round(notification_response_rate, 3),
            "onboarding_score": round(onboarding_score, 3) if onboarding_score is not None else np.nan,
            "captain_assigned": captain_assigned,
            "churn_prob_true": round(churn_prob_true, 3),
            "churned": churned,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    df = make_dataset()
    out = os.path.join(out_dir, "mitra.csv")
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} mitra → {out}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"By role:\n{df.groupby('role')['churned'].mean()}")
    print(f"By tenure segment:")
    print(f"  New (<30d):       {df[df.tenure_days < 30]['churned'].mean():.1%}")
    print(f"  Mid (30-180d):    {df[(df.tenure_days >= 30) & (df.tenure_days < 180)]['churned'].mean():.1%}")
    print(f"  Tenured (180d+):  {df[df.tenure_days >= 180]['churned'].mean():.1%}")
