# Predict & Care — Streamlit Prototype

Hackathon prototype untuk Astro Mitra Happiness theme.
Owner of dashboard: **Tim Mitra Management**.

## What this is

A working demo of the **Mitra Churn Prediction & Recommendation Engine** described in the pitch deck.
Loads 250 synthetic mitra, trains a logistic regression model, and serves an
interactive dashboard for Mitra Management to:

1. See risk distribution across hubs/roles
2. Drill down into individual mitra at risk
3. Get **ranked action recommendations** with confidence + ROI per action
4. Track cohort survival curves
5. Inspect model performance & feature importance

## Quick start (3 steps)

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. (Optional) regenerate synthetic data — already pre-generated
python data_generator.py

# 3. Run the dashboard
streamlit run app.py
```

App opens at `http://localhost:8501` in your browser.

## File structure

```
predict_care_prototype/
├── app.py                # Streamlit dashboard (5 tabs)
├── data_generator.py     # Synthetic mitra dataset generator
├── model.py              # Logistic regression training + scoring
├── recommendations.py    # Action library + recommendation engine
├── data/
│   └── mitra.csv         # 250 mitra, pre-generated
├── requirements.txt
└── README.md
```

## Demo script for hackathon (7-10 min)

**Min 0-2: Pitch the problem** (use slide deck)

**Min 2-4: Show overview tab**
- "Pagi ini Mitra Management buka dashboard."
- Point to KPIs: "26 mitra Red, 89 Yellow."
- Filter by hub Tebet → "Risk concentration di sini lebih tinggi."

**Min 4-6: Drill-down to mitra detail tab**
- Pick highest-risk mitra (top of list).
- "Ini Budi, driver 8 bulan, Hub Tebet. 82% churn probability."
- Walk through Top 3 Drivers: "Working hours 11.2 jam — burnout. Earnings turun 42%. Late arrival 5x."
- Show recommended actions: "Bukan cuma flag — sistem kasih ranked actions."
- "Top action: Cap shift + earnings guarantee. Confidence 78%, expected +28 pts retention, cost Rp 800k, ROI 5x."
- Show combo recommendation: "Mitra Management approve combo, action otomatis di-trigger ke Hub Manager + Captain."

**Min 6-7: Show cohort survival tab**
- "Cohort W16 dropping below baseline at Week 4 — sistem auto-flag investigasi."
- "Bukan reactive ke individual, tapi pattern-level juga ter-monitor."

**Min 7-8: Show model performance tab**
- "Test AUC 0.75+. Top global drivers explainable."
- "Bukan blackbox, Mitra Management bisa override kalau ada konteks yang model tidak tahu."

**Min 8-10: Roadmap + Ask + close** (use slide deck)

## Tech notes

- **Synthetic data with planted patterns**: data_generator.py generates 250 mitra
  dengan pola realistic — beberapa high-hours-low-earnings (burnout case),
  beberapa first-30-day cold-start, beberapa high-distance churners.
  Risk model belajar pola ini saat training.

- **Model**: Logistic Regression dengan StandardScaler. Phase 2 (production)
  ganti ke XGBoost + SHAP. Per-mitra "top drivers" dihitung dari
  `coefficient × standardized_value` — proxy untuk SHAP values.

- **Recommendation engine**: Phase 1 pakai rule-based mapping
  (driver → action library) dengan calibrated effect sizes.
  Phase 2 (production): replace dengan uplift model
  (X-learner / causal forest) yang trained on historical action-outcome data.

- **Confidence**: kombinasi model uncertainty (entropy of prediction) +
  driver-action match strength + base confidence per action (dari evidence n).

## Deploy ke Streamlit Cloud (gratis)

```bash
# 1. Push repo ini ke GitHub
git init && git add . && git commit -m "init"
git remote add origin https://github.com/<you>/predict-care-prototype
git push -u origin main

# 2. Buka https://share.streamlit.io
# 3. Connect GitHub, pick repo, deploy
# 4. Live URL siap untuk presentasi (bisa share QR code di slide)
```

## Q&A defense talking points

**"Data ini real apa fake?"**
Synthetic — dibuat untuk demo. Pattern-nya calibrated ke literature
quick commerce churn (Foodpanda, DoorDash, Zepto), bukan random.
Production butuh 6-12 bulan historical labeled data dari Astro.

**"AUC 0.75 — apakah cukup?"**
Untuk targeting (siapa yang dapat Captain attention prioritas), ya.
Untuk firing decision, tentu tidak — tapi sistem ini bukan untuk
firing, untuk targeted retention support. Industry benchmark 0.70-0.85.

**"Confidence 78% — angkanya dari mana?"**
Kombinasi: (a) model entropy untuk prediction itu sendiri,
(b) match strength antara mitra's top drivers vs action's targeted drivers,
(c) base confidence dari historical evidence n. Phase 1 ada ilustrasi,
Phase 2 angka real dari A/B test data.

**"Apakah perlu replace mitra app?"**
Tidak. Dashboard ini standalone web app untuk Mitra Management.
Captain dapat notification via existing mitra app (tambah 1 tab "Mentee").
Hub Manager dapat task assignment via existing ops tool.
Zero UX change untuk 95% mitra (non-Captain non-mentee).

## Credits

Built for Astro Hackathon 2026 · Theme: Mitra Happiness · Tim Mitra Management
