"""
Recommendation engine.

Maps top churn drivers to ranked actions with:
- expected churn-reduction (delta retention pts)
- confidence (based on driver match strength + sample evidence)
- cost
- ROI estimate

Phase 1 implementation: rule-based mapping with calibrated effect sizes.
Phase 2 (production): replace with proper uplift model trained on
historical action-outcome pairs (X-learner / causal forest).
"""
from typing import Optional


# Average mitra LTV (illustrative — protected revenue if retained 6 months)
LTV_DRIVER = 25_000_000   # Rp 25 jt / 6 months
LTV_PICKER = 16_000_000   # Rp 16 jt / 6 months


# Action library — phase-1 rule-based with calibrated effect sizes
ACTION_LIBRARY = [
    {
        "id": "schedule_cap",
        "label": "Cap shift <=8h/day + earnings guarantee 1 minggu",
        "category": "Schedule + Financial",
        "applies_to_drivers": ["avg_working_hours", "shift_completion_rate"],
        "delta_retention_pts": 28,
        "base_confidence": 0.78,
        "cost_rp": 800_000,
        "cost_label": "Rp 800k earnings guarantee",
        "evidence_n": 84,
        "owner": "Hub Manager (executor)",
    },
    {
        "id": "captain_check",
        "label": "Captain emotional check-in daily, 1 minggu",
        "category": "Human / Peer",
        "applies_to_drivers": ["complaints_received", "complaints_filed", "tenure_days", "shift_completion_rate", "notification_response_rate"],
        "delta_retention_pts": 19,
        "base_confidence": 0.71,
        "cost_rp": 250_000,
        "cost_label": "Rp 250k Captain bonus + 30 min/hari",
        "evidence_n": 156,
        "owner": "Captain (executor)",
    },
    {
        "id": "earnings_guarantee",
        "label": "Earnings guarantee 2 minggu (commit >=6h/hari -> minimum Rp 200k/hari)",
        "category": "Financial",
        "applies_to_drivers": ["earnings_trend_pct", "earnings_volatility", "earnings_4w_jt"],
        "delta_retention_pts": 26,
        "base_confidence": 0.74,
        "cost_rp": 1_500_000,
        "cost_label": "Rp 1.5jt income smoothing",
        "evidence_n": 67,
        "owner": "Mitra Management (approver)",
    },
    {
        "id": "hub_reassign",
        "label": "Hub reassignment ke hub lebih dekat (jika tersedia)",
        "category": "Operational",
        "applies_to_drivers": ["distance_to_hub_km", "late_arrival_7d"],
        "delta_retention_pts": 22,
        "base_confidence": 0.62,
        "cost_rp": 0,
        "cost_label": "Admin cost (gratis)",
        "evidence_n": 23,
        "owner": "Mitra Management + Ops",
    },
    {
        "id": "mandatory_off",
        "label": "Mandatory 2 hari off + health check",
        "category": "Wellness",
        "applies_to_drivers": ["avg_working_hours", "no_show_30d", "shift_completion_rate"],
        "delta_retention_pts": 14,
        "base_confidence": 0.64,
        "cost_rp": 400_000,
        "cost_label": "Rp 400k income compensation + clinic",
        "evidence_n": 47,
        "owner": "Mitra Management",
    },
    {
        "id": "complaint_debrief",
        "label": "Customer complaint debrief + coaching session",
        "category": "Performance",
        "applies_to_drivers": ["complaints_received", "shift_completion_rate"],
        "delta_retention_pts": 12,
        "base_confidence": 0.58,
        "cost_rp": 100_000,
        "cost_label": "Rp 100k + 1 jam coach time",
        "evidence_n": 38,
        "owner": "Hub Manager + Captain",
    },
    {
        "id": "captain_assign",
        "label": "Assign Captain mentorship (kalau belum ada)",
        "category": "Human / Peer",
        "applies_to_drivers": ["captain_assigned", "tenure_days"],
        "delta_retention_pts": 23,
        "base_confidence": 0.69,
        "cost_rp": 300_000,
        "cost_label": "Rp 300k Captain stipend / bulan",
        "evidence_n": 102,
        "owner": "Mitra Management",
    },
    {
        "id": "first30_intensive",
        "label": "First-30-day intensive: Captain daily + onboarding refresh",
        "category": "Onboarding",
        "applies_to_drivers": ["tenure_days", "captain_assigned", "notification_response_rate"],
        "delta_retention_pts": 31,
        "base_confidence": 0.72,
        "cost_rp": 500_000,
        "cost_label": "Rp 500k Captain bonus + materi",
        "evidence_n": 91,
        "owner": "Mitra Management + Captain",
    },
    {
        "id": "skill_training",
        "label": "Skill training enrollment (route opt. / customer service)",
        "category": "Career",
        "applies_to_drivers": ["complaints_received", "shift_completion_rate"],
        "delta_retention_pts": 9,
        "base_confidence": 0.51,
        "cost_rp": 250_000,
        "cost_label": "Rp 250k training cost",
        "evidence_n": 29,
        "owner": "Mitra Management",
    },
    {
        "id": "vehicle_subsidy",
        "label": "Vehicle repair / maintenance subsidy (driver only)",
        "category": "Financial",
        "applies_to_drivers": ["late_arrival_7d", "no_show_30d"],
        "delta_retention_pts": 17,
        "base_confidence": 0.55,
        "cost_rp": 600_000,
        "cost_label": "Rp 600k subsidy",
        "evidence_n": 18,
        "owner": "Mitra Management",
        "only_for_role": "Driver",
    },
]


def _ltv_for(role: str) -> int:
    return LTV_DRIVER if role == "Driver" else LTV_PICKER


def recommend_for_mitra(mitra_row: dict, top_drivers: list, risk_prob: float, model_confidence: float, top_k: int = 4) -> list:
    """Generate ranked action recommendations for a mitra."""
    role = mitra_row.get("role", "Picker")
    ltv = _ltv_for(role)

    driver_features = [d["feature"] for d in top_drivers]

    candidates = []
    for action in ACTION_LIBRARY:
        if "only_for_role" in action and action["only_for_role"] != role:
            continue

        matches = [f for f in action["applies_to_drivers"] if f in driver_features]
        if not matches:
            continue

        # Confidence: base x match_strength + small model_conf adjustment
        n_matches = len(matches)
        match_strength = min(1.0, 0.5 + 0.25 * n_matches)
        confidence = action["base_confidence"] * match_strength
        confidence = confidence + 0.10 * (model_confidence - 0.5)
        confidence = float(min(0.95, max(0.25, confidence)))

        # Effect scales with how at-risk the mitra is
        expected_delta = action["delta_retention_pts"] * (0.7 + 0.6 * risk_prob)
        expected_delta = float(min(45, expected_delta))

        ltv_protected = (expected_delta / 100.0) * ltv * confidence
        roi = (ltv_protected - action["cost_rp"]) / max(action["cost_rp"], 1)

        candidates.append({
            "id": action["id"],
            "label": action["label"],
            "category": action["category"],
            "confidence": confidence,
            "expected_delta_pts": expected_delta,
            "cost_rp": action["cost_rp"],
            "cost_label": action["cost_label"],
            "evidence_n": action["evidence_n"],
            "owner": action["owner"],
            "ltv_protected_rp": ltv_protected,
            "roi": roi,
            "matched_drivers": matches,
        })

    # Sort by effectiveness x confidence
    candidates.sort(key=lambda x: x["expected_delta_pts"] * x["confidence"], reverse=True)
    return candidates[:top_k]


def combo_recommendation(actions: list, max_combo: int = 2) -> Optional[dict]:
    """Suggest the best combination of 2 actions if it makes sense."""
    if len(actions) < 2:
        return None
    a, b = actions[0], None
    for act in actions[1:]:
        if act["category"] != a["category"]:
            b = act
            break
    if b is None:
        return None

    combined_delta = a["expected_delta_pts"] + 0.7 * b["expected_delta_pts"]
    combined_delta = min(45, combined_delta)
    combined_cost = a["cost_rp"] + b["cost_rp"]
    combined_conf = (a["confidence"] + b["confidence"]) / 2

    return {
        "actions": [a["id"], b["id"]],
        "labels": [a["label"], b["label"]],
        "expected_delta_pts": combined_delta,
        "cost_rp": combined_cost,
        "confidence": combined_conf,
    }
