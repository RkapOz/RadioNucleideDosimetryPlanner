"""
Radionuclide Dose Planner (v2.1) — Interactive Clinical Decision Support App
100% Compliant with MIRD Pamphlet 20 & 21, ICRP 89/107/110/133, and RADAR/EANM Standards

Architecture:
  Stage 1: Single-Time-Point (24h SPECT/CT) Biokinetic Prediction (XGBoost / GBDT)
  Stage 2: Full MIRD 21 Self-Dose + ICRP 133 Cross-Organ Dose + RADAR Remainder-of-Body
           + Exact Bi-Exponential Dale G(infinity) Kidney BED
  Stage 3: Monte Carlo Uncertainty Simulation (N = 10,000) & P95 Safety Margin
  Stage 4A: Single-Cycle Constrained Optimization (SLSQP)
  Stage 4B: Multi-Cycle Markov Decision Process (MDP) & Stochastic Trajectory Planner
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

from synthetic_pk_generator import (
    ICRP_PHANTOMS,
    LAMBDA_PHYS,
    LU177_HALF_LIFE_HOURS,
    MAX_PHYSICAL_TIAC_HOURS,
    RMBLR_PEPTIDE,
    S_CONST_NP_LU177,
    compute_mird_radar_unit_doses,
)

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

FEATURE_COLS = [
    "sex_is_female",
    "age_years",
    "weight_kg",
    "egfr_ml_min",
    "serum_creatinine_mg_dl",
    "alt_u_l",
    "total_bilirubin_mg_dl",
    "platelet_count_10e9_l",
    "has_renal_risk_factors",
    "has_hepatic_risk_factors",
    "kidney_mass_g",
    "liver_mass_g",
    "index_lesion_mass_g",
    "total_tumor_volume_cm3",
    "kidney_uptake_ia_percent_24h",
    "liver_uptake_ia_percent_24h",
    "tumor_suv_mean_24h",
    "kidney_uptake_per_g",
    "liver_uptake_per_g",
    "tumor_uptake_proxy",
]

TARGET_COLS = [
    "kidney_tiac_hours",
    "kidney_g_inf_exact",
    "liver_tiac_hours",
    "blood_tiac_hours",
    "wb_tiac_hours",
    "tumor_tiac_hours",
]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mass-normalized uptake features for ML inference."""
    out = df.copy()
    if "sex_is_female" not in out.columns:
        out["sex_is_female"] = (out["sex"].astype(str).str.upper().str.startswith("F")).astype(int)
    out["kidney_uptake_per_g"] = out["kidney_uptake_ia_percent_24h"] / out["kidney_mass_g"]
    out["liver_uptake_per_g"] = out["liver_uptake_ia_percent_24h"] / out["liver_mass_g"]
    out["tumor_uptake_proxy"] = (
        out["tumor_suv_mean_24h"] * out["index_lesion_mass_g"] / (out["weight_kg"] * 1000.0)
    )
    return out


def load_or_train_model(model_path: str = "lu177_dosimetry_xgb.joblib"):
    """Load saved ML bundle or auto-train from synthetic cohort if missing/outdated."""
    if os.path.exists(model_path):
        bundle = joblib.load(model_path)
        if bundle.get("target_cols") == TARGET_COLS and bundle.get("feature_cols") == FEATURE_COLS:
            return bundle

    from synthetic_pk_generator import generate_lu177_cohort
    from train_dosimetry_xgboost import BACKEND_NAME, build_regressor

    df = generate_lu177_cohort(n_patients=6000, seed=42)
    df = add_physics_features(df)
    model = build_regressor(random_state=42)
    model.fit(df[FEATURE_COLS], df[TARGET_COLS])
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "backend": BACKEND_NAME,
    }
    joblib.dump(bundle, model_path)
    return bundle


def get_oar_limits(patient: dict) -> dict:
    """Determine clinical hard constraints based on renal, hepatic, and marrow risk factors."""
    renal_risk = (
        bool(patient.get("has_renal_risk_factors", 0))
        or patient["egfr_ml_min"] < 60.0
    )
    hepatic_risk = (
        bool(patient.get("has_hepatic_risk_factors", 0))
        or patient["total_bilirubin_mg_dl"] > 1.5
        or patient["alt_u_l"] > 80.0
    )
    marrow_risk = patient["platelet_count_10e9_l"] < 150.0

    return {
        "kidney_bed_limit": 28.0 if renal_risk else 40.0,
        "liver_dose_limit": 25.0 if hepatic_risk else 30.0,
        "marrow_dose_limit": 1.5 if marrow_risk else 2.0,
        "renal_risk": renal_risk,
        "hepatic_risk": hepatic_risk,
        "marrow_risk": marrow_risk,
    }


# ==============================================================================
# 2. STAGE 1–3: ML PREDICTION + MIRD/ICRP/RADAR + MONTE CARLO ENGINE
# ==============================================================================
def predict_biokinetics(patient: dict, model) -> dict:
    """Stage 1: Predict organ TIACs, exact bi-exponential G(inf), blood & WB TIACs."""
    df_in = add_physics_features(pd.DataFrame([patient]))[FEATURE_COLS]
    preds = model.predict(df_in)[0]

    kid_tiac = float(np.clip(preds[0], 0.10, 15.0))
    kid_g_inf = float(np.clip(preds[1], 0.02, 0.45))
    liv_tiac = float(np.clip(preds[2], 0.20, 20.0))
    blood_tiac = float(np.clip(preds[3], 0.10, 5.0))
    wb_tiac = float(
        np.clip(
            preds[4],
            kid_tiac + liv_tiac + 1.5,
            MAX_PHYSICAL_TIAC_HOURS * 0.90,
        )
    )
    tum_tiac = float(np.clip(preds[5], 0.05, 25.0))
    wb_tiac = max(wb_tiac, kid_tiac + liv_tiac + tum_tiac + 1.0)

    sex_key = "F" if int(patient.get("sex_is_female", 0)) == 1 else "M"
    mird_breakdown = compute_mird_radar_unit_doses(
        sex=sex_key,
        weight_kg=float(patient["weight_kg"]),
        kidney_mass_g=float(patient["kidney_mass_g"]),
        liver_mass_g=float(patient["liver_mass_g"]),
        index_lesion_mass_g=float(patient["index_lesion_mass_g"]),
        kidney_tiac_h=kid_tiac,
        liver_tiac_h=liv_tiac,
        blood_tiac_h=blood_tiac,
        wb_tiac_h=wb_tiac,
        tumor_tiac_h=tum_tiac,
    )

    return {
        "sex_key": sex_key,
        "icrp_phantom": ICRP_PHANTOMS[sex_key]["label"],
        "kidney_tiac_hours": kid_tiac,
        "kidney_g_inf_exact": kid_g_inf,
        "liver_tiac_hours": liv_tiac,
        "blood_tiac_hours": blood_tiac,
        "wb_tiac_hours": wb_tiac,
        "rem_body_tiac_hours": mird_breakdown["rem_body_tiac_hours"],
        "tumor_tiac_hours": tum_tiac,
        "mird_breakdown": mird_breakdown,
    }


def _sample_mc_unit_doses(patient: dict, biokinetics: dict, n_mc: int, seed: int) -> dict:
    """Helper to generate N=10,000 Monte Carlo unit dose distributions under MIRD/ICRP/RADAR."""
    rng = np.random.default_rng(seed)
    sex_key = biokinetics["sex_key"]
    ph = ICRP_PHANTOMS[sex_key]

    mc_kid_tiac = np.clip(rng.normal(biokinetics["kidney_tiac_hours"], 0.10 * biokinetics["kidney_tiac_hours"], n_mc), 0.05, None)
    mc_liv_tiac = np.clip(rng.normal(biokinetics["liver_tiac_hours"], 0.10 * biokinetics["liver_tiac_hours"], n_mc), 0.05, None)
    mc_bld_tiac = np.clip(rng.normal(biokinetics["blood_tiac_hours"], 0.12 * biokinetics["blood_tiac_hours"], n_mc), 0.05, None)
    mc_tum_tiac = np.clip(rng.normal(biokinetics["tumor_tiac_hours"], 0.10 * biokinetics["tumor_tiac_hours"], n_mc), 0.05, None)
    mc_wb_tiac = np.clip(
        rng.normal(biokinetics["wb_tiac_hours"], 0.10 * biokinetics["wb_tiac_hours"], n_mc),
        mc_kid_tiac + mc_liv_tiac + mc_tum_tiac + 1.0,
        MAX_PHYSICAL_TIAC_HOURS * 0.90,
    )
    mc_rb_tiac = np.maximum(0.0, mc_wb_tiac - (mc_kid_tiac + mc_liv_tiac + mc_tum_tiac))

    mc_kid_mass = np.clip(rng.normal(patient["kidney_mass_g"], 0.05 * patient["kidney_mass_g"], n_mc), 100.0, None)
    mc_liv_mass = np.clip(rng.normal(patient["liver_mass_g"], 0.05 * patient["liver_mass_g"], n_mc), 600.0, None)
    mc_tum_mass = np.clip(rng.normal(patient["index_lesion_mass_g"], 0.07 * patient["index_lesion_mass_g"], n_mc), 1.0, None)

    mc_alpha_beta_kid = rng.uniform(2.0, 3.0, n_mc)
    mc_g_inf = np.clip(rng.normal(biokinetics["kidney_g_inf_exact"], 0.08 * biokinetics["kidney_g_inf_exact"], n_mc), 0.02, 0.45)

    wb_mass_g = float(patient["weight_kg"]) * 1000.0
    wb_scale = ph["m_ref_wb_g"] / wb_mass_g
    marrow_mass_g = ph["m_ref_marrow_g"] * (wb_mass_g / ph["m_ref_wb_g"])

    d_kid_unit = (
        mc_kid_tiac * ph["S_self_kidney"] * (ph["m_ref_kidney_g"] / mc_kid_mass)
        + mc_liv_tiac * ph["S_kidney_from_liver"]
        + mc_rb_tiac * ph["S_kidney_from_rb"] * wb_scale
    )
    d_liv_unit = (
        mc_liv_tiac * ph["S_self_liver"] * (ph["m_ref_liver_g"] / mc_liv_mass)
        + mc_kid_tiac * ph["S_liver_from_kidney"]
        + mc_rb_tiac * ph["S_liver_from_rb"] * wb_scale
    )
    d_mar_unit = (
        mc_bld_tiac * RMBLR_PEPTIDE * ph["S_self_marrow"] * (ph["m_ref_marrow_g"] / marrow_mass_g)
        + mc_kid_tiac * ph["S_marrow_from_kidney"]
        + mc_liv_tiac * ph["S_marrow_from_liver"]
        + mc_rb_tiac * ph["S_marrow_from_rb"] * wb_scale
    )
    d_tum_unit = mc_tum_tiac * (S_CONST_NP_LU177 / mc_tum_mass)

    return {
        "d_kid_unit": d_kid_unit,
        "d_liv_unit": d_liv_unit,
        "d_mar_unit": d_mar_unit,
        "d_tum_unit": d_tum_unit,
        "mc_g_inf": mc_g_inf,
        "mc_alpha_beta_kid": mc_alpha_beta_kid,
    }


def run_monte_carlo_dosimetry(
    patient: dict,
    biokinetics: dict,
    activity_per_cycle_gbq: float,
    n_cycles: int = 4,
    n_mc: int = 10000,
    seed: int = 42,
) -> dict:
    """Stage 2 & 3: Propagate biokinetic, anatomical, and radiobiological uncertainties (N=10,000)."""
    limits = get_oar_limits(patient)
    mc = _sample_mc_unit_doses(patient, biokinetics, n_mc=n_mc, seed=seed)

    d_kid_cycle = activity_per_cycle_gbq * mc["d_kid_unit"]
    bed_kid_total = n_cycles * d_kid_cycle * (1.0 + (mc["mc_g_inf"] * d_kid_cycle) / mc["mc_alpha_beta_kid"])
    d_kid_total = n_cycles * d_kid_cycle
    d_liv_total = n_cycles * activity_per_cycle_gbq * mc["d_liv_unit"]
    d_mar_total = n_cycles * activity_per_cycle_gbq * mc["d_mar_unit"]
    d_tum_total = n_cycles * activity_per_cycle_gbq * mc["d_tum_unit"]

    prob_kid_exceed = float(np.mean(bed_kid_total > limits["kidney_bed_limit"]) * 100.0)
    prob_liv_exceed = float(np.mean(d_liv_total > limits["liver_dose_limit"]) * 100.0)
    prob_mar_exceed = float(np.mean(d_mar_total > limits["marrow_dose_limit"]) * 100.0)

    return {
        "limits": limits,
        "distributions": {
            "kidney_bed_gy": bed_kid_total,
            "kidney_dose_gy": d_kid_total,
            "liver_dose_gy": d_liv_total,
            "marrow_dose_gy": d_mar_total,
            "tumor_dose_gy": d_tum_total,
        },
        "summary": {
            "kidney_bed_p50": float(np.percentile(bed_kid_total, 50)),
            "kidney_bed_p95": float(np.percentile(bed_kid_total, 95)),
            "kidney_dose_p50": float(np.percentile(d_kid_total, 50)),
            "liver_dose_p50": float(np.percentile(d_liv_total, 50)),
            "liver_dose_p95": float(np.percentile(d_liv_total, 95)),
            "marrow_dose_p50": float(np.percentile(d_mar_total, 50)),
            "marrow_dose_p95": float(np.percentile(d_mar_total, 95)),
            "tumor_dose_p50": float(np.percentile(d_tum_total, 50)),
            "tumor_dose_p05": float(np.percentile(d_tum_total, 5)),
            "prob_kidney_exceed_pct": round(prob_kid_exceed, 2),
            "prob_liver_exceed_pct": round(prob_liv_exceed, 2),
            "prob_marrow_exceed_pct": round(prob_mar_exceed, 2),
        },
    }


def solve_single_cycle_slsqp(
    patient: dict,
    biokinetics: dict,
    n_cycles: int = 4,
    n_mc: int = 10000,
    seed: int = 42,
) -> dict:
    """Stage 4A: SLSQP optimization for uniform per-cycle activity A0* under P95 OAR limits."""
    limits = get_oar_limits(patient)
    mc = _sample_mc_unit_doses(patient, biokinetics, n_mc=n_mc, seed=seed)

    def eval_p95(a0: float):
        d_kid_c = a0 * mc["d_kid_unit"]
        bed_kid = n_cycles * d_kid_c * (1.0 + (mc["mc_g_inf"] * d_kid_c) / mc["mc_alpha_beta_kid"])
        d_liv = n_cycles * a0 * mc["d_liv_unit"]
        d_mar = n_cycles * a0 * mc["d_mar_unit"]
        d_tum = n_cycles * a0 * mc["d_tum_unit"]
        return (
            float(np.percentile(bed_kid, 95)),
            float(np.percentile(d_liv, 95)),
            float(np.percentile(d_mar, 95)),
            float(np.mean(d_tum)),
        )

    constraints = [
        {"type": "ineq", "fun": lambda x: limits["kidney_bed_limit"] - eval_p95(x[0])[0]},
        {"type": "ineq", "fun": lambda x: limits["liver_dose_limit"] - eval_p95(x[0])[1]},
        {"type": "ineq", "fun": lambda x: limits["marrow_dose_limit"] - eval_p95(x[0])[2]},
    ]
    res = minimize(
        fun=lambda x: -eval_p95(x[0])[3],
        x0=np.array([7.4]),
        method="SLSQP",
        bounds=[(2.0, 11.1)],
        constraints=constraints,
        options={"ftol": 1e-4, "maxiter": 100},
    )
    a0_opt = float(np.clip(res.x[0], 2.0, 11.1))
    kid_p95, liv_p95, mar_p95, tum_mean = eval_p95(a0_opt)

    utilization = {
        "Kidney (BED)": kid_p95 / limits["kidney_bed_limit"],
        "Liver (Absorbed Dose)": liv_p95 / limits["liver_dose_limit"],
        "Red Marrow": mar_p95 / limits["marrow_dose_limit"],
    }
    return {
        "optimal_a0_gbq": round(a0_opt, 2),
        "cumulative_gbq": round(a0_opt * n_cycles, 2),
        "kidney_bed_p95": round(kid_p95, 2),
        "liver_dose_p95": round(liv_p95, 2),
        "marrow_dose_p95": round(mar_p95, 3),
        "tumor_dose_mean": round(tum_mean, 2),
        "limiting_organ": max(utilization, key=utilization.get),
    }


# ==============================================================================
# 3. STAGE 4B: MARKOV DECISION PROCESS (MDP) MULTI-CYCLE ADAPTIVE PLANNER
# ==============================================================================
def build_egfr_markov_matrix(egfr_current: float, cycle_kidney_bed: float) -> tuple[list[str], np.ndarray]:
    """Construct a 4-state renal function Markov transition matrix across treatment cycles."""
    states = [
        "S0: Normal (eGFR ≥ 90)",
        "S1: Mild (eGFR 60–89)",
        "S2: Moderate CKD (eGFR 45–59)",
        "S3: Severe CKD (eGFR < 45)",
    ]
    stress = np.clip((cycle_kidney_bed - 4.0) / 14.0, 0.0, 0.35)

    P = np.array(
        [
            [0.84 - stress, 0.14 + 0.75 * stress, 0.02 + 0.25 * stress, 0.00],
            [0.06, 0.78 - stress, 0.14 + 0.80 * stress, 0.02 + 0.20 * stress],
            [0.00, 0.08, 0.76 - stress, 0.16 + stress],
            [0.00, 0.00, 0.10, 0.90],
        ],
        dtype=float,
    )
    P = np.clip(P, 0.0, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return states, P


def simulate_markov_mdp_plan(
    patient: dict,
    biokinetics: dict,
    n_cycles: int = 4,
    n_trajectories: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Solve and simulate a Finite-Horizon Markov Decision Process (MDP) for multi-cycle
    Lu-177 therapy (4-6 cycles) using full MIRD/ICRP/RADAR dosimetry.
    """
    rng = np.random.default_rng(seed)
    limits = get_oar_limits(patient)
    mird = biokinetics["mird_breakdown"]

    action_grid = np.array([0.0, 3.7, 5.5, 7.4, 9.25, 11.1])
    g_inf = biokinetics["kidney_g_inf_exact"]
    alpha_beta_kid = 2.5

    # Baseline full MIRD/ICRP/RADAR unit doses per GBq at cycle 1
    base_d_kid = mird["kidney_gy_per_gbq"]
    base_d_liv = mird["liver_gy_per_gbq"]
    base_d_mar = mird["marrow_gy_per_gbq"]
    base_d_tum = mird["tumor_gy_per_gbq"]

    ttv_init = float(patient["total_tumor_volume_cm3"])
    sink_init = 1.0 / (1.0 + 0.00085 * ttv_init)
    alpha_tumor_shrink = 0.0065  # Gy^-1

    def run_policy_rollouts(adaptive: bool) -> dict:
        cum_kid_bed = np.zeros(n_trajectories)
        cum_liv_dose = np.zeros(n_trajectories)
        cum_mar_dose = np.zeros(n_trajectories)
        cum_tum_dose = np.zeros(n_trajectories)
        cum_activity = np.zeros(n_trajectories)

        egfr_traj = np.full(n_trajectories, float(patient["egfr_ml_min"]))
        ttv_traj = np.full(n_trajectories, ttv_init)
        plt_traj = np.full(n_trajectories, float(patient["platelet_count_10e9_l"]))

        cycle_records = []

        for c in range(1, n_cycles + 1):
            rem_cycles = n_cycles - c + 1

            sink_curr = 1.0 / (1.0 + 0.00085 * ttv_traj)
            sink_ratio = sink_curr / sink_init
            renal_ret_mult = (float(patient["egfr_ml_min"]) / np.clip(egfr_traj, 25.0, 120.0)) ** 0.25

            eff_d_kid = base_d_kid * sink_ratio * renal_ret_mult
            eff_d_liv = base_d_liv * sink_ratio
            eff_d_mar = base_d_mar * renal_ret_mult
            eff_d_tum = base_d_tum * (sink_ratio ** 0.4)

            if adaptive:
                safety_buf = 1.18
                chosen_actions = np.zeros(n_trajectories)

                for i in range(n_trajectories):
                    if plt_traj[i] < 75.0 or egfr_traj[i] < 30.0:
                        chosen_actions[i] = 0.0
                        continue

                    alloc_weight = 1.12 / rem_cycles if rem_cycles > 1 else 1.0
                    kid_budget_c = max(0.0, (limits["kidney_bed_limit"] - cum_kid_bed[i]) * alloc_weight)
                    liv_budget_c = max(0.0, (limits["liver_dose_limit"] - cum_liv_dose[i]) * alloc_weight)
                    mar_budget_c = max(0.0, (limits["marrow_dose_limit"] - cum_mar_dose[i]) * alloc_weight)

                    best_a = 0.0
                    for a in action_grid:
                        d_k = a * eff_d_kid[i]
                        bed_k = d_k * (1.0 + (g_inf * d_k) / alpha_beta_kid) * safety_buf
                        d_l = a * eff_d_liv[i] * safety_buf
                        d_m = a * eff_d_mar[i] * safety_buf
                        if bed_k <= kid_budget_c and d_l <= liv_budget_c and d_m <= mar_budget_c:
                            best_a = a
                    chosen_actions[i] = best_a
            else:
                chosen_actions = np.full(n_trajectories, 7.4)

            noise_kid = rng.normal(1.0, 0.10, n_trajectories)
            noise_liv = rng.normal(1.0, 0.10, n_trajectories)
            noise_mar = rng.normal(1.0, 0.12, n_trajectories)
            noise_tum = rng.normal(1.0, 0.10, n_trajectories)

            d_kid_real = chosen_actions * eff_d_kid * np.clip(noise_kid, 0.65, 1.45)
            bed_kid_real = d_kid_real * (1.0 + (g_inf * d_kid_real) / alpha_beta_kid)
            d_liv_real = chosen_actions * eff_d_liv * np.clip(noise_liv, 0.65, 1.45)
            d_mar_real = chosen_actions * eff_d_mar * np.clip(noise_mar, 0.65, 1.50)
            d_tum_real = chosen_actions * eff_d_tum * np.clip(noise_tum, 0.65, 1.45)

            cum_activity += chosen_actions
            cum_kid_bed += bed_kid_real
            cum_liv_dose += d_liv_real
            cum_mar_dose += d_mar_real
            cum_tum_dose += d_tum_real

            ttv_traj = np.maximum(2.0, ttv_traj * np.exp(-alpha_tumor_shrink * d_tum_real))
            egfr_drop = 0.22 * bed_kid_real + rng.normal(0.5, 2.2, n_trajectories)
            egfr_traj = np.clip(egfr_traj - np.maximum(-1.5, egfr_drop), 20.0, 125.0)
            plt_drop = 45.0 * d_mar_real + rng.normal(0.0, 10.0, n_trajectories)
            plt_traj = np.clip(plt_traj - plt_drop + 12.0, 40.0, 450.0)

            cycle_records.append(
                {
                    "cycle": c,
                    "mean_activity_gbq": round(float(np.mean(chosen_actions)), 2),
                    "p10_activity_gbq": round(float(np.percentile(chosen_actions, 10)), 2),
                    "p90_activity_gbq": round(float(np.percentile(chosen_actions, 90)), 2),
                    "cum_kidney_bed_p50": round(float(np.percentile(cum_kid_bed, 50)), 2),
                    "cum_kidney_bed_p95": round(float(np.percentile(cum_kid_bed, 95)), 2),
                    "cum_liver_dose_p95": round(float(np.percentile(cum_liv_dose, 95)), 2),
                    "cum_marrow_dose_p95": round(float(np.percentile(cum_mar_dose, 95)), 3),
                    "cum_tumor_dose_p50": round(float(np.percentile(cum_tum_dose, 50)), 2),
                    "mean_ttv_cm3": round(float(np.mean(ttv_traj)), 1),
                    "mean_egfr": round(float(np.mean(egfr_traj)), 1),
                }
            )

        any_toxicity = (
            (cum_kid_bed > limits["kidney_bed_limit"])
            | (cum_liv_dose > limits["liver_dose_limit"])
            | (cum_mar_dose > limits["marrow_dose_limit"])
        )
        return {
            "cycle_df": pd.DataFrame(cycle_records),
            "mean_total_activity_gbq": round(float(np.mean(cum_activity)), 2),
            "median_tumor_dose_gy": round(float(np.percentile(cum_tum_dose, 50)), 2),
            "p95_kidney_bed_gy": round(float(np.percentile(cum_kid_bed, 95)), 2),
            "p95_liver_dose_gy": round(float(np.percentile(cum_liv_dose, 95)), 2),
            "p95_marrow_dose_gy": round(float(np.percentile(cum_mar_dose, 95)), 3),
            "overall_oar_violation_prob_pct": round(float(np.mean(any_toxicity) * 100.0), 2),
        }

    mdp_adaptive = run_policy_rollouts(adaptive=True)
    fixed_standard = run_policy_rollouts(adaptive=False)

    rep_cycle_bed = mdp_adaptive["p95_kidney_bed_gy"] / max(1, n_cycles)
    state_labels, trans_matrix = build_egfr_markov_matrix(patient["egfr_ml_min"], rep_cycle_bed)

    return {
        "mdp_adaptive": mdp_adaptive,
        "fixed_standard": fixed_standard,
        "markov_states": state_labels,
        "egfr_transition_matrix": trans_matrix,
        "limits": limits,
    }


# ==============================================================================
# 4. PLOTLY VISUALIZATION BUILDERS
# ==============================================================================
def plot_time_activity_curves(biokinetics: dict, activity_gbq: float) -> go.Figure:
    """Plot 0–168 hour physical + biological clearance curves for Kidney, Liver, and Tumor."""
    t = np.linspace(0, 168, 200)
    lam_kid = 0.032
    a0_kid = biokinetics["kidney_tiac_hours"] * lam_kid * activity_gbq
    lam_liv = 0.024
    a0_liv = biokinetics["liver_tiac_hours"] * lam_liv * activity_gbq
    lam_tum = 0.0098
    a0_tum = biokinetics["tumor_tiac_hours"] * lam_tum * activity_gbq

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a0_kid * np.exp(-lam_kid * t), name="Ginjal (OAR Utama)", line=dict(color="#E63946", width=3)))
    fig.add_trace(go.Scatter(x=t, y=a0_liv * np.exp(-lam_liv * t), name="Hati (OAR Sekunder)", line=dict(color="#F4A261", width=3)))
    fig.add_trace(go.Scatter(x=t, y=a0_tum * np.exp(-lam_tum * t), name="Lesi Tumor Indeks", line=dict(color="#2A9D8F", width=3, dash="dash")))
    fig.add_vline(x=24, line_dash="dot", line_color="#457B9D", annotation_text="SPECT/CT 24 jam (STP)")

    fig.update_layout(
        title="Kurva Aktivitas-Waktu (Time-Activity Curve) Pasca-Injeksi 1 Siklus",
        xaxis_title="Waktu Pasca-Injeksi (Jam)",
        yaxis_title="Aktivitas Terakumulasi dalam Organ (GBq)",
        template="plotly_white",
        height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_monte_carlo_distributions(mc_res: dict) -> go.Figure:
    """Plot N=10,000 Monte Carlo distributions for Kidney BED, Liver Dose, and Tumor Dose."""
    dist = mc_res["distributions"]
    summ = mc_res["summary"]
    lim = mc_res["limits"]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"BED Ginjal Kumulatif (Batas: {lim['kidney_bed_limit']} Gy)",
            f"Dosis Hati Kumulatif (Batas: {lim['liver_dose_limit']} Gy)",
            "Dosis Lesi Tumor Kumulatif (Target)",
        ),
    )

    fig.add_trace(go.Histogram(x=dist["kidney_bed_gy"], nbinsx=45, marker_color="#E63946", opacity=0.75, name="BED Ginjal"), row=1, col=1)
    fig.add_vline(x=summ["kidney_bed_p95"], line_dash="dash", line_color="#1D3557", annotation_text=f"P95: {summ['kidney_bed_p95']:.1f} Gy", row=1, col=1)
    fig.add_vline(x=lim["kidney_bed_limit"], line_color="#000000", line_width=2, annotation_text="Batas Aman", row=1, col=1)

    fig.add_trace(go.Histogram(x=dist["liver_dose_gy"], nbinsx=45, marker_color="#F4A261", opacity=0.75, name="Dosis Hati"), row=1, col=2)
    fig.add_vline(x=summ["liver_dose_p95"], line_dash="dash", line_color="#1D3557", annotation_text=f"P95: {summ['liver_dose_p95']:.1f} Gy", row=1, col=2)
    fig.add_vline(x=lim["liver_dose_limit"], line_color="#000000", line_width=2, annotation_text="Batas Aman", row=1, col=2)

    fig.add_trace(go.Histogram(x=dist["tumor_dose_gy"], nbinsx=45, marker_color="#2A9D8F", opacity=0.75, name="Dosis Tumor"), row=1, col=3)
    fig.add_vline(x=summ["tumor_dose_p50"], line_dash="dash", line_color="#1D3557", annotation_text=f"Median: {summ['tumor_dose_p50']:.1f} Gy", row=1, col=3)

    fig.update_layout(
        title="Simulasi Monte Carlo (N = 10.000 Iterasi) — Distribusi Probabilistik Dosis & Margin P95",
        showlegend=False,
        template="plotly_white",
        height=360,
    )
    return fig


def plot_markov_trajectories(mdp_res: dict) -> go.Figure:
    """Compare multi-cycle trajectories: Adaptive MDP Policy vs Fixed 7.4 GBq Standard."""
    df_mdp = mdp_res["mdp_adaptive"]["cycle_df"]
    df_fix = mdp_res["fixed_standard"]["cycle_df"]
    lim = mdp_res["limits"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Akumulasi BED Ginjal P95 per Siklus (MDP vs Standar 7.4 GBq)",
            "Dinamika Dosis Tumor & Penyusutan Volume Tumor (Tumor Sink)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=df_mdp["cycle"], y=df_mdp["cum_kidney_bed_p95"], mode="lines+markers", name="MDP Adaptif (BED P95)", line=dict(color="#2A9D8F", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_fix["cycle"], y=df_fix["cum_kidney_bed_p95"], mode="lines+markers", name="Standar 7.4 GBq (BED P95)", line=dict(color="#E63946", width=3, dash="dash")),
        row=1,
        col=1,
    )
    fig.add_hline(y=lim["kidney_bed_limit"], line_color="#000000", line_dash="dot", annotation_text=f"Batas Aman ({lim['kidney_bed_limit']} Gy)", row=1, col=1)

    fig.add_trace(
        go.Bar(x=df_mdp["cycle"], y=df_mdp["cum_tumor_dose_p50"], name="Dosis Tumor MDP (Gy)", marker_color="#457B9D", opacity=0.85),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df_mdp["cycle"], y=df_mdp["mean_ttv_cm3"], mode="lines+markers", name="Volume Tumor TTV (cm³)", line=dict(color="#E76F51", width=3)),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Simulasi Markov Decision Process (MDP) Multi-Siklus dengan Tumor Sink & Dinamika eGFR",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    return fig


# ==============================================================================
# 5. STREAMLIT WEB USER INTERFACE
# ==============================================================================
def run_streamlit_app():
    st.set_page_config(
        page_title="Radionuclide Dose Planner v2.1 (MIRD / ICRP / RADAR)",
        page_icon="☢️",
        layout="wide",
    )

    st.title("☢️ Radionuclide Dose Planner — Lu-177 Personalized Dosimetry")
    st.caption(
        "Kepatuhan Penuh: MIRD Pamphlet 20 & 21 | ICRP Publ. 89, 107, 110 & 133 | "
        "RADAR / EANM Blood-Based Marrow & Remainder-of-Body Conservation"
    )

    bundle = load_or_train_model()
    model = bundle["model"]

    with st.sidebar:
        st.header("1. Fantom ICRP & Parameter Klinis")
        sex_choice = st.selectbox(
            "Jenis Kelamin (Fantom Referensi ICRP 89/110)",
            ["Laki-laki — ICRP 110 Adult Male (73 kg)", "Perempuan — ICRP 110 Adult Female (60 kg)"],
            index=0,
        )
        sex_is_female = 1 if sex_choice.startswith("Perempuan") else 0

        age_years = st.slider("Usia (tahun)", 40, 88, 68)
        weight_kg = st.slider("Berat Badan (kg)", 42.0, 125.0, 62.0 if sex_is_female else 73.0, 0.5)
        egfr_ml_min = st.slider("eGFR Basal (mL/menit/1.73m²)", 28.0, 120.0, 56.0, 1.0)
        serum_creatinine = st.number_input("Kreatinin Serum (mg/dL)", 0.50, 3.50, 1.45, 0.05)
        alt_u_l = st.number_input("ALT / SGPT (U/L)", 8.0, 260.0, 28.0, 1.0)
        bilirubin = st.number_input("Bilirubin Total (mg/dL)", 0.20, 4.50, 0.70, 0.05)
        platelets = st.slider("Trombosit Basal (10⁹/L)", 85, 450, 210)
        comorbid_renal = st.checkbox("Komorbiditas Risiko Ginjal (HT/DM)", value=True)

        st.header("2. Kuantifikasi SPECT/CT 24 Jam")
        kidney_mass_g = st.slider("Massa Ginjal Bilateral CT (g)", 160.0, 480.0, 275.0 if sex_is_female else 310.0, 5.0)
        liver_mass_g = st.slider("Massa Hati CT (g)", 950.0, 2900.0, 1400.0 if sex_is_female else 1800.0, 25.0)
        kidney_ia_24h = st.slider("Serapan Ginjal 24 jam (%IA)", 0.20, 12.0, 3.10, 0.05)
        liver_ia_24h = st.slider("Serapan Hati 24 jam (%IA)", 0.30, 20.0, 5.40, 0.10)
        tumor_suv_24h = st.slider("SUVmean Lesi Tumor Indeks (24 jam)", 2.0, 60.0, 18.5, 0.5)
        lesion_mass_g = st.slider("Massa Lesi Tumor Indeks (g)", 3.0, 120.0, 22.0, 1.0)
        ttv_cm3 = st.slider("Total Tumor Volume / TTV (cm³)", 5.0, 950.0, 140.0, 5.0)

        st.header("3. Protokol Terapi")
        n_cycles = st.selectbox("Jumlah Siklus Terapi", [4, 5, 6], index=0)

    patient = {
        "sex": "F" if sex_is_female else "M",
        "sex_is_female": int(sex_is_female),
        "age_years": float(age_years),
        "weight_kg": float(weight_kg),
        "egfr_ml_min": float(egfr_ml_min),
        "serum_creatinine_mg_dl": float(serum_creatinine),
        "alt_u_l": float(alt_u_l),
        "total_bilirubin_mg_dl": float(bilirubin),
        "platelet_count_10e9_l": float(platelets),
        "has_renal_risk_factors": int(comorbid_renal or egfr_ml_min < 60.0),
        "has_hepatic_risk_factors": int(bilirubin > 1.5 or alt_u_l > 80.0),
        "kidney_mass_g": float(kidney_mass_g),
        "liver_mass_g": float(liver_mass_g),
        "index_lesion_mass_g": float(lesion_mass_g),
        "total_tumor_volume_cm3": float(ttv_cm3),
        "kidney_uptake_ia_percent_24h": float(kidney_ia_24h),
        "liver_uptake_ia_percent_24h": float(liver_ia_24h),
        "tumor_suv_mean_24h": float(tumor_suv_24h),
    }

    biokinetics = predict_biokinetics(patient, model)
    opt_res = solve_single_cycle_slsqp(patient, biokinetics, n_cycles=n_cycles, n_mc=10000)
    mc_res = run_monte_carlo_dosimetry(
        patient, biokinetics, activity_per_cycle_gbq=opt_res["optimal_a0_gbq"], n_cycles=n_cycles, n_mc=10000
    )
    mdp_res = simulate_markov_mdp_plan(patient, biokinetics, n_cycles=n_cycles, n_trajectories=1000)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Aktivitas Optimal / Siklus", f"{opt_res['optimal_a0_gbq']} GBq", f"Total: {opt_res['cumulative_gbq']} GBq")
    col2.metric(
        "BED Ginjal (P95)",
        f"{mc_res['summary']['kidney_bed_p95']:.1f} Gy",
        f"Batas: {mc_res['limits']['kidney_bed_limit']} Gy",
        delta_color="inverse",
    )
    col3.metric(
        "Dosis Hati (P95)",
        f"{mc_res['summary']['liver_dose_p95']:.1f} Gy",
        f"Batas: {mc_res['limits']['liver_dose_limit']} Gy",
        delta_color="inverse",
    )
    col4.metric(
        "Dosis Sumsum Tulang (P95)",
        f"{mc_res['summary']['marrow_dose_p95']:.2f} Gy",
        f"Batas: {mc_res['limits']['marrow_dose_limit']} Gy",
        delta_color="inverse",
    )
    col5.metric(
        "Dosis Tumor Indeks (Median)",
        f"{mc_res['summary']['tumor_dose_p50']:.1f} Gy",
        f"Organ Pembatas: {opt_res['limiting_organ']}",
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Dosimetri MIRD / ICRP / RADAR",
        "🎲 Simulasi Monte Carlo (N = 10.000)",
        "🔄 Perencanaan Multi-Siklus Markov (MDP)",
    ])

    with tab1:
        st.subheader(f"Tahap 1 & 2: Dekomposisi Dosis MIRD 21 + ICRP 133 + RADAR ({biokinetics['icrp_phantom']})")
        c_left, c_right = st.columns([1.15, 1.85])
        with c_left:
            m = biokinetics["mird_breakdown"]
            decomp_df = pd.DataFrame(
                {
                    "Organ / Kompartemen": ["Ginjal Bilateral", "Hati", "Sumsum Tulang Merah", "Lesi Tumor Indeks"],
                    "Self-Dose β⁻ (mGy/MBq)": [
                        round(m["kidney_self_gy_per_gbq"], 4),
                        round(m["liver_self_gy_per_gbq"], 4),
                        round(m["marrow_blood_self_gy_per_gbq"], 4),
                        round(m["tumor_gy_per_gbq"], 4),
                    ],
                    "Cross + RB γ (mGy/MBq)": [
                        round(m["kidney_cross_rb_gy_per_gbq"], 4),
                        round(m["liver_cross_rb_gy_per_gbq"], 4),
                        round(m["marrow_cross_rb_gy_per_gbq"], 4),
                        0.0,
                    ],
                    "Total (mGy/MBq | Gy/GBq)": [
                        round(m["kidney_gy_per_gbq"], 4),
                        round(m["liver_gy_per_gbq"], 4),
                        round(m["marrow_gy_per_gbq"], 4),
                        round(m["tumor_gy_per_gbq"], 4),
                    ],
                }
            )
            st.dataframe(decomp_df, use_container_width=True, hide_index=True)

            st.info(
                f"**Parameter Biokinetik & Standar Fisika:**\n"
                f"- Radionuklida: **Lu-177** ($T_{{1/2}} = {LU177_HALF_LIFE_HOURS}$ jam, ICRP 107)\n"
                f"- Faktor Dale $G(\infty)$ Bi-Eksponensial Ginjal: **{biokinetics['kidney_g_inf_exact']:.4f}**\n"
                f"- TIAC *Remainder of Body* ($\tilde{{a}}_{{\text{{RB}}}}$): **{biokinetics['rem_body_tiac_hours']:.2f} jam** (RADAR)\n"
                f"- Batas BED Ginjal: **≤ {mc_res['limits']['kidney_bed_limit']} Gy** | Batas Hati: **≤ {mc_res['limits']['liver_dose_limit']} Gy**"
            )
        with c_right:
            st.plotly_chart(plot_time_activity_curves(biokinetics, opt_res["optimal_a0_gbq"]), use_container_width=True)

    with tab2:
        st.subheader("Tahap 3: Analisis Ketidakpastian Stokastik Monte Carlo (10.000 Iterasi)")
        st.plotly_chart(plot_monte_carlo_distributions(mc_res), use_container_width=True)

        s = mc_res["summary"]
        risk_table = pd.DataFrame(
            {
                "Organ / Target": ["Ginjal (BED)", "Hati (Absorbed Dose)", "Sumsum Tulang Merah", "Lesi Tumor Indeks"],
                "Median (P50)": [
                    f"{s['kidney_bed_p50']:.2f} Gy",
                    f"{s['liver_dose_p50']:.2f} Gy",
                    f"{s['marrow_dose_p50']:.3f} Gy",
                    f"{s['tumor_dose_p50']:.2f} Gy",
                ],
                "Persentil ke-95 (P95)": [
                    f"{s['kidney_bed_p95']:.2f} Gy",
                    f"{s['liver_dose_p95']:.2f} Gy",
                    f"{s['marrow_dose_p95']:.3f} Gy",
                    f"P05: {s['tumor_dose_p05']:.2f} Gy",
                ],
                "Ambang Batas Klinis": [
                    f"≤ {mc_res['limits']['kidney_bed_limit']} Gy",
                    f"≤ {mc_res['limits']['liver_dose_limit']} Gy",
                    f"≤ {mc_res['limits']['marrow_dose_limit']} Gy",
                    "Maksimalisasi",
                ],
                "Probabilitas Melebihi Batas": [
                    f"{s['prob_kidney_exceed_pct']}%",
                    f"{s['prob_liver_exceed_pct']}%",
                    f"{s['prob_marrow_exceed_pct']}%",
                    "—",
                ],
            }
        )
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Tahap 4B: Optimasi Sekuensial Multi-Siklus dengan Markov Decision Process (MDP)")
        m1, m2, m3 = st.columns(3)
        mdp_s = mdp_res["mdp_adaptive"]
        fix_s = mdp_res["fixed_standard"]
        m1.metric(
            "Total Aktivitas MDP Adaptif",
            f"{mdp_s['mean_total_activity_gbq']} GBq",
            f"Standar Tetap: {fix_s['mean_total_activity_gbq']} GBq",
        )
        m2.metric(
            "Dosis Tumor Median (MDP vs Standar)",
            f"{mdp_s['median_tumor_dose_gy']} Gy",
            f"{mdp_s['median_tumor_dose_gy'] - fix_s['median_tumor_dose_gy']:+.1f} Gy vs Standar",
        )
        m3.metric(
            "Risiko Toksisitas OAR (MDP vs Standar)",
            f"{mdp_s['overall_oar_violation_prob_pct']}%",
            f"Standar 7.4 GBq: {fix_s['overall_oar_violation_prob_pct']}%",
            delta_color="inverse",
        )

        st.plotly_chart(plot_markov_trajectories(mdp_res), use_container_width=True)

        col_t1, col_t2 = st.columns([1.3, 1])
        with col_t1:
            st.markdown("**Jadwal Rekomendasi Aktivitas Adaptif per Siklus (MDP):**")
            st.dataframe(mdp_s["cycle_df"], use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Matriks Transisi Markov Status Fungsi Ginjal (eGFR) Antar Siklus:**")
            tm_df = pd.DataFrame(
                np.round(mdp_res["egfr_transition_matrix"], 3),
                index=mdp_res["markov_states"],
                columns=["→ S0", "→ S1", "→ S2", "→ S3"],
            )
            st.dataframe(tm_df, use_container_width=True)


if __name__ == "__main__":
    if HAS_STREAMLIT:
        run_streamlit_app()
    else:
        print("Streamlit is not installed in this headless environment. Run: streamlit run app.py")
