from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize


@dataclass
class PatientDosimetryProfile:
    """Parameter dosimetri pasien-spesifik hasil ekstraksi VOI & konvolusi VSV Tahap 3."""
    d_tumor_gy_per_gbq: float       # Dosis serap rata-rata tumor per GBq (Gy/GBq)
    d_kidney_gy_per_gbq: float      # Dosis serap rata-rata ginjal per GBq (Gy/GBq)
    d_marrow_gy_per_gbq: float      # Dosis serap rata-rata sumsum tulang per GBq (Gy/GBq)
    d_salivary_gy_per_gbq: float    # Dosis serap kelenjar saliva per GBq (Gy/GBq)
    t_eff_tumor_h: float = 68.0
    t_eff_kidney_h: float = 52.0
    t_eff_marrow_h: float = 42.0
    t_eff_salivary_h: float = 48.0
    baseline_egfr: float = 78.0     # mL/min/1.73 m^2
    has_renal_risk_factors: bool = False  # Hipertensi, DM, atau kemoterapi nefrotoksik sebelumnya
    sigma_rel_uncertainty: float = 0.10   # Ketidakpastian relatif dari Monte Carlo (Tahap 3)


class SLSQPDoseOptimizer:
    """
    Tahap 4A: Optimasi Aktivitas Injeksi per Siklus (GBq) berbasis SLSQP
    dengan kendala radiobiologis BED dan koreksi Tumor Sink Effect.
    """

    def __init__(
        self,
        profile: PatientDosimetryProfile,
        n_cycles: int = 4,
        gamma_shrink: float = 0.014,
        kappa_sink: float = 0.25,
    ):
        self.profile = profile
        self.n_cycles = n_cycles
        self.gamma_shrink = gamma_shrink
        self.kappa_sink = kappa_sink

    @staticmethod
    def compute_bed(dose_gy: float, t_eff_h: float, t_rep_h: float, alpha_beta: float) -> float:
        if dose_gy <= 0.0:
            return 0.0
        lam = np.log(2.0) / max(t_eff_h, 1e-3)
        mu = np.log(2.0) / max(t_rep_h, 1e-3)
        relative_effectiveness = 1.0 + (dose_gy / alpha_beta) * (lam / (lam + mu))
        return float(dose_gy * relative_effectiveness)

    def simulate_regimen(self, activities_gbq: np.ndarray, safety_z: float = 0.0) -> Dict:
        """
        Mensimulasikan trajektori dosis serap & BED lintas siklus untuk vektor aktivitas [A_1..A_N].
        `safety_z` (mis. 1.645 untuk margin 95%) menambahkan buffer konservatif pada dosis OAR.
        """
        cum_tumor_dose = 0.0
        tot_tumor_bed = 0.0
        tot_kidney_bed = 0.0
        tot_kidney_dose = 0.0
        tot_marrow_dose = 0.0
        tot_salivary_dose = 0.0
        cycle_records: List[Dict] = []

        uncertainty_mult = 1.0 + safety_z * self.profile.sigma_rel_uncertainty

        for idx, act in enumerate(activities_gbq, start=1):
            v_ratio = float(np.exp(-self.gamma_shrink * cum_tumor_dose))
            sink_mult = 1.0 + self.kappa_sink * (1.0 - v_ratio)

            dt = float(act * self.profile.d_tumor_gy_per_gbq)
            dk = float(act * self.profile.d_kidney_gy_per_gbq * sink_mult * uncertainty_mult)
            dm = float(act * self.profile.d_marrow_gy_per_gbq * sink_mult * uncertainty_mult)
            ds = float(act * self.profile.d_salivary_gy_per_gbq * sink_mult * uncertainty_mult)

            bed_t = self.compute_bed(dt, self.profile.t_eff_tumor_h, t_rep_h=1.5, alpha_beta=10.0)
            bed_k = self.compute_bed(dk, self.profile.t_eff_kidney_h, t_rep_h=2.8, alpha_beta=2.6)

            cum_tumor_dose += dt
            tot_tumor_bed += bed_t
            tot_kidney_bed += bed_k
            tot_kidney_dose += dk
            tot_marrow_dose += dm
            tot_salivary_dose += ds

            cycle_records.append(
                {
                    "Siklus": idx,
                    "Aktivitas (GBq)": round(float(act), 2),
                    "Fraksi Vol Tumor": round(v_ratio, 3),
                    "Sink Multiplier": round(sink_mult, 3),
                    "Dosis Tumor (Gy)": round(dt, 2),
                    "BED Tumor (Gy_10)": round(bed_t, 2),
                    "Dosis Ginjal (Gy)": round(dk, 2),
                    "BED Ginjal (Gy_2.6)": round(bed_k, 2),
                    "Dosis Sumsum (Gy)": round(dm, 3),
                }
            )

        return {
            "total_activity_gbq": round(float(np.sum(activities_gbq)), 2),
            "cum_tumor_dose_gy": round(cum_tumor_dose, 2),
            "tot_tumor_bed_gy10": round(tot_tumor_bed, 2),
            "tot_kidney_dose_gy": round(tot_kidney_dose, 2),
            "tot_kidney_bed_gy2_6": round(tot_kidney_bed, 2),
            "tot_marrow_dose_gy": round(tot_marrow_dose, 3),
            "tot_salivary_dose_gy": round(tot_salivary_dose, 2),
            "cycle_records": cycle_records,
        }

    def optimize(
        self,
        min_cycle_gbq: float = 0.0,
        max_cycle_gbq: float = 11.1,
        max_total_gbq: float = 44.4,
        marrow_limit_gy: float = 2.0,
        salivary_limit_gy: float = 35.0,
        probabilistic_z: float = 1.28,
    ) -> Dict:
        """
        Menjalankan optimasi SLSQP untuk menemukan jadwal aktivitas [A_1, ..., A_N] optimal.
        """
        # Stratifikasi batas BED ginjal berdasarkan guideline EANM/MIRD (Bodei et al.)
        kidney_bed_limit = (
            28.0
            if (self.profile.baseline_egfr < 60.0 or self.profile.has_renal_risk_factors)
            else 40.0
        )

        x0 = np.full(self.n_cycles, min(7.4, max_cycle_gbq), dtype=np.float64)
        bounds = [(min_cycle_gbq, max_cycle_gbq)] * self.n_cycles

        constraints = [
            {
                "type": "ineq",
                "fun": lambda a: kidney_bed_limit - self.simulate_regimen(a, probabilistic_z)["tot_kidney_bed_gy2_6"],
            },
            {
                "type": "ineq",
                "fun": lambda a: marrow_limit_gy - self.simulate_regimen(a, probabilistic_z)["tot_marrow_dose_gy"],
            },
            {
                "type": "ineq",
                "fun": lambda a: salivary_limit_gy - self.simulate_regimen(a, probabilistic_z)["tot_salivary_dose_gy"],
            },
            {
                "type": "ineq",
                "fun": lambda a: max_total_gbq - float(np.sum(a)),
            },
        ]

        res = minimize(
            fun=lambda a: -self.simulate_regimen(a, safety_z=0.0)["tot_tumor_bed_gy10"],
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 300},
        )

        optimal_activities = np.clip(res.x, min_cycle_gbq, max_cycle_gbq)
        nominal_eval = self.simulate_regimen(optimal_activities, safety_z=0.0)
        conservative_eval = self.simulate_regimen(optimal_activities, safety_z=probabilistic_z)

        return {
            "converged": bool(res.success),
            "message": str(res.message),
            "kidney_bed_limit_gy2_6": kidney_bed_limit,
            "optimal_activities_gbq": np.round(optimal_activities, 2).tolist(),
            "nominal_metrics": nominal_eval,
            "conservative_metrics": conservative_eval,
        }
