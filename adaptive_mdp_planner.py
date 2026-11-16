from typing import Dict, List, Tuple
import numpy as np
from slsqp_dose_optimizer import PatientDosimetryProfile, SLSQPDoseOptimizer


class AdaptiveMDPPlanner:
    """
    Tahap 4B: Perencanaan Multi-Siklus Adaptif berbasis Finite-Horizon Markov Decision Process (MDP)
    dengan Backward Induction Value Iteration dan Simulasi Monte Carlo.
    """

    ACTIONS_GBQ: List[float] = [0.0, 3.7, 5.55, 7.4, 9.25]
    EGFR_LABELS: Dict[int, str] = {
        0: "Normal (eGFR >= 60)",
        1: "Penurunan Ringan-Sedang (45-59)",
        2: "Gangguan Ginjal (< 45)",
    }
    TUMOR_LABELS: Dict[int, str] = {
        0: "100% Baseline",
        1: "75% (Respons Parsial Awal)",
        2: "50% (Respons Parsial)",
        3: "<=25% (Respons Mayor)",
    }

    def __init__(
        self,
        profile: PatientDosimetryProfile,
        n_cycles: int = 4,
        kappa_sink: float = 0.25,
        bed_step: float = 2.0,
    ):
        self.profile = profile
        self.n_cycles = n_cycles
        self.kappa_sink = kappa_sink
        self.bed_step = bed_step

        self.max_bed_limit = (
            28.0
            if (profile.baseline_egfr < 60.0 or profile.has_renal_risk_factors)
            else 40.0
        )
        self.n_bed_bins = int(np.ceil(self.max_bed_limit / self.bed_step)) + 1

        self.V: Optional[np.ndarray] = None
        self.policy: Optional[np.ndarray] = None

    @staticmethod
    def egfr_to_category(egfr: float) -> int:
        if egfr >= 60.0:
            return 0
        if egfr >= 45.0:
            return 1
        return 2

    def solve_policy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menyelesaikan persamaan Bellman finite-horizon menggunakan Backward Induction
        untuk menghasilkan nilai ekspektasi V(s) dan kebijakan optimal pi*(s).
        """
        V = np.zeros((self.n_cycles + 1, self.n_bed_bins, 3, 4), dtype=np.float64)
        policy = np.zeros((self.n_cycles, self.n_bed_bins, 3, 4), dtype=int)

        # Nilai terminal (setelah siklus N selesai)
        for b in range(self.n_bed_bins):
            for e in range(3):
                for tv in range(4):
                    if b == 0:
                        V[self.n_cycles, b, e, tv] = -160.0  # Penalti pelanggaran batas ginjal
                    else:
                        V[self.n_cycles, b, e, tv] = (tv * 16.0) - (e * 12.0)

        sink_multipliers = [
            1.0,
            1.0 + 0.25 * self.kappa_sink,
            1.0 + 0.50 * self.kappa_sink,
            1.0 + 0.80 * self.kappa_sink,
        ]
        # Penurunan eGFR memperlambat klirens renal dan menaikkan dosis serap ginjal
        egfr_renal_multipliers = [1.0, 1.18, 1.42]

        # 3 skenario stokastik klirens & serapan per siklus: (probabilitas, pengali dosis ginjal, prob penurunan eGFR)
        clearance_scenarios = [
            (0.60, 1.00, 0.12),  # Klirens nominal
            (0.25, 1.22, 0.32),  # Klirens melambat / uptake tinggi
            (0.15, 0.85, 0.04),  # Klirens cepat
        ]

        for k in range(self.n_cycles - 1, -1, -1):
            for b in range(self.n_bed_bins):
                rem_bed = b * self.bed_step
                for e in range(3):
                    for tv in range(4):
                        best_q = -1e9
                        best_a = 0

                        for a_idx, act_gbq in enumerate(self.ACTIONS_GBQ):
                            if act_gbq == 0.0:
                                # Tunda/hentikan terapi: nol BED tumor, peluang kecil pemulihan eGFR
                                q_val = 0.80 * V[k + 1, b, e, tv] + 0.20 * V[k + 1, b, max(0, e - 1), tv]
                                if q_val > best_q:
                                    best_q = q_val
                                    best_a = a_idx
                                continue

                            dt = act_gbq * self.profile.d_tumor_gy_per_gbq
                            dk_nominal = (
                                act_gbq
                                * self.profile.d_kidney_gy_per_gbq
                                * sink_multipliers[tv]
                                * egfr_renal_multipliers[e]
                            )
                            bed_tumor = SLSQPDoseOptimizer.compute_bed(
                                dt, self.profile.t_eff_tumor_h, t_rep_h=1.5, alpha_beta=10.0
                            )

                            q_val = 0.0
                            p_shrink = min(0.88, 0.18 + 0.018 * dt)
                            next_tv_shrink = min(3, tv + 1)

                            for prob_s, dk_mult, p_egfr_drop in clearance_scenarios:
                                bed_kidney = SLSQPDoseOptimizer.compute_bed(
                                    dk_nominal * dk_mult,
                                    self.profile.t_eff_kidney_h,
                                    t_rep_h=2.8,
                                    alpha_beta=2.6,
                                )
                                new_rem_bed = rem_bed - bed_kidney
                                if new_rem_bed <= 0.0:
                                    new_b = 0
                                    toxicity_penalty = -130.0
                                else:
                                    new_b = int(np.floor(new_rem_bed / self.bed_step))
                                    toxicity_penalty = 0.0

                                for e_next, p_e in [(e, 1.0 - p_egfr_drop), (min(2, e + 1), p_egfr_drop)]:
                                    renal_drop_pen = -35.0 if (e_next == 2 and e < 2) else 0.0
                                    for tv_next, p_tv in [(next_tv_shrink, p_shrink), (tv, 1.0 - p_shrink)]:
                                        trans_prob = prob_s * p_e * p_tv
                                        reward = bed_tumor + toxicity_penalty + renal_drop_pen
                                        q_val += trans_prob * (reward + V[k + 1, new_b, e_next, tv_next])

                            if q_val > best_q:
                                best_q = q_val
                                best_a = a_idx

                        V[k, b, e, tv] = best_q
                        policy[k, b, e, tv] = best_a

        self.V = V
        self.policy = policy
        return V, policy

    def recommend_next_action(
        self,
        current_cycle_1based: int,
        cumulative_kidney_bed_so_far: float,
        current_egfr: float,
        current_tumor_vol_fraction: float,
    ) -> Dict:
        """
        Memberikan rekomendasi keputusan klinis real-time untuk siklus berikutnya
        berdasarkan kondisi aktual pasien pasca-siklus sebelumnya.
        """
        if self.policy is None:
            self.solve_policy()

        k = int(np.clip(current_cycle_1based - 1, 0, self.n_cycles - 1))
        rem_bed = max(0.0, self.max_bed_limit - cumulative_kidney_bed_so_far)
        b_idx = int(np.clip(np.floor(rem_bed / self.bed_step), 0, self.n_bed_bins - 1))
        e_idx = self.egfr_to_category(current_egfr)

        if current_tumor_vol_fraction > 0.85:
            tv_idx = 0
        elif current_tumor_vol_fraction > 0.60:
            tv_idx = 1
        elif current_tumor_vol_fraction > 0.35:
            tv_idx = 2
        else:
            tv_idx = 3

        best_act_idx = int(self.policy[k, b_idx, e_idx, tv_idx])
        rec_gbq = self.ACTIONS_GBQ[best_act_idx]

        return {
            "siklus": current_cycle_1based,
            "sisa_kuota_bed_ginjal": round(rem_bed, 2),
            "kategori_egfr": self.EGFR_LABELS[e_idx],
            "status_tumor": self.TUMOR_LABELS[tv_idx],
            "rekomendasi_aktivitas_gbq": rec_gbq,
            "expected_cumulative_utility": round(float(self.V[k, b_idx, e_idx, tv_idx]), 2),
        }

    def run_monte_carlo_cohort_simulation(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, Dict[str, float]]:
        """
        Membandingkan performa probabilistik 3 strategi pada 1.000 pasien virtual:
        1. Protokol Tetap Standar (N x 7.4 GBq)
        2. Optimasi Statis Pra-Terapi (SLSQP)
        3. Kebijakan Adaptif Tertutup (MDP)
        """
        if self.policy is None:
            self.solve_policy()

        rng = np.random.default_rng(seed)
        slsqp_opt = SLSQPDoseOptimizer(self.profile, self.n_cycles, kappa_sink=self.kappa_sink)
        slsqp_acts = slsqp_opt.optimize()["optimal_activities_gbq"]

        sink_multipliers = [
            1.0,
            1.0 + 0.25 * self.kappa_sink,
            1.0 + 0.50 * self.kappa_sink,
            1.0 + 0.80 * self.kappa_sink,
        ]
        egfr_renal_multipliers = [1.0, 1.18, 1.42]

        strategies = ["Protokol Tetap (7.4 GBq)", "Optimasi SLSQP (Tahap 4A)", "Kebijakan Adaptif MDP (Tahap 4B)"]
        results = {s: {"tumor_bed": [], "kidney_bed": [], "violated": [], "total_gbq": []} for s in strategies}

        for _ in range(n_simulations):
            # Sampel heterogenitas pasien acak per siklus
            noise_k = rng.normal(1.0, self.profile.sigma_rel_uncertainty, size=self.n_cycles)
            noise_t = rng.normal(1.0, self.profile.sigma_rel_uncertainty, size=self.n_cycles)
            rand_u_egfr = rng.random(size=self.n_cycles)
            rand_u_tumor = rng.random(size=self.n_cycles)

            for strat in strategies:
                e_state = self.egfr_to_category(self.profile.baseline_egfr)
                tv_state = 0
                cum_t_bed = 0.0
                cum_k_bed = 0.0
                cum_gbq = 0.0

                for k in range(self.n_cycles):
                    if strat == "Protokol Tetap (7.4 GBq)":
                        act = 7.4
                    elif strat == "Optimasi SLSQP (Tahap 4A)":
                        act = slsqp_acts[k]
                    else:
                        rem_bed = max(0.0, self.max_bed_limit - cum_k_bed)
                        b_idx = int(np.clip(np.floor(rem_bed / self.bed_step), 0, self.n_bed_bins - 1))
                        act = self.ACTIONS_GBQ[int(self.policy[k, b_idx, e_state, tv_state])]

                    if act > 0.0:
                        dt = act * self.profile.d_tumor_gy_per_gbq * max(0.5, noise_t[k])
                        dk = (
                            act
                            * self.profile.d_kidney_gy_per_gbq
                            * sink_multipliers[tv_state]
                            * egfr_renal_multipliers[e_state]
                            * max(0.5, noise_k[k])
                        )
                        cum_t_bed += SLSQPDoseOptimizer.compute_bed(dt, self.profile.t_eff_tumor_h, 1.5, 10.0)
                        cum_k_bed += SLSQPDoseOptimizer.compute_bed(dk, self.profile.t_eff_kidney_h, 2.8, 2.6)
                        cum_gbq += act

                        p_shrink = min(0.88, 0.18 + 0.018 * dt)
                        if rand_u_tumor[k] < p_shrink:
                            tv_state = min(3, tv_state + 1)
                        p_drop = 0.25 if noise_k[k] > 1.10 else 0.10
                        if rand_u_egfr[k] < p_drop:
                            e_state = min(2, e_state + 1)

                results[strat]["tumor_bed"].append(cum_t_bed)
                results[strat]["kidney_bed"].append(cum_k_bed)
                results[strat]["violated"].append(1.0 if cum_k_bed > self.max_bed_limit else 0.0)
                results[strat]["total_gbq"].append(cum_gbq)

        summary: Dict[str, Dict[str, float]] = {}
        for strat, metrics in results.items():
            summary[strat] = {
                "Mean_BED_Tumor_Gy10": round(float(np.mean(metrics["tumor_bed"])), 2),
                "SD_BED_Tumor": round(float(np.std(metrics["tumor_bed"])), 2),
                "Mean_BED_Ginjal_Gy2_6": round(float(np.mean(metrics["kidney_bed"])), 2),
                "P95_BED_Ginjal_Gy2_6": round(float(np.percentile(metrics["kidney_bed"], 95)), 2),
                "Prob_Toksisitas_Ginjal_pct": round(float(np.mean(metrics["violated"]) * 100.0), 1),
                "Mean_Total_Aktivitas_GBq": round(float(np.mean(metrics["total_gbq"])), 2),
            }
        return summary
