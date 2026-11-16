import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from slsqp_dose_optimizer import PatientDosimetryProfile, SLSQPDoseOptimizer
from adaptive_mdp_planner import AdaptiveMDPPlanner


def render_stage4_adaptive_planning() -> None:
    st.header("Tahap 4: Optimasi Multi-Siklus (SLSQP) & Perencanaan Adaptif (MDP)")

    with st.sidebar:
        st.subheader("Parameter Dosimetri & Klinis Pasien")
        d_tumor = st.number_input("Koefisien Dosis Tumor (Gy/GBq)", 0.5, 15.0, 3.60, 0.10)
        d_kidney = st.number_input("Koefisien Dosis Ginjal (Gy/GBq)", 0.1, 3.0, 0.82, 0.02)
        d_marrow = st.number_input("Koefisien Dosis Sumsum (Gy/GBq)", 0.005, 0.20, 0.035, 0.005, format="%.3f")
        d_salivary = st.number_input("Koefisien Dosis Saliva (Gy/GBq)", 0.1, 2.5, 0.60, 0.05)

        baseline_egfr = st.slider("Baseline eGFR (mL/min/1.73m²)", 30, 120, 68)
        renal_risk = st.checkbox("Faktor Risiko Ginjal (Hipertensi / DM / Riwayat Cisplatin)", value=False)
        n_cycles = st.slider("Jumlah Siklus Rencana", 2, 6, 4)
        kappa_sink = st.slider("Intensitas Tumor Sink Effect (κ)", 0.0, 0.50, 0.25, 0.05)

    profile = PatientDosimetryProfile(
        d_tumor_gy_per_gbq=d_tumor,
        d_kidney_gy_per_gbq=d_kidney,
        d_marrow_gy_per_gbq=d_marrow,
        d_salivary_gy_per_gbq=d_salivary,
        baseline_egfr=float(baseline_egfr),
        has_renal_risk_factors=renal_risk,
    )

    tab_4a, tab_4b = st.tabs(["Tahap 4A: Optimasi SLSQP", "Tahap 4B: Kebijakan Adaptif MDP & Monte Carlo"])

    # --- TAB 4A: SLSQP ---
    with tab_4a:
        slsqp = SLSQPDoseOptimizer(profile=profile, n_cycles=n_cycles, kappa_sink=kappa_sink)
        opt_out = slsqp.optimize()
        std_out = slsqp.simulate_regimen([7.4] * n_cycles)
        nom = opt_out["nominal_metrics"]

        st.info(
            f"Batas Aman BED Ginjal Pasien: **{opt_out['kidney_bed_limit_gy2_6']} Gy₂.₆** | "
            f"Status Konvergensi SLSQP: **{'Optimal' if opt_out['converged'] else opt_out['message']}**"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Total BED Tumor (SLSQP)",
            f"{nom['tot_tumor_bed_gy10']} Gy₁₀",
            delta=f"{round(nom['tot_tumor_bed_gy10'] - std_out['tot_tumor_bed_gy10'], 2)} vs 7.4 GBq tetap",
        )
        m2.metric(
            "Total BED Ginjal (SLSQP)",
            f"{nom['tot_kidney_bed_gy2_6']} Gy₂.₆",
            delta=f"Batas {opt_out['kidney_bed_limit_gy2_6']} Gy₂.₆",
            delta_color="off",
        )
        m3.metric("Dosis Sumsum Tulang", f"{nom['tot_marrow_dose_gy']} Gy", delta="Batas 2.0 Gy", delta_color="off")
        m4.metric("Total Aktivitas Terapi", f"{nom['total_activity_gbq']} GBq")

        df_cycles = pd.DataFrame(nom["cycle_records"])
        st.subheader("Jadwal Aktivitas Injeksi Terpersonalisasi per Siklus")
        st.dataframe(df_cycles, use_container_width=True, hide_index=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=[f"Siklus {i}" for i in range(1, n_cycles + 1)],
                y=[7.4] * n_cycles,
                name="Protokol Tetap (7.4 GBq)",
                marker_color="#6C757D",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=[f"Siklus {i}" for i in range(1, n_cycles + 1)],
                y=opt_out["optimal_activities_gbq"],
                name="Optimasi SLSQP (Sink-Adapted)",
                marker_color="#08D9D6",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            title="Perbandingan Distribusi Aktivitas Injeksi Antar-Siklus (GBq)",
            yaxis_title="Aktivitas Injeksi (GBq)",
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 4B: MDP & MONTE CARLO ---
    with tab_4b:
        mdp = AdaptiveMDPPlanner(profile=profile, n_cycles=n_cycles, kappa_sink=kappa_sink)
        mdp.solve_policy()

        st.subheader("Asisten Keputusan Klinis Real-Time Antar-Siklus")
        c1, c2, c3, c4 = st.columns(4)
        cur_cycle = c1.number_input("Siklus Berikutnya", min_value=1, max_value=n_cycles, value=2)
        cum_k_bed = c2.number_input("Kumulatif BED Ginjal Terpakai (Gy₂.₆)", 0.0, 45.0, 9.5, 0.5)
        cur_egfr = c3.number_input("eGFR Terkini Pasca-Siklus", 20.0, 120.0, float(baseline_egfr - 4), 1.0)
        cur_tvol = c4.slider("Rasio Volume Tumor Tersisa (V / V₁)", 0.10, 1.00, 0.72, 0.05)

        rec = mdp.recommend_next_action(cur_cycle, cum_k_bed, cur_egfr, cur_tvol)
        st.success(
            f"**Rekomendasi Kebijakan MDP untuk Siklus {rec['siklus']}**: Injeksi **{rec['rekomendasi_aktivitas_gbq']} GBq** "
            f"(Sisa Kuota Ginjal: {rec['sisa_kuota_bed_ginjal']} Gy₂.₆ | Status: {rec['kategori_egfr']} | {rec['status_tumor']})"
        )

        st.subheader("Validasi Probabilistik Kohor Virtual (1.000 Simulasi Monte Carlo)")
        mc_summary = mdp.run_monte_carlo_cohort_simulation(n_simulations=1000)
        df_mc = pd.DataFrame.from_dict(mc_summary, orient="index").reset_index().rename(columns={"index": "Strategi Terapi"})
        st.dataframe(df_mc, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Radionuclide Dose Planner - Tahap 4", layout="wide")
    render_stage4_adaptive_planning()
