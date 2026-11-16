import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import urllib.error

import numpy as np
import streamlit as st

from slsqp_dose_optimizer import PatientDosimetryProfile, SLSQPDoseOptimizer
from adaptive_mdp_planner import AdaptiveMDPPlanner


class TheranosticsMultiAgentOrchestrator:
    """
    Tahap 5: Orkestrasi Multi-Agent Klinis untuk Audit Keselamatan Radiobiologi
    dan Penyusunan Laporan Multidisciplinary Tumor Board (MTB).
    """

    def __init__(
        self,
        profile: PatientDosimetryProfile,
        profile_source: str,
        n_cycles: int = 4,
        kappa_sink: float = 0.25,
    ):
        self.profile = profile
        self.profile_source = profile_source
        self.n_cycles = n_cycles
        self.kappa_sink = kappa_sink

    def run_dosimetry_physicist_agent(self) -> Dict:
        """
        Agen 1 (Medical Physicist Agent): Mengonsolidasikan metrik dosimetri,
        jadwal SLSQP, kebijakan adaptif MDP, dan simulasi Monte Carlo.
        """
        slsqp = SLSQPDoseOptimizer(
            profile=self.profile,
            n_cycles=self.n_cycles,
            kappa_sink=self.kappa_sink,
        )
        opt_res = slsqp.optimize()
        std_res = slsqp.simulate_regimen(np.array([7.4] * self.n_cycles))

        mdp = AdaptiveMDPPlanner(
            profile=self.profile,
            n_cycles=self.n_cycles,
            kappa_sink=self.kappa_sink,
        )
        mdp.solve_policy()
        cycle1_rec = mdp.recommend_next_action(
            current_cycle_1based=1,
            cumulative_kidney_bed_so_far=0.0,
            current_egfr=self.profile.baseline_egfr,
            current_tumor_vol_fraction=1.0,
        )
        mc_summary = mdp.run_monte_carlo_cohort_simulation(n_simulations=1000)

        return {
            "source_pipeline": self.profile_source,
            "unit_doses_gy_per_gbq": {
                "tumor": self.profile.d_tumor_gy_per_gbq,
                "kidney": self.profile.d_kidney_gy_per_gbq,
                "red_marrow": self.profile.d_marrow_gy_per_gbq,
                "salivary_glands": self.profile.d_salivary_gy_per_gbq,
            },
            "effective_half_life_h": {
                "tumor": self.profile.t_eff_tumor_h,
                "kidney": self.profile.t_eff_kidney_h,
            },
            "slsqp_plan": opt_res,
            "standard_fixed_7_4gbq": std_res,
            "mdp_cycle1_action": cycle1_rec,
            "monte_carlo_validation": mc_summary,
        }

    def run_safety_auditor_agent(self, physics_packet: Dict) -> Dict:
        """
        Agen 2 (Radiation Safety & OAR Auditor Agent): Melakukan audit keras
        terhadap batas toleransi EANM/MIRD (Ginjal BED, Sumsum Tulang, Saliva, eGFR).
        """
        nom = physics_packet["slsqp_plan"]["nominal_metrics"]
        cons = physics_packet["slsqp_plan"]["conservative_metrics"]
        kidney_limit = physics_packet["slsqp_plan"]["kidney_bed_limit_gy2_6"]
        mc_mdp = physics_packet["monte_carlo_validation"]["Kebijakan Adaptif MDP (Tahap 4B)"]
        mc_std = physics_packet["monte_carlo_validation"]["Protokol Tetap (7.4 GBq)"]

        flags: List[Dict[str, str]] = []
        overall_status = "PASS (AMAN DENGAN MARGIN TERKONTROL)"

        # 1. Audit BED Ginjal (Nominal & Konservatif P90)
        if nom["tot_kidney_bed_gy2_6"] > kidney_limit:
            overall_status = "HOLD / REVISI DOSIS DIPERLUKAN"
            flags.append(
                {
                    "severity": "KRITIS",
                    "organ": "Ginjal",
                    "temuan": f"BED nominal ({nom['tot_kidney_bed_gy2_6']} Gy₂.₆) melampaui batas {kidney_limit} Gy₂.₆.",
                }
            )
        elif cons["tot_kidney_bed_gy2_6"] > kidney_limit:
            if overall_status.startswith("PASS"):
                overall_status = "PASS DENGAN PERINGATAN PEMANTAUAN"
            flags.append(
                {
                    "severity": "PERINGATAN",
                    "organ": "Ginjal",
                    "temuan": (
                        f"BED nominal aman ({nom['tot_kidney_bed_gy2_6']} Gy₂.₆), namun batas atas ketidakpastian "
                        f"mencapai {cons['tot_kidney_bed_gy2_6']} Gy₂.₆ (> {kidney_limit} Gy₂.₆). Wajib dosimetri ulang pasca-siklus 1."
                    ),
                }
            )
        else:
            flags.append(
                {
                    "severity": "AMAN",
                    "organ": "Ginjal",
                    "temuan": f"BED kumulatif terproyeksi {nom['tot_kidney_bed_gy2_6']} Gy₂.₆ (batas {kidney_limit} Gy₂.₆).",
                }
            )

        # 2. Audit Sumsum Tulang Merah (Batas klasik 2.0 Gy)
        if nom["tot_marrow_dose_gy"] > 2.0:
            overall_status = "HOLD / REVISI DOSIS DIPERLUKAN"
            flags.append(
                {
                    "severity": "KRITIS",
                    "organ": "Sumsum Tulang",
                    "temuan": f"Dosis kumulatif {nom['tot_marrow_dose_gy']} Gy melampaui batas mielotoksisitas 2.0 Gy.",
                }
            )
        elif nom["tot_marrow_dose_gy"] > 1.6:
            flags.append(
                {
                    "severity": "PERINGATAN",
                    "organ": "Sumsum Tulang",
                    "temuan": f"Dosis kumulatif {nom['tot_marrow_dose_gy']} Gy mendekati 2.0 Gy; pantau DPL & trombosit tiap 2 minggu.",
                }
            )
        else:
            flags.append(
                {
                    "severity": "AMAN",
                    "organ": "Sumsum Tulang",
                    "temuan": f"Dosis sumsum tulang {nom['tot_marrow_dose_gy']} Gy (aman di bawah 2.0 Gy).",
                }
            )

        # 3. Audit Kelenjar Saliva (Batas 35 Gy)
        if nom["tot_salivary_dose_gy"] > 35.0:
            flags.append(
                {
                    "severity": "PERINGATAN",
                    "organ": "Kelenjar Saliva",
                    "temuan": f"Proyeksi dosis {nom['tot_salivary_dose_gy']} Gy (> 35 Gy); risiko xerostomia derajat >= 2.",
                }
            )
        else:
            flags.append(
                {
                    "severity": "AMAN",
                    "organ": "Kelenjar Saliva",
                    "temuan": f"Proyeksi dosis {nom['tot_salivary_dose_gy']} Gy (di bawah ambang xerostomia 35 Gy).",
                }
            )

        # 4. Perbandingan Risiko Probabilistik Monte Carlo
        risk_reduction = round(
            mc_std["Prob_Toksisitas_Ginjal_pct"] - mc_mdp["Prob_Toksisitas_Ginjal_pct"], 1
        )
        flags.append(
            {
                "severity": "INFO",
                "organ": "Simulasi Monte Carlo (N=1.000)",
                "temuan": (
                    f"Probabilitas pelanggaran batas ginjal: {mc_mdp['Prob_Toksisitas_Ginjal_pct']}% (Kebijakan MDP) "
                    f"vs {mc_std['Prob_Toksisitas_Ginjal_pct']}% (Protokol tetap 7.4 GBq; reduksi risiko {risk_reduction}%)."
                ),
            }
        )

        return {
            "overall_status": overall_status,
            "kidney_bed_limit_gy2_6": kidney_limit,
            "audit_items": flags,
        }

    def _synthesize_deterministic_clinical_note(self, physics_packet: Dict, audit_packet: Dict) -> str:
        """
        Agen 3A (Rule-Based Nuclear Medicine Physician Agent): Menyusun laporan
        klinis formal tanpa ketergantungan API eksternal.
        """
        nom = physics_packet["slsqp_plan"]["nominal_metrics"]
        std = physics_packet["standard_fixed_7_4gbq"]
        mdp1 = physics_packet["mdp_cycle1_action"]
        acts = physics_packet["slsqp_plan"]["optimal_activities_gbq"]
        bed_gain = round(nom["tot_tumor_bed_gy10"] - std["tot_tumor_bed_gy10"], 2)

        schedule_lines = "\n".join(
            [
                f"  - Siklus {r['Siklus']}: **{r['Aktivitas (GBq)']} GBq** "
                f"(Tumor BED: {r['BED Tumor (Gy_10)']} Gy₁₀ | Ginjal BED: {r['BED Ginjal (Gy_2.6)']} Gy₂.₆ | "
                f"Sink Multiplier: {r['Sink Multiplier']}x)"
                for r in nom["cycle_records"]
            ]
        )

        audit_lines = "\n".join(
            [f"  - **[{item['severity']}] {item['organ']}**: {item['temuan']}" for item in audit_packet["audit_items"]]
        )

        return f"""### RINGKASAN KLINIS MULTIDISCIPLINARY TUMOR BOARD (MTB)
**Tanggal Evaluasi:** {datetime.now().strftime("%d %B %Y, %H:%M WIB")}  
**Sumber Kalibrasi Dosimetri:** {physics_packet['source_pipeline']}  
**Status Audit Keselamatan:** `{audit_packet['overall_status']}`

---

#### 1. Evaluasi Dosimetri Pasien-Spesifik & Fungsi Ginjal
- **Baseline Fungsi Ginjal:** eGFR {self.profile.baseline_egfr:.0f} mL/min/1.73m² (Batas aman kumulatif ginjal ditetapkan pada **{audit_packet['kidney_bed_limit_gy2_6']} Gy₂.₆** sesuai stratifikasi risiko EANM/MIRD).
- **Koefisien Dosis Terukur:** Tumor **{self.profile.d_tumor_gy_per_gbq:.2f} Gy/GBq** ($T_{{\\text{{eff}}}} = {self.profile.t_eff_tumor_h:.0f}$ jam), Ginjal **{self.profile.d_kidney_gy_per_gbq:.2f} Gy/GBq** ($T_{{\\text{{eff}}}} = {self.profile.t_eff_kidney_h:.0f}$ jam), Sumsum Tulang **{self.profile.d_marrow_gy_per_gbq:.3f} Gy/GBq**.

#### 2. Rekomendasi Regimen Terpersonalisasi (SLSQP & Kebijakan Adaptif MDP)
- **Jadwal Aktivitas Teroptimasi ({self.n_cycles} Siklus):** `{acts}` GBq (Total: **{nom['total_activity_gbq']} GBq**).
- **Rekomendasi Aksi Segera Siklus 1 (MDP):** Injeksi **{mdp1['rekomendasi_aktivitas_gbq']} GBq** dengan pemantauan SPECT/CT kuantitatif pasca-infus untuk memvalidasi *tumor sink effect* ($\\kappa = {self.kappa_sink}$).
- **Keuntungan Radiobiologis:** Proyeksi total $\\text{{BED}}_{{\\text{{tumor}}}}$ sebesar **{nom['tot_tumor_bed_gy10']} Gy₁₀** (selisih **{bed_gain:+.2f} Gy₁₀** dibandingkan protokol standar tetap $7.4\\text{{ GBq}}/\\text{{siklus}}$).

**Rincian Proyeksi per Siklus:**
{schedule_lines}

#### 3. Hasil Audit Keselamatan Organ Berisiko (OAR)
{audit_lines}

#### 4. Rencana Pemantauan Klinis & Laboratorium
1. Pemeriksaan darah perifer lengkap (Hb, Leukosit, Trombosit) serta fungsi ginjal (Ureum, Kreatinin, eGFR) pada **minggu ke-2 dan ke-4** pasca-setiap siklus.
2. Pembaruan status keadaan MDP (*state update*: sisa kuota BED ginjal, eGFR terkini, dan rasio volume tumor) sebelum pemberian siklus berikutnya.
"""

    def _synthesize_gemini_clinical_note(
         self,
         physics_packet: Dict,
         audit_packet: Dict,
         api_key: str,
         model_candidates: tuple = ("gemini-2.5-flash", "gemini-2.0-flash"),
     ) -> str:
         """
         Agen 3B (LLM Narrative Synthesizer via Gemini REST API):
         Menyusun narasi konsultasi dokter spesialis kedokteran nuklir dengan grounding ketat pada JSON fisika.
         """
         prompt = (
             "Anda adalah Dokter Spesialis Kedokteran Nuklir dan Teranostik Molekuler (Sp.KN-TM). "
             "Susun ringkasan klinis formal dalam Bahasa Indonesia untuk diskusi Multidisciplinary Tumor Board (MTB) "
             "berdasarkan data hasil perhitungan fisika dan audit keselamatan berikut. "
             "ATURAN KETAT: Gunakan HANYA angka yang tercantum dalam JSON di bawah ini, jangan mengarang angka baru.\n\n"
             f"DATA FISIKA & OPTIMASI:\n{json.dumps(physics_packet, indent=2)}\n\n"
             f"DATA AUDIT KESELAMATAN:\n{json.dumps(audit_packet, indent=2)}"
         )
    
         payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
         last_err = None
    
         for model_name in model_candidates:
             url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
             req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
             try:
                 with urllib.request.urlopen(req, timeout=25) as resp:
                     data = json.loads(resp.read().decode("utf-8"))
                     return data["candidates"][0]["content"]["parts"][0]["text"]
             except urllib.error.HTTPError as http_exc:
                 err_body = http_exc.read().decode("utf-8", errors="ignore")
                 last_err = f"HTTP {http_exc.code} ({model_name}): {err_body[:180]}"
             except Exception as exc:
                 last_err = str(exc)
    
         raise RuntimeError(last_err or "Gagal menghubungi Gemini API")
    def generate_full_report(self, gemini_api_key: Optional[str] = None) -> Dict:
        physics_packet = self.run_dosimetry_physicist_agent()
        audit_packet = self.run_safety_auditor_agent(physics_packet)

        narrative_mode = "Rule-Based Clinical Engine (Offline / Zero-Cost)"
        if gemini_api_key and gemini_api_key.strip():
            try:
                clinical_markdown = self._synthesize_gemini_clinical_note(
                    physics_packet, audit_packet, api_key=gemini_api_key.strip()
                )
                narrative_mode = "Gemini Clinical Narrative Agent (Grounded)"
            except Exception as exc:
                clinical_markdown = (
                    f"> *Catatan: Koneksi Gemini API gagal ({exc}), beralih otomatis ke generator klinis offline.*\n\n"
                    + self._synthesize_deterministic_clinical_note(physics_packet, audit_packet)
                )
        else:
            clinical_markdown = self._synthesize_deterministic_clinical_note(physics_packet, audit_packet)

        return {
            "narrative_mode": narrative_mode,
            "physics_packet": physics_packet,
            "audit_packet": audit_packet,
            "clinical_markdown": clinical_markdown,
        }


def render_tab_stage5() -> None:
    """Komponen antarmuka Streamlit untuk Tahap 5."""
    st.header("Tahap 5: Multi-Agent Clinical Decision Support & Laporan Tumor Board")
    prof: PatientDosimetryProfile = st.session_state["patient_profile"]
    source: str = st.session_state.get("profile_source", "Default Klinis")

    c1, c2, c3 = st.columns([1, 1, 2])
    n_cycles = c1.number_input("Jumlah Siklus Evaluasi", 2, 6, 4, key="s5_cycles")
    kappa_sink = c2.slider("Tumor Sink Factor (κ)", 0.0, 0.50, 0.25, 0.05, key="s5_sink")
    env_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = c3.text_input(
        "Gemini API Key (Opsional — kosongkan untuk mode 100% offline gratis)",
        value=env_key,
        type="password",
    )

    if st.button("Jalankan Orkestrasi Multi-Agent & Buat Laporan MTB", type="primary"):
        orchestrator = TheranosticsMultiAgentOrchestrator(
            profile=prof,
            profile_source=source,
            n_cycles=int(n_cycles),
            kappa_sink=float(kappa_sink),
        )
        with st.spinner("Menjalankan Agen Fisikawan Medis, Agen Auditor OAR, dan Agen Sintesis Klinis..."):
            report_bundle = orchestrator.generate_full_report(gemini_api_key=api_key_input)
        st.session_state["stage5_report"] = report_bundle

    if "stage5_report" not in st.session_state:
        st.info("Klik tombol di atas untuk mengaudit profil pasien aktif dan menghasilkan dokumen laporan Tumor Board.")
        return

    rep = st.session_state["stage5_report"]
    audit = rep["audit_packet"]

    # Tampilkan status audit keselamatan
    status_str = audit["overall_status"]
    if status_str.startswith("HOLD"):
        st.error(f"**Status Audit Keselamatan:** {status_str}")
    elif "PERINGATAN" in status_str:
        st.warning(f"**Status Audit Keselamatan:** {status_str}")
    else:
        st.success(f"**Status Audit Keselamatan:** {status_str}")

    st.caption(f"Mesin Sintesis Narasi: **{rep['narrative_mode']}**")

    # Tampilkan Laporan Klinis Siap-Cetak
    st.markdown(rep["clinical_markdown"])

    # Tombol Unduh Laporan (Markdown & JSON Audit Terstruktur)
    d1, d2 = st.columns(2)
    d1.download_button(
        label="⬇️ Unduh Laporan Klinis Tumor Board (.md)",
        data=rep["clinical_markdown"],
        file_name=f"MTB_Dosimetry_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )
    d2.download_button(
        label="⬇️ Unduh Paket Audit Fisika & MDP (.json)",
        data=json.dumps(
            {"physics": rep["physics_packet"], "audit": rep["audit_packet"]},
            indent=2,
            default=str,
        ),
        file_name=f"Dosimetry_Audit_Packet_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )
