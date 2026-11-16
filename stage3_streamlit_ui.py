from typing import Dict, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dicom_vsv_dosimetry import ClinicalDicomDosimetry
from vsv_kernel_manager import VSVKernelManager


def _create_calibrated_clinical_phantom(
    voxel_mm: float = 3.0,
    tumor_activity_kbq_ml: float = 180.0,
    kidney_activity_kbq_ml: float = 42.0,
    liver_activity_kbq_ml: float = 18.0,
) -> Tuple[sitk.Image, sitk.Image, Dict[str, sitk.Image], Dict]:
    """
    Membuat phantom anatomis 3D (64x64x48) untuk simulasi & verifikasi cepat
    ketika folder DICOM pasien eksternal belum dimuat.
    """
    nz, ny, nx = 48, 64, 64
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]

    # Tubuh dasar (jaringan lunak ~ +35 HU) dan tulang belakang (+650 HU)
    body_mask = ((yy - 32) / 22.0) ** 2 + ((xx - 32) / 26.0) ** 2 <= 1.0
    spine_mask = ((yy - 46) / 3.5) ** 2 + ((xx - 32) / 3.5) ** 2 <= 1.0

    ct_arr = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    ct_arr[body_mask] = 35.0
    ct_arr[spine_mask] = 650.0

    # Struktur VOI: Ginjal Kiri, Ginjal Kanan, Hepar, dan Lesi Tumor
    kidney_l = (((zz - 22) / 7.0) ** 2 + ((yy - 38) / 4.5) ** 2 + ((xx - 43) / 4.0) ** 2) <= 1.0
    kidney_r = (((zz - 22) / 7.0) ** 2 + ((yy - 38) / 4.5) ** 2 + ((xx - 21) / 4.0) ** 2) <= 1.0
    liver = (((zz - 28) / 8.0) ** 2 + ((yy - 26) / 7.5) ** 2 + ((xx - 20) / 8.5) ** 2) <= 1.0
    tumor = (((zz - 25) / 3.8) ** 2 + ((yy - 24) / 3.8) ** 2 + ((xx - 38) / 3.8) ** 2) <= 1.0

    # Distribusi konsentrasi aktivitas (Bq/mL) dengan variasi heterogenitas realistis
    rng = np.random.default_rng(42)
    bqml_arr = np.zeros((nz, ny, nx), dtype=np.float32)
    bqml_arr[body_mask] = 3500.0 * rng.uniform(0.85, 1.15, size=np.sum(body_mask))
    bqml_arr[liver] = (liver_activity_kbq_ml * 1000.0) * rng.uniform(0.90, 1.10, size=np.sum(liver))
    bqml_arr[kidney_l] = (kidney_activity_kbq_ml * 1000.0) * rng.uniform(0.85, 1.15, size=np.sum(kidney_l))
    bqml_arr[kidney_r] = (kidney_activity_kbq_ml * 1000.0) * rng.uniform(0.85, 1.15, size=np.sum(kidney_r))
    # Tumor memiliki inti nekrotik ringan & rim hipermetabolik
    bqml_arr[tumor] = (tumor_activity_kbq_ml * 1000.0) * rng.uniform(0.80, 1.25, size=np.sum(tumor))

    def to_sitk(arr: np.ndarray, is_mask: bool = False) -> sitk.Image:
        img = sitk.GetImageFromArray(arr.astype(np.uint8 if is_mask else np.float32))
        img.SetSpacing((voxel_mm, voxel_mm, voxel_mm))
        return img

    vois = {
        "Tumor Target": to_sitk(tumor, is_mask=True),
        "Ginjal Kiri (OAR)": to_sitk(kidney_l, is_mask=True),
        "Ginjal Kanan (OAR)": to_sitk(kidney_r, is_mask=True),
        "Hepar (OAR)": to_sitk(liver, is_mask=True),
    }
    metadata = {
        "modality": "SPECT/CT (Phantom Terkalibrasi)",
        "units_original": "BQML",
        "patient_weight_kg": 68.0,
        "injected_dose_bq": 7.4e9,  # 7.4 GBq (200 mCi)
        "half_life_sec": 574300.8,
        "decay_correction_to_inj": 1.11,
        "spacing_mm": (voxel_mm, voxel_mm, voxel_mm),
    }
    return to_sitk(bqml_arr), to_sitk(ct_arr), vois, metadata


def _plot_mpr_slices(
    bqml_arr: np.ndarray,
    dose_arr: np.ndarray,
    voi_masks: Dict[str, np.ndarray],
    z_idx: int,
    y_idx: int,
    x_idx: int,
) -> go.Figure:
    """
    Visualisasi Multi-Planar Reconstruction (Axial, Coronal, Sagittal)
    membandingkan Peta Aktivitas (kBq/mL) dan Peta Dosis Serap 3D (Gy) dengan kontur VOI.
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            f"Axial Aktivitas (Z={z_idx})",
            f"Coronal Aktivitas (Y={y_idx})",
            f"Sagittal Aktivitas (X={x_idx})",
            f"Axial Dosis Gy (Z={z_idx})",
            f"Coronal Dosis Gy (Y={y_idx})",
            f"Sagittal Dosis Gy (X={x_idx})",
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    act_kbq = bqml_arr / 1000.0
    slices_act = [act_kbq[z_idx, :, :], act_kbq[:, y_idx, :], act_kbq[:, :, x_idx]]
    slices_dose = [dose_arr[z_idx, :, :], dose_arr[:, y_idx, :], dose_arr[:, :, x_idx]]

    voi_colors = {
        "Tumor Target": "#FF2E63",
        "Ginjal Kiri (OAR)": "#08D9D6",
        "Ginjal Kanan (OAR)": "#00ADB5",
        "Hepar (OAR)": "#F9ED69",
    }

    for col_i in range(3):
        # Baris 1: Peta Aktivitas
        fig.add_trace(
            go.Heatmap(
                z=slices_act[col_i],
                colorscale="Viridis",
                showscale=(col_i == 2),
                colorbar=dict(title="kBq/mL", y=0.80, len=0.42),
            ),
            row=1,
            col=col_i + 1,
        )
        # Baris 2: Peta Dosis Serap (Gy)
        fig.add_trace(
            go.Heatmap(
                z=slices_dose[col_i],
                colorscale="Inferno",
                showscale=(col_i == 2),
                colorbar=dict(title="Dosis (Gy)", y=0.22, len=0.42),
            ),
            row=2,
            col=col_i + 1,
        )

        # Overlay Kontur VOI pada Peta Dosis
        for voi_name, mask_3d in voi_masks.items():
            mask_2d = (
                mask_3d[z_idx, :, :]
                if col_i == 0
                else (mask_3d[:, y_idx, :] if col_i == 1 else mask_3d[:, :, x_idx])
            )
            if np.any(mask_2d):
                fig.add_trace(
                    go.Contour(
                        z=mask_2d.astype(float),
                        contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                        line=dict(color=voi_colors.get(voi_name, "#FFFFFF"), width=2),
                        showscale=False,
                        name=voi_name,
                        showlegend=(col_i == 0),
                    ),
                    row=2,
                    col=col_i + 1,
                )

    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        height=640,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
    )
    return fig


def _plot_cumulative_dvh(dose_arr: np.ndarray, voi_masks: Dict[str, np.ndarray]) -> go.Figure:
    """Membuat kurva Cumulative Dose-Volume Histogram (cDVH) klinis."""
    fig = go.Figure()
    palette = ["#FF2E63", "#08D9D6", "#00ADB5", "#F38181", "#AA96DA"]

    max_dose = max(float(np.max(dose_arr)), 30.0)
    dose_bins = np.linspace(0.0, max_dose * 1.05, 250)

    for idx, (voi_name, mask_3d) in enumerate(voi_masks.items()):
        voxels = dose_arr[mask_3d > 0]
        if voxels.size == 0:
            continue
        # Fraksi volume (%) yang menerima dosis >= D
        vol_pct = [float(np.mean(voxels >= d) * 100.0) for d in dose_bins]
        fig.add_trace(
            go.Scatter(
                x=dose_bins,
                y=vol_pct,
                mode="lines",
                name=voi_name,
                line=dict(width=3, color=palette[idx % len(palette)]),
            )
        )

    # Garis referensi batas toleransi klinis OAR (Ginjal ~23 Gy) & Target Tumor (~80 Gy)
    fig.add_vline(
        x=23.0,
        line_dash="dash",
        line_color="#08D9D6",
        annotation_text="Batas Ginjal (23 Gy)",
        annotation_position="top right",
    )
    if max_dose >= 60.0:
        fig.add_vline(
            x=80.0,
            line_dash="dot",
            line_color="#FF2E63",
            annotation_text="Target Tumor (80 Gy)",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Cumulative Dose-Volume Histogram (cDVH)",
        xaxis_title="Dosis Serap (Gy)",
        yaxis_title="Volume Struktur (%)",
        yaxis=dict(range=[0, 105]),
        height=420,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_stage3_clinical_dosimetry() -> None:
    """Komponen utama Streamlit untuk Tahap 3: Integrasi DICOM & Konvolusi VSV 3D."""
    st.header("Tahap 3: Dosimetri Voxel 3D & Integrasi Data Klinis (SPECT/PET-CT)")

    with st.sidebar:
        st.subheader("Konfigurasi Fisika & Kernel VSV")
        radionuclide = st.selectbox("Radionuklida Terapi", ["Lu-177", "Y-90", "I-131"], index=0)
        voxel_mm = st.select_slider("Resolusi Grid VSV (mm)", options=[2.21, 3.0, 4.42], value=3.0)
        kernel_radius = st.slider("Radius Kernel VSV (voxel)", min_value=3, max_value=7, value=5)
        t_eff_hours = st.number_input(
            "Waktu Paruh Efektif T_eff (jam)",
            min_value=5.0,
            max_value=240.0,
            value=68.0 if radionuclide == "Lu-177" else (48.0 if radionuclide == "Y-90" else 85.0),
            step=1.0,
        )
        apply_pve = st.checkbox("Aktifkan Koreksi Partial Volume Effect (PVE)", value=True)
        apply_ct_density = st.checkbox("Aktifkan Koreksi Densitas Jaringan CT (HU)", value=True)

    # Bangun & verifikasi kernel VSV 3D
    kernel_3d = VSVKernelManager.generate_3d_kernel(
        radionuclide=radionuclide,
        voxel_mm=voxel_mm,
        radius_voxels=kernel_radius,
    )
    kernel_stats = VSVKernelManager.validate_kernel_physics(kernel_3d, voxel_mm=voxel_mm)

    with st.expander(f"Validasi Fisika Kernel VSV 3D ({radionuclide} @ {voxel_mm} mm)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dimensi Kernel", kernel_stats["kernel_shape"])
        c2.metric("S(0,0,0) Self-Dose", f"{kernel_stats['S_self_mGy_MBqs']} mGy/MBq·s")
        c3.metric("Fraksi Self-Dose", f"{kernel_stats['self_dose_fraction_pct']}%")
        c4.metric("Energi Efektif Total", f"{kernel_stats['effective_deposited_energy_MeV']} MeV")

    # Pilihan sumber data: DICOM Nyata vs Phantom Terkalibrasi
    data_mode = st.radio(
        "Sumber Data Pencitraan",
        ["Phantom Klinis Terkalibrasi (Demo Cepat)", "Direktori Seri DICOM Lokal (SPECT/PET + CT + Mask NIfTI)"],
        horizontal=True,
    )

    engine = ClinicalDicomDosimetry(target_voxel_mm=voxel_mm)

    if data_mode == "Phantom Klinis Terkalibrasi (Demo Cepat)":
        col_p1, col_p2, col_p3 = st.columns(3)
        tumor_act = col_p1.slider("Konsentrasi Aktivitas Tumor (kBq/mL)", 50.0, 400.0, 185.0, 5.0)
        kidney_act = col_p2.slider("Konsentrasi Aktivitas Ginjal (kBq/mL)", 10.0, 120.0, 42.0, 2.0)
        liver_act = col_p3.slider("Konsentrasi Aktivitas Hepar (kBq/mL)", 5.0, 60.0, 18.0, 1.0)

        bqml_img, ct_img, voi_dict, metadata = _create_calibrated_clinical_phantom(
            voxel_mm=voxel_mm,
            tumor_activity_kbq_ml=tumor_act,
            kidney_activity_kbq_ml=kidney_act,
            liver_activity_kbq_ml=liver_act,
        )
    else:
        col_d1, col_d2, col_d3 = st.columns(3)
        emission_dir = col_d1.text_input("Path Folder DICOM SPECT/PET", value="./data/patient01/SPECT")
        ct_dir = col_d2.text_input("Path Folder DICOM CT (Opsional)", value="./data/patient01/CT")
        mask_path = col_d3.text_input("Path Mask VOI (.nii.gz)", value="./data/patient01/tumor_mask.nii.gz")
        calib_factor = st.number_input(
            "Faktor Kalibrasi SPECT (Bq/mL per count, isi 0 jika Units=BQML)",
            min_value=0.0,
            value=0.0,
            format="%.4f",
        )
        engine.spect_calib_factor = calib_factor if calib_factor > 0 else None

        if not st.button("Proses & Konvolusi Data DICOM", type="primary"):
            st.info("Masukkan path direktori DICOM pasien lalu klik tombol di atas untuk menjalankan perhitungan.")
            return

        bqml_img, metadata = engine.load_quantitative_emission_series(emission_dir)
        ct_img = None
        if ct_dir and apply_ct_density:
            reader = sitk.ImageSeriesReader()
            ct_files = reader.GetGDCMSeriesFileNames(ct_dir)
            if ct_files:
                reader.SetFileNames(ct_files)
                ct_raw = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(bqml_img)
                resampler.SetInterpolator(sitk.sitkLinear)
                ct_img = resampler.Execute(ct_raw)

        voi_mask = sitk.ReadImage(mask_path)
        voi_dict = {"VOI Utama": voi_mask}

    # Eksekusi Resampling & Konvolusi VSV 3D
    resampled_bqml = engine.resample_to_vsv_grid(bqml_img)
    resampled_ct = None
    if apply_ct_density and ct_img is not None:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(resampled_bqml)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled_ct = resampler.Execute(ct_img)

    dose_img = engine.compute_3d_voxel_dose(
        resampled_bqml_img=resampled_bqml,
        effective_half_life_hours=t_eff_hours,
        vsv_kernel_mgy_per_mbqs=kernel_3d,
        resampled_ct_img=resampled_ct,
    )

    # Hitung metrik VOI & DVH untuk seluruh struktur
    summary_rows = []
    aligned_masks_np: Dict[str, np.ndarray] = {}

    for voi_name, mask_img in voi_dict.items():
        voi_metrics = engine.extract_voi_metrics(bqml_img, mask_img, metadata, apply_pve_rc=apply_pve)
        dvh_metrics = engine.compute_dvh_metrics(dose_img, mask_img)

        # Selaraskan mask dengan grid dosis untuk overlay visualisasi
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(dose_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        aligned_mask = resampler.Execute(mask_img)
        aligned_masks_np[voi_name] = sitk.GetArrayFromImage(aligned_mask) > 0

        summary_rows.append(
            {
                "Struktur / VOI": voi_name,
                "Volume (mL)": voi_metrics["volume_ml"],
                "RC (PVE)": voi_metrics["recovery_coefficient"],
                "SUV_mean": voi_metrics["suv_mean"],
                "SUV_max": voi_metrics["suv_max"],
                "Aktivitas (MBq)": voi_metrics["total_activity_mbq"],
                "D_mean (Gy)": dvh_metrics["D_mean_Gy"],
                "D_max (Gy)": dvh_metrics["D_max_Gy"],
                "D90 (Gy)": dvh_metrics["D90_Gy"],
                "V20Gy (%)": dvh_metrics["V20Gy_pct"],
            }
        )

    st.subheader("Ringkasan Kuantitatif VOI, SUV, & Metrik Dosimetri DVH")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Visualisasi Irisan 3D MPR & Kurva cDVH
    bqml_arr = sitk.GetArrayFromImage(resampled_bqml)
    dose_arr = sitk.GetArrayFromImage(dose_img)
    nz, ny, nx = dose_arr.shape

    st.subheader("Visualisasi Irisan Multi-Planar (MPR) & Peta Dosis 3D")
    s1, s2, s3 = st.columns(3)
    z_idx = s1.slider("Irisan Axial (Z)", 0, nz - 1, nz // 2)
    y_idx = s2.slider("Irisan Coronal (Y)", 0, ny - 1, ny // 2)
    x_idx = s3.slider("Irisan Sagittal (X)", 0, nx - 1, nx // 2)

    fig_mpr = _plot_mpr_slices(bqml_arr, dose_arr, aligned_masks_np, z_idx, y_idx, x_idx)
    st.plotly_chart(fig_mpr, use_container_width=True)

    st.subheader("Analisis Dose-Volume Histogram (DVH)")
    fig_dvh = _plot_cumulative_dvh(dose_arr, aligned_masks_np)
    st.plotly_chart(fig_dvh, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Radionuclide Dose Planner - Tahap 3", layout="wide")
    render_stage3_clinical_dosimetry()
