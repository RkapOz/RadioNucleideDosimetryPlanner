import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.signal import fftconvolve


class ClinicalDicomDosimetry:
    """
    Pipeline Tahap 3: Pembaca DICOM SPECT/PET & CT, Ekstraksi VOI/SUV,
    dan Konvolusi Voxel S-Value (VSV) 3D berbasis formalisme MIRD.
    """

    def __init__(self, target_voxel_mm: float = 3.0, spect_calib_factor_bq_per_cnt: Optional[float] = None):
        """
        Args:
            target_voxel_mm: Ukuran sisi voxel kubik (mm) yang harus cocok dengan kernel VSV (mis. 2.21, 3.0, 4.42 mm).
            spect_calib_factor_bq_per_cnt: Faktor kalibrasi kuantitatif SPECT (Bq/mL per unit piksel ter-rescale)
                                           jika tag DICOM Units bukan 'BQML'.
        """
        self.target_voxel_mm = target_voxel_mm
        self.spect_calib_factor = spect_calib_factor_bq_per_cnt

    @staticmethod
    def _parse_dicom_time(time_str: str) -> float:
        """Mengonversi format waktu DICOM (HHMMSS.ffffff) menjadi detik sejak tengah malam."""
        clean_str = time_str.split(".")[0].ljust(6, "0")
        dt = datetime.strptime(clean_str[:6], "%H%M%S")
        frac = float("0." + time_str.split(".")[1]) if "." in time_str else 0.0
        return dt.hour * 3600 + dt.minute * 60 + dt.second + frac

    def load_quantitative_emission_series(self, dicom_dir: str) -> Tuple[sitk.Image, Dict]:
        """
        Membaca seri DICOM PET atau SPECT kuantitatif, mempertahankan geometri 3D,
        dan mengonversi intensitas piksel menjadi konsentrasi aktivitas (Bq/mL).
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            raise FileNotFoundError(f"Tidak ditemukan seri DICOM valid di direktori: {dicom_dir}")

        reader.SetFileNames(dicom_names)
        sitk_img = reader.Execute()
        sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

        # Ekstraksi metadata klinis dari slice pertama
        ds = pydicom.dcmread(dicom_names[0], stop_before_pixels=True)
        modality = getattr(ds, "Modality", "UNKNOWN")
        units = getattr(ds, "Units", "CNTS")
        patient_weight_kg = float(getattr(ds, "PatientWeight", 70.0))

        # Ekstraksi informasi radiofarmaka
        rph_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", [None])[0]
        injected_dose_bq = float(getattr(rph_seq, "RadionuclideTotalDose", 0.0)) if rph_seq else 0.0
        half_life_sec = float(getattr(rph_seq, "RadionuclideHalfLife", 574300.8)) if rph_seq else 574300.8  # Default Lu-177

        # Hitung selisih waktu injeksi ke akuisisi untuk koreksi decay
        inj_time_str = getattr(rph_seq, "RadiopharmaceuticalStartTime", "") if rph_seq else ""
        acq_time_str = getattr(ds, "AcquisitionTime", getattr(ds, "SeriesTime", ""))
        decay_factor = 1.0
        if inj_time_str and acq_time_str:
            dt_sec = self._parse_dicom_time(acq_time_str) - self._parse_dicom_time(inj_time_str)
            if dt_sec > 0:
                lam = np.log(2.0) / half_life_sec
                decay_factor = np.exp(lam * dt_sec)

        # Konversi array gambar ke Bq/mL (SimpleITK otomatis menerapkan RescaleSlope & Intercept)
        arr_raw = sitk.GetArrayFromImage(sitk_img)  # Shape: (Z, Y, X)
        if units == "BQML":
            arr_bqml = arr_raw
        elif self.spect_calib_factor is not None:
            arr_bqml = arr_raw * self.spect_calib_factor
        else:
            raise ValueError(
                f"Units DICOM adalah '{units}' (bukan BQML). "
                "Masukkan `spect_calib_factor_bq_per_cnt` dari kalibrasi kamera SPECT."
            )

        bqml_img = sitk.GetImageFromArray(arr_bqml.astype(np.float32))
        bqml_img.CopyInformation(sitk_img)

        metadata = {
            "modality": modality,
            "units_original": units,
            "patient_weight_kg": patient_weight_kg,
            "injected_dose_bq": injected_dose_bq,
            "half_life_sec": half_life_sec,
            "decay_correction_to_inj": decay_factor,
            "spacing_mm": sitk_img.GetSpacing(),  # (dx, dy, dz)
        }
        return bqml_img, metadata

    def extract_voi_metrics(
        self,
        bqml_img: sitk.Image,
        voi_mask_img: sitk.Image,
        metadata: Dict,
        apply_pve_rc: bool = True,
    ) -> Dict[str, float]:
        """
        Menghitung SUV_mean, SUV_max, Volume VOI, Konsentrasi Aktivitas, dan Aktivitas Total VOI.
        Mendukung koreksi Partial Volume Effect (PVE) berbasis volume untuk lesi kecil.
        """
        # Pastikan mask VOI berada pada grid spasial yang identik dengan gambar emisi
        if voi_mask_img.GetSize() != bqml_img.GetSize() or voi_mask_img.GetSpacing() != bqml_img.GetSpacing():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(bqml_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            voi_mask_img = resampler.Execute(voi_mask_img)

        arr_bqml = sitk.GetArrayFromImage(bqml_img)
        arr_mask = sitk.GetArrayFromImage(voi_mask_img) > 0

        if not np.any(arr_mask):
            raise ValueError("Mask VOI kosong (tidak ada voxel bernilai > 0 di dalam grid gambar).")

        dx, dy, dz = bqml_img.GetSpacing()
        voxel_vol_ml = (dx * dy * dz) / 1000.0
        n_voxels = int(np.sum(arr_mask))
        voi_vol_ml = n_voxels * voxel_vol_ml

        voi_bqml = arr_bqml[arr_mask]
        mean_bqml = float(np.mean(voi_bqml))
        max_bqml = float(np.max(voi_bqml))

        # Recovery Coefficient (RC) empiris untuk koreksi PVE pada lesi kecil (< 15 mL)
        rc_factor = 1.0
        if apply_pve_rc and voi_vol_ml < 15.0:
            # Model kurva sigmoid/eksponensial kalibrasi SPECT standar
            rc_factor = float(1.0 - 0.65 * np.exp(-0.28 * voi_vol_ml))
            rc_factor = max(rc_factor, 0.25)

        corrected_mean_bqml = mean_bqml / rc_factor
        total_activity_mbq = (corrected_mean_bqml * voi_vol_ml) / 1e6

        # Hitung SUV (dengan koreksi peluruhan ke waktu injeksi)
        suv_mean, suv_max = 0.0, 0.0
        if metadata["injected_dose_bq"] > 0:
            weight_g = metadata["patient_weight_kg"] * 1000.0
            norm_factor = (metadata["injected_dose_bq"] / weight_g) / metadata["decay_correction_to_inj"]
            suv_mean = corrected_mean_bqml / norm_factor
            suv_max = max_bqml / norm_factor

        return {
            "volume_ml": round(voi_vol_ml, 3),
            "voxel_count": n_voxels,
            "recovery_coefficient": round(rc_factor, 3),
            "mean_activity_kbq_ml": round(corrected_mean_bqml / 1000.0, 3),
            "total_activity_mbq": round(total_activity_mbq, 4),
            "suv_mean": round(suv_mean, 2),
            "suv_max": round(suv_max, 2),
        }

    def resample_to_vsv_grid(self, bqml_img: sitk.Image) -> sitk.Image:
        """
        Melakukan resampling isotropik pada peta konsentrasi aktivitas (Bq/mL)
        agar ukuran voxel identik dengan resolusi kernel Voxel S-Value (mis. 3x3x3 mm^3).
        """
        orig_spacing = np.array(bqml_img.GetSpacing())
        orig_size = np.array(bqml_img.GetSize())
        new_spacing = np.array([self.target_voxel_mm] * 3, dtype=np.float64)
        new_size = np.ceil(orig_size * (orig_spacing / new_spacing)).astype(int).tolist()

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(bqml_img.GetDirection())
        resampler.SetOutputOrigin(bqml_img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        return resampler.Execute(bqml_img)

    @staticmethod
    def hu_to_density_g_cm3(ct_arr_hu: np.ndarray) -> np.ndarray:
        """
        Konversi nilai Hounsfield Unit (HU) CT menjadi densitas massa jaringan (g/cm^3)
        menggunakan pendekatan bilinear Schneider untuk koreksi heterogenitas dosis.
        """
        rho = np.ones_like(ct_arr_hu, dtype=np.float32) * 1.04
        # Paru & udara (-1000 s/d -100 HU)
        mask_low = ct_arr_hu <= -100
        rho[mask_low] = np.clip(1.0 + 0.001 * ct_arr_hu[mask_low], 0.0012, 0.90)
        # Jaringan lunak (-100 s/d +100 HU)
        mask_soft = (ct_arr_hu > -100) & (ct_arr_hu <= 100)
        rho[mask_soft] = 1.018 + 0.000893 * ct_arr_hu[mask_soft]
        # Tulang kortikal/trabekular (> +100 HU)
        mask_bone = ct_arr_hu > 100
        rho[mask_bone] = 1.04 + 0.00062 * ct_arr_hu[mask_bone]
        return rho

    def compute_3d_voxel_dose(
        self,
        resampled_bqml_img: sitk.Image,
        effective_half_life_hours: float,
        vsv_kernel_mgy_per_mbqs: np.ndarray,
        resampled_ct_img: Optional[sitk.Image] = None,
    ) -> sitk.Image:
        """
        Menghitung Peta Dosis Serap 3D (Gy) via Konvolusi FFT 3D antara peta
        Time-Integrated Activity (MBq*s) dan Kernel Voxel S-Value (mGy/(MBq*s)).
        """
        arr_bqml = sitk.GetArrayFromImage(resampled_bqml_img)
        voxel_vol_ml = (self.target_voxel_mm ** 3) / 1000.0

        # 1. Hitung Time-Integrated Activity (TIA) per voxel dalam MBq*s
        # Menggunakan model single-timepoint / effective clearance: A_tilde = A(t) * T_eff / ln(2)
        t_eff_sec = effective_half_life_hours * 3600.0
        voxel_activity_mbq = (arr_bqml * voxel_vol_ml) / 1e6
        tia_map_mbqs = voxel_activity_mbq * (t_eff_sec / np.log(2.0))

        # 2. Konvolusi 3D FFT dengan Kernel VSV (hasil awal dalam mGy)
        dose_map_mgy = fftconvolve(tia_map_mbqs, vsv_kernel_mgy_per_mbqs, mode="same")
        dose_map_mgy = np.maximum(dose_map_mgy, 0.0)  # Hilangkan artefak numerik negatif kecil

        # 3. Koreksi densitas jaringan berbasis CT (opsional tapi direkomendasikan untuk tulang/paru)
        if resampled_ct_img is not None:
            ct_arr = sitk.GetArrayFromImage(resampled_ct_img)
            rho_tissue = self.hu_to_density_g_cm3(ct_arr)
            rho_soft_tissue_ref = 1.04  # g/cm^3 (referensi kernel VSV jaringan lunak MIRD)
            dose_map_mgy = dose_map_mgy * (rho_soft_tissue_ref / rho_tissue)

        # Konversi mGy ke Gy
        dose_map_gy = (dose_map_mgy / 1000.0).astype(np.float32)

        dose_img = sitk.GetImageFromArray(dose_map_gy)
        dose_img.CopyInformation(resampled_bqml_img)
        return dose_img

    @staticmethod
    def compute_dvh_metrics(dose_img: sitk.Image, voi_mask_img: sitk.Image) -> Dict[str, float]:
        """Menghitung metrik Dose-Volume Histogram (DVH) klinis: D_mean, D_max, D90, D50, V20Gy."""
        if voi_mask_img.GetSize() != dose_img.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(dose_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            voi_mask_img = resampler.Execute(voi_mask_img)

        dose_arr = sitk.GetArrayFromImage(dose_img)
        mask_arr = sitk.GetArrayFromImage(voi_mask_img) > 0
        voi_doses = dose_arr[mask_arr]

        if voi_doses.size == 0:
            return {"D_mean_Gy": 0.0, "D_max_Gy": 0.0, "D90_Gy": 0.0, "D50_Gy": 0.0, "V20Gy_pct": 0.0}

        return {
            "D_mean_Gy": round(float(np.mean(voi_doses)), 2),
            "D_max_Gy": round(float(np.max(voi_doses)), 2),
            "D90_Gy": round(float(np.percentile(voi_doses, 10)), 2),  # Dosis diterima >= 90% volume
            "D50_Gy": round(float(np.percentile(voi_doses, 50)), 2),  # Median dose
            "V20Gy_pct": round(float(np.mean(voi_doses >= 20.0) * 100.0), 1),
        }
