import os
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


class VSVKernelManager:
    """
    Generator dan Loader Kernel Voxel S-Value (VSV) 3D berbasis tabel koordinat
    radial MIRD Pamphlet No. 17 / OpenDose untuk jaringan lunak (rho = 1.04 g/cm^3).
    Satuan keluaran kernel: mGy / (MBq * s).
    """

    # Tabel referensi terurut (|i| >= |j| >= |k|) pada voxel kubik 3.0 mm (jaringan lunak)
    # Nilai dalam mGy / (MBq * s)
    MIRD_3MM_OCTANT_TABLES: Dict[str, Dict] = {
        "Lu-177": {
            "half_life_hours": 159.528,  # 6.647 hari
            "mean_beta_ic_mev": 0.1479,
            "photon_tail_coeff": 1.80e-5,
            "mu_soft_cm": 0.032,
            "octant_s_values": {
                (0, 0, 0): 0.755000,
                (1, 0, 0): 0.016800,
                (1, 1, 0): 0.001420,
                (1, 1, 1): 0.000215,
                (2, 0, 0): 0.000145,
                (2, 1, 0): 0.000068,
                (2, 1, 1): 0.000041,
                (2, 2, 0): 0.000025,
                (3, 0, 0): 0.000022,
            },
        },
        "Y-90": {
            "half_life_hours": 64.053,  # 2.669 hari
            "mean_beta_ic_mev": 0.9331,
            "photon_tail_coeff": 1.50e-6,  # Trace Bremsstrahlung
            "mu_soft_cm": 0.025,
            "octant_s_values": {
                (0, 0, 0): 1.650000,
                (1, 0, 0): 0.272000,
                (1, 1, 0): 0.081500,
                (1, 1, 1): 0.032500,
                (2, 0, 0): 0.027800,
                (2, 1, 0): 0.014400,
                (2, 1, 1): 0.007600,
                (2, 2, 0): 0.003400,
                (2, 2, 1): 0.002000,
                (3, 0, 0): 0.002200,
                (3, 1, 0): 0.001300,
                (3, 1, 1): 0.000750,
                (3, 2, 0): 0.000310,
                (4, 0, 0): 0.000100,
            },
        },
        "I-131": {
            "half_life_hours": 192.497,  # 8.021 hari
            "mean_beta_ic_mev": 0.1917,
            "photon_tail_coeff": 1.35e-4,  # Kontribusi gamma 364 keV
            "mu_soft_cm": 0.030,
            "octant_s_values": {
                (0, 0, 0): 0.902000,
                (1, 0, 0): 0.032500,
                (1, 1, 0): 0.003800,
                (1, 1, 1): 0.000940,
                (2, 0, 0): 0.000810,
                (2, 1, 0): 0.000420,
                (2, 1, 1): 0.000265,
                (2, 2, 0): 0.000175,
                (3, 0, 0): 0.000160,
            },
        },
    }

    @classmethod
    def generate_3d_kernel(
        cls,
        radionuclide: str = "Lu-177",
        voxel_mm: float = 3.0,
        radius_voxels: int = 5,
    ) -> np.ndarray:
        """
        Membangun matriks kernel VSV 3D berukuran (2*radius+1)^3 dari tabel radial MIRD.
        Jika voxel_mm != 3.0, kernel di-rescale secara volumetrik dengan konservasi energi.
        """
        if radionuclide not in cls.MIRD_3MM_OCTANT_TABLES:
            raise ValueError(
                f"Radionuklida '{radionuclide}' tidak tersedia. "
                f"Pilihan bawaan: {list(cls.MIRD_3MM_OCTANT_TABLES.keys())}"
            )

        spec = cls.MIRD_3MM_OCTANT_TABLES[radionuclide]
        octant_table = spec["octant_s_values"]
        photon_coeff = spec["photon_tail_coeff"]
        mu_cm = spec["mu_soft_cm"]

        # Rekonstruksi pada grid referensi 3.0 mm
        base_voxel_mm = 3.0
        size = 2 * radius_voxels + 1
        kernel_3mm = np.zeros((size, size, size), dtype=np.float64)
        center = radius_voxels

        for z in range(size):
            for y in range(size):
                for x in range(size):
                    # Simetri kubik penuh: urutkan |dz|, |dy|, |dx| secara menurun
                    key = tuple(sorted([abs(z - center), abs(y - center), abs(x - center)], reverse=True))
                    if key in octant_table:
                        kernel_3mm[z, y, x] = octant_table[key]
                    else:
                        r_cm = np.sqrt(key[0] ** 2 + key[1] ** 2 + key[2] ** 2) * (base_voxel_mm / 10.0)
                        if r_cm > 0 and photon_coeff > 0:
                            kernel_3mm[z, y, x] = (photon_coeff * np.exp(-mu_cm * r_cm)) / (r_cm ** 2)

        # Jika ukuran voxel target sama dengan 3.0 mm, langsung kembalikan
        if np.isclose(voxel_mm, base_voxel_mm, atol=1e-3):
            return kernel_3mm.astype(np.float32)

        # Rescaling geometris untuk ukuran voxel custom dengan menjaga kekekalan energi total
        total_energy_ref = np.sum(kernel_3mm) * (base_voxel_mm ** 3)
        scale_factor = base_voxel_mm / voxel_mm
        kernel_scaled = zoom(kernel_3mm, zoom=scale_factor, order=1)

        # Pastikan dimensi ganjil agar pusat kernel tetap berada tepat di tengah
        if kernel_scaled.shape[0] % 2 == 0:
            kernel_scaled = kernel_scaled[:-1, :-1, :-1]

        current_energy = np.sum(kernel_scaled) * (voxel_mm ** 3)
        if current_energy > 0:
            kernel_scaled *= total_energy_ref / current_energy

        return np.maximum(kernel_scaled, 0.0).astype(np.float32)

    @staticmethod
    def load_from_opendose_csv(csv_path: str, radius_voxels: int = 5) -> np.ndarray:
        """
        Memuat tabel VSV eksternal berformat CSV OpenDose dengan kolom: i, j, k, S_mGy_MBqs
        dan menyusunnya menjadi matriks kernel 3D simetris.
        """
        df = pd.read_csv(csv_path)
        required_cols = {"i", "j", "k", "S_mGy_MBqs"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV harus memiliki kolom: {required_cols}")

        lookup: Dict[Tuple[int, int, int], float] = {}
        for _, row in df.iterrows():
            key = tuple(sorted([abs(int(row["i"])), abs(int(row["j"])), abs(int(row["k"]))], reverse=True))
            lookup[key] = float(row["S_mGy_MBqs"])

        size = 2 * radius_voxels + 1
        kernel = np.zeros((size, size, size), dtype=np.float32)
        center = radius_voxels
        for z in range(size):
            for y in range(size):
                for x in range(size):
                    key = tuple(sorted([abs(z - center), abs(y - center), abs(x - center)], reverse=True))
                    kernel[z, y, x] = lookup.get(key, 0.0)
        return kernel

    @staticmethod
    def validate_kernel_physics(kernel: np.ndarray, voxel_mm: float = 3.0) -> Dict[str, float]:
        """
        Memverifikasi fisika kernel VSV: fraksi self-dose dan total energi terdeposit (MeV/decay).
        """
        center = kernel.shape[0] // 2
        s_self = float(kernel[center, center, center])
        s_total = float(np.sum(kernel))

        vol_cm3 = (voxel_mm / 10.0) ** 3
        mass_kg = (vol_cm3 * 1.04) / 1000.0
        # 1 MeV dalam 1 MBq*s -> mGy
        mev_to_mgy_per_mbqs = (1.602176634e-13 * 1e6 / mass_kg) * 1000.0
        effective_edep_mev = s_total / mev_to_mgy_per_mbqs

        return {
            "kernel_shape": f"{kernel.shape[0]}x{kernel.shape[1]}x{kernel.shape[2]}",
            "S_self_mGy_MBqs": round(s_self, 5),
            "S_total_mGy_MBqs": round(s_total, 5),
            "self_dose_fraction_pct": round((s_self / s_total) * 100.0, 2) if s_total > 0 else 0.0,
            "effective_deposited_energy_MeV": round(effective_edep_mev, 4),
        }
