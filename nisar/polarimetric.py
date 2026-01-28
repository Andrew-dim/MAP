"""
Polarimetric SAR Processing Module
Quad-pol decomposition algorithms for NISAR data

Implements:
- Covariance/Coherency matrix computation
- H-α-A (Cloude-Pottier) decomposition
- Freeman-Durden 3-component decomposition
- Yamaguchi 4-component decomposition
- Pauli RGB decomposition
- Polarimetric indices
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class PolarimetricResult:
    """Container for polarimetric decomposition results."""
    # Backscatter intensities
    hh: np.ndarray
    vv: np.ndarray
    hv: np.ndarray
    vh: np.ndarray
    
    # Pauli components
    pauli_r: np.ndarray  # |HH - VV| (double bounce)
    pauli_g: np.ndarray  # |HV + VH| (volume)
    pauli_b: np.ndarray  # |HH + VV| (surface)
    
    # H-α-A decomposition
    entropy: np.ndarray      # H: 0-1 (randomness)
    anisotropy: np.ndarray   # A: 0-1 (eigenvalue spread)
    alpha: np.ndarray        # α: 0-90° (scattering mechanism)
    
    # Freeman-Durden components
    fd_surface: np.ndarray   # Ps: Surface scattering power
    fd_double: np.ndarray    # Pd: Double-bounce power
    fd_volume: np.ndarray    # Pv: Volume scattering power
    
    # Yamaguchi 4-component
    yam_surface: np.ndarray  # Surface
    yam_double: np.ndarray   # Double-bounce
    yam_volume: np.ndarray   # Volume
    yam_helix: np.ndarray    # Helix scattering
    
    # H-α classification (9 zones)
    h_alpha_class: np.ndarray
    
    # Additional indices
    span: np.ndarray         # Total power
    pedestal_height: np.ndarray
    
    # Metadata
    geotransform: tuple = None
    projection: str = None
    shape: tuple = None


class PolarimetricProcessor:
    """
    Process quad-pol SAR data for polarimetric analysis.
    
    Supports NISAR L2-GCOV (Geocoded Covariance) products.
    """
    
    # H-α classification zone boundaries
    ALPHA_BOUNDS = [42.5, 47.5, 50.0, 55.0, 60.0]  # degrees
    ENTROPY_BOUNDS = [0.5, 0.9]
    
    # Zone labels (Cloude-Pottier)
    ZONE_LABELS = {
        1: "High Entropy Surface",
        2: "High Entropy Vegetation",
        3: "High Entropy Multiple",
        4: "Medium Entropy Surface",
        5: "Medium Entropy Vegetation",
        6: "Medium Entropy Multiple",
        7: "Low Entropy Surface",
        8: "Low Entropy Dipole",
        9: "Low Entropy Multiple",
    }
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the polarimetric processor.
        
        Args:
            window_size: Window size for spatial averaging (must be odd)
        """
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.half_win = window_size // 2
    
    def process_nisar_gcov(self, h5_path: Path) -> PolarimetricResult:
        """
        Process NISAR L2-GCOV product.
        
        Args:
            h5_path: Path to NISAR GCOV HDF5 file
            
        Returns:
            PolarimetricResult with all decompositions
        """
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            # Navigate to covariance data
            # NISAR GCOV structure: /science/LSAR/GCOV/grids/frequencyA/
            
            freq_path = self._find_frequency_group(f)
            if not freq_path:
                raise ValueError("Could not find frequency group in NISAR file")
            
            gcov = f[freq_path]
            
            # Extract covariance matrix elements
            # C3 matrix: [[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]]
            # For quad-pol: HH, HV, VH, VV
            
            C11 = self._read_dataset(gcov, 'HHHH')  # |HH|²
            C22 = self._read_dataset(gcov, 'HVHV')  # |HV|²
            C33 = self._read_dataset(gcov, 'VVVV')  # |VV|²
            
            # Cross-pol covariances (complex)
            C12_real = self._read_dataset(gcov, 'HHHV_real', 'HHHVReal')
            C12_imag = self._read_dataset(gcov, 'HHHV_imag', 'HHHVImag')
            C13_real = self._read_dataset(gcov, 'HHVV_real', 'HHVVReal')
            C13_imag = self._read_dataset(gcov, 'HHVV_imag', 'HHVVImag')
            C23_real = self._read_dataset(gcov, 'HVVV_real', 'HVVVReal')
            C23_imag = self._read_dataset(gcov, 'HVVV_imag', 'HVVVImag')
            
            # Build complex covariance values
            C12 = C12_real + 1j * C12_imag if C12_real is not None else None
            C13 = C13_real + 1j * C13_imag if C13_real is not None else None
            C23 = C23_real + 1j * C23_imag if C23_real is not None else None
            
            # Get geolocation info
            geotransform, projection = self._get_geolocation(f, freq_path)
        
        # Process covariance matrix
        return self._process_covariance_matrix(
            C11, C22, C33, C12, C13, C23,
            geotransform, projection
        )
    
    def process_slc_quad_pol(
        self,
        hh: np.ndarray,
        hv: np.ndarray,
        vh: np.ndarray,
        vv: np.ndarray,
        geotransform: tuple = None,
        projection: str = None
    ) -> PolarimetricResult:
        """
        Process quad-pol SLC data directly.
        
        Args:
            hh, hv, vh, vv: Complex SLC arrays for each polarization
            geotransform: GDAL geotransform tuple
            projection: WKT projection string
            
        Returns:
            PolarimetricResult with all decompositions
        """
        # Compute covariance matrix from SLC
        # Apply multi-looking / spatial averaging
        
        # Intensities
        C11 = self._spatial_average(np.abs(hh) ** 2)
        C22 = self._spatial_average(np.abs(hv) ** 2)
        C33 = self._spatial_average(np.abs(vv) ** 2)
        
        # Cross products
        C12 = self._spatial_average(hh * np.conj(hv))
        C13 = self._spatial_average(hh * np.conj(vv))
        C23 = self._spatial_average(hv * np.conj(vv))
        
        return self._process_covariance_matrix(
            C11, C22, C33, C12, C13, C23,
            geotransform, projection
        )
    
    def _process_covariance_matrix(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray,
        C12: Optional[np.ndarray],
        C13: Optional[np.ndarray],
        C23: Optional[np.ndarray],
        geotransform: tuple = None,
        projection: str = None
    ) -> PolarimetricResult:
        """
        Process covariance matrix elements into all decompositions.
        """
        shape = C11.shape
        
        # Ensure we have all elements
        if C12 is None:
            C12 = np.zeros(shape, dtype=complex)
        if C13 is None:
            C13 = np.zeros(shape, dtype=complex)
        if C23 is None:
            C23 = np.zeros(shape, dtype=complex)
        
        # 1. Backscatter intensities (in dB)
        hh = self._to_db(C11)
        vv = self._to_db(C33)
        hv = self._to_db(C22)
        vh = self._to_db(C22)  # Reciprocity: HV ≈ VH
        
        # 2. Span (total power)
        span = C11 + 2 * C22 + C33
        
        # 3. Pauli decomposition
        pauli_r, pauli_g, pauli_b = self._pauli_decomposition(C11, C22, C33, C13)
        
        # 4. H-α-A decomposition
        entropy, anisotropy, alpha = self._h_alpha_decomposition(
            C11, C22, C33, C12, C13, C23
        )
        
        # 5. Freeman-Durden decomposition
        fd_surface, fd_double, fd_volume = self._freeman_durden(
            C11, C22, C33, C13
        )
        
        # 6. Yamaguchi 4-component
        yam_surface, yam_double, yam_volume, yam_helix = self._yamaguchi(
            C11, C22, C33, C12, C13, C23
        )
        
        # 7. H-α classification
        h_alpha_class = self._classify_h_alpha(entropy, alpha)
        
        # 8. Pedestal height
        pedestal = self._pedestal_height(C11, C22, C33)
        
        return PolarimetricResult(
            hh=hh, vv=vv, hv=hv, vh=vh,
            pauli_r=pauli_r, pauli_g=pauli_g, pauli_b=pauli_b,
            entropy=entropy, anisotropy=anisotropy, alpha=alpha,
            fd_surface=fd_surface, fd_double=fd_double, fd_volume=fd_volume,
            yam_surface=yam_surface, yam_double=yam_double,
            yam_volume=yam_volume, yam_helix=yam_helix,
            h_alpha_class=h_alpha_class,
            span=self._to_db(span),
            pedestal_height=pedestal,
            geotransform=geotransform,
            projection=projection,
            shape=shape
        )
    
    def _pauli_decomposition(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray,
        C13: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pauli decomposition for RGB visualization.
        
        R = |HH - VV|  (double bounce - urban, dihedrals)
        G = |HV + VH|  (volume - vegetation)
        B = |HH + VV|  (surface - water, smooth surfaces)
        """
        # From covariance matrix
        # |HH - VV|² = C11 + C33 - 2*Re(C13)
        # |HH + VV|² = C11 + C33 + 2*Re(C13)
        # |HV + VH|² ≈ 4*C22
        
        double_bounce = np.sqrt(np.maximum(C11 + C33 - 2 * np.real(C13), 0))
        volume = np.sqrt(4 * C22)
        surface = np.sqrt(np.maximum(C11 + C33 + 2 * np.real(C13), 0))
        
        # Normalize for visualization
        max_val = np.percentile(np.stack([double_bounce, volume, surface]), 99)
        
        pauli_r = np.clip(double_bounce / max_val, 0, 1)
        pauli_g = np.clip(volume / max_val, 0, 1)
        pauli_b = np.clip(surface / max_val, 0, 1)
        
        return pauli_r, pauli_g, pauli_b
    
    def _h_alpha_decomposition(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray,
        C12: np.ndarray,
        C13: np.ndarray,
        C23: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cloude-Pottier H-α-A decomposition.
        
        H (Entropy): Randomness of scattering (0=deterministic, 1=random)
        α (Alpha): Mean scattering mechanism (0°=surface, 45°=dipole, 90°=dihedral)
        A (Anisotropy): Relative importance of 2nd and 3rd eigenvalues
        """
        rows, cols = C11.shape
        entropy = np.zeros((rows, cols), dtype=np.float32)
        alpha = np.zeros((rows, cols), dtype=np.float32)
        anisotropy = np.zeros((rows, cols), dtype=np.float32)
        
        # Process pixel by pixel (can be optimized with vectorization)
        for i in range(rows):
            for j in range(cols):
                # Build coherency matrix T3
                T = np.array([
                    [C11[i,j], C12[i,j], C13[i,j]],
                    [np.conj(C12[i,j]), C22[i,j], C23[i,j]],
                    [np.conj(C13[i,j]), np.conj(C23[i,j]), C33[i,j]]
                ], dtype=complex)
                
                # Eigendecomposition
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(T)
                    
                    # Sort by descending eigenvalue
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = np.real(eigenvalues[idx])
                    eigenvectors = eigenvectors[:, idx]
                    
                    # Ensure positive eigenvalues
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    
                    # Normalize to probabilities
                    total = np.sum(eigenvalues)
                    p = eigenvalues / total if total > 0 else np.array([1/3, 1/3, 1/3])
                    
                    # Entropy
                    H = -np.sum(p * np.log2(p + 1e-10)) / np.log2(3)
                    entropy[i, j] = np.clip(H, 0, 1)
                    
                    # Alpha angles from eigenvectors
                    alphas = np.arccos(np.abs(eigenvectors[0, :])) * 180 / np.pi
                    alpha[i, j] = np.sum(p * alphas)
                    
                    # Anisotropy
                    if eigenvalues[1] + eigenvalues[2] > 0:
                        A = (eigenvalues[1] - eigenvalues[2]) / (eigenvalues[1] + eigenvalues[2])
                    else:
                        A = 0
                    anisotropy[i, j] = np.clip(A, 0, 1)
                    
                except np.linalg.LinAlgError:
                    entropy[i, j] = 0.5
                    alpha[i, j] = 45
                    anisotropy[i, j] = 0.5
        
        return entropy, anisotropy, alpha
    
    def _freeman_durden(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray,
        C13: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Freeman-Durden 3-component decomposition.
        
        Decomposes total power into:
        - Ps: Surface scattering (single bounce)
        - Pd: Double-bounce scattering (dihedrals)
        - Pv: Volume scattering (vegetation)
        """
        # Volume scattering first (from cross-pol)
        # Pv = 8/3 * C22 (for uniform distribution of thin cylinders)
        Pv = (8/3) * C22
        
        # Remaining covariance after volume removal
        C11_r = C11 - Pv / 2
        C33_r = C33 - Pv / 2
        C13_r = C13 - Pv / 6
        
        # Ensure non-negative
        C11_r = np.maximum(C11_r, 0)
        C33_r = np.maximum(C33_r, 0)
        
        # Determine dominant mechanism
        rho = C13_r / np.sqrt(C11_r * C33_r + 1e-10)
        
        # Surface scattering (Re(ρ) > 0)
        # Double-bounce (Re(ρ) < 0)
        
        Ps = np.zeros_like(C11)
        Pd = np.zeros_like(C11)
        
        # Surface dominant
        surface_mask = np.real(rho) >= 0
        beta = np.sqrt(C33_r / (C11_r + 1e-10))
        Ps[surface_mask] = C11_r[surface_mask] * (1 + beta[surface_mask]**2)
        Pd[surface_mask] = 0
        
        # Double-bounce dominant
        Pd[~surface_mask] = C33_r[~surface_mask] * 2
        Ps[~surface_mask] = 0
        
        # Normalize
        total = Ps + Pd + Pv
        Ps = np.where(total > 0, Ps / total, 0)
        Pd = np.where(total > 0, Pd / total, 0)
        Pv = np.where(total > 0, Pv / total, 0)
        
        return Ps, Pd, Pv
    
    def _yamaguchi(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray,
        C12: np.ndarray,
        C13: np.ndarray,
        C23: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Yamaguchi 4-component decomposition.
        
        Extends Freeman-Durden with helix scattering component:
        - Ps: Surface scattering
        - Pd: Double-bounce
        - Pv: Volume
        - Ph: Helix (asymmetric targets)
        """
        # Helix scattering from imaginary part of C23
        Ph = 2 * np.abs(np.imag(C23))
        
        # Remove helix contribution
        C22_r = C22 - Ph / 4
        C22_r = np.maximum(C22_r, 0)
        
        # Apply Freeman-Durden to remainder
        Ps, Pd, Pv = self._freeman_durden(C11, C22_r, C33, C13)
        
        # Normalize including helix
        total = Ps + Pd + Pv + Ph / (C11 + C33 + 2*C22 + 1e-10)
        
        # Scale helix to same range
        Ph_norm = Ph / (C11 + C33 + 2*C22 + 1e-10)
        
        return Ps, Pd, Pv, Ph_norm
    
    def _classify_h_alpha(
        self,
        entropy: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """
        Classify pixels into 9 H-α zones (Cloude-Pottier).
        
        Zones:
        1-3: High entropy (H > 0.9)
        4-6: Medium entropy (0.5 < H < 0.9)
        7-9: Low entropy (H < 0.5)
        
        Sub-zones by alpha angle.
        """
        classification = np.zeros_like(entropy, dtype=np.uint8)
        
        # High entropy
        high_H = entropy > 0.9
        classification[high_H & (alpha < 42.5)] = 1  # Surface
        classification[high_H & (alpha >= 42.5) & (alpha < 47.5)] = 2  # Vegetation
        classification[high_H & (alpha >= 47.5)] = 3  # Multiple
        
        # Medium entropy
        med_H = (entropy > 0.5) & (entropy <= 0.9)
        classification[med_H & (alpha < 40)] = 4  # Surface
        classification[med_H & (alpha >= 40) & (alpha < 55)] = 5  # Vegetation
        classification[med_H & (alpha >= 55)] = 6  # Multiple
        
        # Low entropy
        low_H = entropy <= 0.5
        classification[low_H & (alpha < 42.5)] = 7  # Surface (Bragg)
        classification[low_H & (alpha >= 42.5) & (alpha < 47.5)] = 8  # Dipole
        classification[low_H & (alpha >= 47.5)] = 9  # Dihedral
        
        return classification
    
    def _pedestal_height(
        self,
        C11: np.ndarray,
        C22: np.ndarray,
        C33: np.ndarray
    ) -> np.ndarray:
        """
        Compute pedestal height (polarimetric purity indicator).
        
        Low values indicate pure/deterministic scatterers.
        High values indicate depolarized/volume scattering.
        """
        min_copol = np.minimum(C11, C33)
        max_copol = np.maximum(C11, C33)
        
        pedestal = min_copol / (max_copol + 1e-10)
        return np.clip(pedestal, 0, 1)
    
    def _spatial_average(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial averaging (multi-looking)."""
        from scipy.ndimage import uniform_filter
        
        if np.iscomplexobj(data):
            real = uniform_filter(data.real, self.window_size)
            imag = uniform_filter(data.imag, self.window_size)
            return real + 1j * imag
        else:
            return uniform_filter(data.astype(float), self.window_size)
    
    def _to_db(self, linear: np.ndarray) -> np.ndarray:
        """Convert linear power to dB."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            db = 10 * np.log10(np.maximum(linear, 1e-10))
        return np.clip(db, -50, 50)
    
    def _find_frequency_group(self, f) -> Optional[str]:
        """Find the frequency group in NISAR HDF5 file."""
        possible_paths = [
            '/science/LSAR/GCOV/grids/frequencyA',
            '/science/LSAR/GCOV/grids/frequencyB',
            '/science/SSAR/GCOV/grids/frequencyA',
            '/GCOV/frequencyA',
            '/grids/frequencyA',
        ]
        
        for path in possible_paths:
            if path in f:
                return path
        
        # Search recursively
        def find_gcov(name, obj):
            if 'HHHH' in obj:
                return name
            return None
        
        result = None
        def visitor(name, obj):
            nonlocal result
            if hasattr(obj, 'keys') and 'HHHH' in obj.keys():
                result = name
        
        f.visititems(visitor)
        return result
    
    def _read_dataset(self, group, *names) -> Optional[np.ndarray]:
        """Read dataset by trying multiple possible names."""
        for name in names:
            if name in group:
                return np.array(group[name])
        return None
    
    def _get_geolocation(self, f, freq_path: str) -> Tuple[tuple, str]:
        """Extract geolocation information from NISAR file."""
        try:
            # Try to get from metadata
            geo_path = freq_path.rsplit('/', 1)[0] + '/metadata/geolocationGrid'
            
            if geo_path in f:
                geo = f[geo_path]
                # Extract geotransform parameters
                x_start = float(geo.get('xCoordinates', [[0]])[0][0])
                y_start = float(geo.get('yCoordinates', [[0]])[0][0])
                x_spacing = float(geo.get('xCoordinateSpacing', 30))
                y_spacing = float(geo.get('yCoordinateSpacing', -30))
                
                geotransform = (x_start, x_spacing, 0, y_start, 0, y_spacing)
                projection = 'EPSG:4326'  # Default to WGS84
                
                return geotransform, projection
        except:
            pass
        
        return None, None
    
    @staticmethod
    def get_zone_label(zone: int) -> str:
        """Get human-readable label for H-α zone."""
        return PolarimetricProcessor.ZONE_LABELS.get(zone, f"Unknown Zone {zone}")
