"""
SAR Calibration Module
Radiometric calibration for Sentinel-1 GRD/SLC products

Provides precise calibration using annotation XML files from .SAFE products.
Falls back to simplified calibration if annotation files are unavailable.

Implements:
- Sigma0 (σ°) calibration - radar backscatter coefficient
- Gamma0 (γ°) calibration - terrain-flattened
- Beta0 (β°) calibration - radar brightness
- Noise correction (NESZ subtraction)
- Incidence angle normalization
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import xml.etree.ElementTree as ET
import warnings
from dataclasses import dataclass
import re


@dataclass
class CalibrationLUT:
    """Calibration Look-Up Table from Sentinel-1 annotation."""
    azimuth_time: List[str]
    line: np.ndarray
    pixel: np.ndarray
    sigma_nought: np.ndarray
    beta_nought: np.ndarray
    gamma: np.ndarray
    dn: np.ndarray


@dataclass
class NoiseLUT:
    """Noise Look-Up Table (NESZ) from Sentinel-1 annotation."""
    azimuth_time: List[str]
    line: np.ndarray
    pixel: np.ndarray
    noise_lut: np.ndarray


class SARCalibrator:
    """
    Radiometric calibration for Sentinel-1 SAR data.
    
    Supports both precise (XML-based) and simplified calibration methods.
    """
    
    # Default calibration constants (for simplified method)
    DEFAULT_CALIBRATION = {
        'IW': {
            'VV': -17.0,  # dB offset for VV
            'VH': -24.0,  # dB offset for VH
            'HH': -17.0,
            'HV': -24.0,
        },
        'EW': {
            'VV': -18.0,
            'VH': -25.0,
            'HH': -18.0,
            'HV': -25.0,
        },
        'SM': {
            'VV': -16.0,
            'VH': -23.0,
            'HH': -16.0,
            'HV': -23.0,
        }
    }
    
    def __init__(self, safe_path: Path = None):
        """
        Initialize SAR calibrator.
        
        Args:
            safe_path: Path to .SAFE product directory (optional)
        """
        self.safe_path = Path(safe_path) if safe_path else None
        self.calibration_lut: Optional[CalibrationLUT] = None
        self.noise_lut: Optional[NoiseLUT] = None
        self.mode = None
        self.polarization = None
        
        if self.safe_path:
            self._parse_product_name()
    
    def _parse_product_name(self):
        """Parse product metadata from SAFE directory name."""
        name = self.safe_path.name
        
        # Extract mode (IW, EW, SM, WV)
        mode_match = re.search(r'S1[AB]_(IW|EW|SM|WV)', name)
        if mode_match:
            self.mode = mode_match.group(1)
        
        # Extract polarization
        pol_match = re.search(r'_(SH|SV|DH|DV)_', name)
        if pol_match:
            pol = pol_match.group(1)
            if pol == 'SV':
                self.polarization = 'VV'
            elif pol == 'SH':
                self.polarization = 'HH'
            elif pol == 'DV':
                self.polarization = 'VV+VH'
            elif pol == 'DH':
                self.polarization = 'HH+HV'
    
    def load_calibration_lut(self, polarization: str = 'vv') -> bool:
        """
        Load calibration LUT from annotation XML.
        
        Args:
            polarization: Polarization channel ('vv', 'vh', 'hh', 'hv')
            
        Returns:
            True if successful, False otherwise
        """
        if not self.safe_path:
            return False
        
        # Find calibration annotation file
        cal_files = list(self.safe_path.glob(f'annotation/calibration/calibration-*-{polarization}-*.xml'))
        
        if not cal_files:
            cal_files = list(self.safe_path.glob('annotation/calibration/*.xml'))
            cal_files = [f for f in cal_files if polarization.lower() in f.name.lower()]
        
        if not cal_files:
            warnings.warn(f"No calibration file found for {polarization}")
            return False
        
        cal_file = cal_files[0]
        
        try:
            tree = ET.parse(cal_file)
            root = tree.getroot()
            
            # Parse calibration vectors
            cal_vectors = root.find('.//calibrationVectorList')
            
            if cal_vectors is None:
                return False
            
            lines = []
            pixels = []
            sigma_values = []
            beta_values = []
            gamma_values = []
            dn_values = []
            azimuth_times = []
            
            for vector in cal_vectors.findall('calibrationVector'):
                azimuth_times.append(vector.findtext('azimuthTime', ''))
                lines.append(int(vector.findtext('line', '0')))
                
                pixel_str = vector.findtext('pixel', '').split()
                pixels.append([int(p) for p in pixel_str])
                
                sigma_str = vector.findtext('sigmaNought', '').split()
                sigma_values.append([float(s) for s in sigma_str])
                
                beta_str = vector.findtext('betaNought', '').split()
                beta_values.append([float(b) for b in beta_str])
                
                gamma_str = vector.findtext('gamma', '').split()
                gamma_values.append([float(g) for g in gamma_str])
                
                dn_str = vector.findtext('dn', '').split()
                if dn_str:
                    dn_values.append([float(d) for d in dn_str])
            
            self.calibration_lut = CalibrationLUT(
                azimuth_time=azimuth_times,
                line=np.array(lines),
                pixel=np.array(pixels[0]) if pixels else np.array([]),
                sigma_nought=np.array(sigma_values),
                beta_nought=np.array(beta_values),
                gamma=np.array(gamma_values),
                dn=np.array(dn_values) if dn_values else np.array([])
            )
            
            return True
            
        except Exception as e:
            warnings.warn(f"Error parsing calibration file: {e}")
            return False
    
    def load_noise_lut(self, polarization: str = 'vv') -> bool:
        """
        Load noise LUT (NESZ) from annotation XML.
        
        Args:
            polarization: Polarization channel
            
        Returns:
            True if successful, False otherwise
        """
        if not self.safe_path:
            return False
        
        # Find noise annotation file
        noise_files = list(self.safe_path.glob(f'annotation/calibration/noise-*-{polarization}-*.xml'))
        
        if not noise_files:
            noise_files = list(self.safe_path.glob('annotation/calibration/noise*.xml'))
            noise_files = [f for f in noise_files if polarization.lower() in f.name.lower()]
        
        if not noise_files:
            return False
        
        noise_file = noise_files[0]
        
        try:
            tree = ET.parse(noise_file)
            root = tree.getroot()
            
            # Parse noise vectors (range or azimuth)
            noise_range = root.find('.//noiseRangeVectorList')
            
            if noise_range is None:
                # Try older format
                noise_range = root.find('.//noiseVectorList')
            
            if noise_range is None:
                return False
            
            lines = []
            pixels = []
            noise_values = []
            azimuth_times = []
            
            for vector in noise_range.findall('.//noiseRangeVector') or noise_range.findall('.//noiseVector'):
                azimuth_times.append(vector.findtext('azimuthTime', ''))
                lines.append(int(vector.findtext('line', '0')))
                
                pixel_str = vector.findtext('pixel', '').split()
                pixels.append([int(p) for p in pixel_str] if pixel_str else [])
                
                noise_str = vector.findtext('noiseRangeLut', '') or vector.findtext('noiseLut', '')
                noise_values.append([float(n) for n in noise_str.split()] if noise_str else [])
            
            if lines and pixels and noise_values:
                self.noise_lut = NoiseLUT(
                    azimuth_time=azimuth_times,
                    line=np.array(lines),
                    pixel=np.array(pixels[0]) if pixels[0] else np.array([]),
                    noise_lut=np.array(noise_values)
                )
                return True
            
            return False
            
        except Exception as e:
            warnings.warn(f"Error parsing noise file: {e}")
            return False
    
    def calibrate_sigma0(
        self,
        dn_image: np.ndarray,
        polarization: str = 'vv',
        apply_noise_correction: bool = True
    ) -> np.ndarray:
        """
        Calibrate digital numbers to sigma-nought (σ°).
        
        σ° = 10 * log10(DN² / A_sigma²) - NESZ (optional)
        
        Args:
            dn_image: Digital number image (intensity or amplitude)
            polarization: Polarization channel
            apply_noise_correction: Whether to subtract noise floor
            
        Returns:
            Sigma-nought in dB
        """
        # Try precise calibration first
        if self.calibration_lut is not None:
            return self._calibrate_precise(
                dn_image, 'sigma_nought', polarization, apply_noise_correction
            )
        
        # Fall back to simplified calibration
        return self._calibrate_simplified(dn_image, polarization)
    
    def calibrate_gamma0(
        self,
        dn_image: np.ndarray,
        polarization: str = 'vv',
        incidence_angle: np.ndarray = None
    ) -> np.ndarray:
        """
        Calibrate to gamma-nought (γ°) - terrain-flattened.
        
        γ° = σ° / cos(θ)
        
        Args:
            dn_image: Digital number image
            polarization: Polarization channel
            incidence_angle: Incidence angle array in radians (optional)
            
        Returns:
            Gamma-nought in dB
        """
        sigma0 = self.calibrate_sigma0(dn_image, polarization)
        
        if incidence_angle is not None:
            # Convert sigma0 to linear, apply correction, back to dB
            sigma0_linear = 10 ** (sigma0 / 10)
            gamma0_linear = sigma0_linear / np.cos(incidence_angle)
            return 10 * np.log10(gamma0_linear + 1e-10)
        
        # Without incidence angle, use calibration LUT gamma if available
        if self.calibration_lut is not None:
            return self._calibrate_precise(dn_image, 'gamma', polarization, False)
        
        return sigma0
    
    def calibrate_beta0(
        self,
        dn_image: np.ndarray,
        polarization: str = 'vv'
    ) -> np.ndarray:
        """
        Calibrate to beta-nought (β°) - radar brightness.
        
        Args:
            dn_image: Digital number image
            polarization: Polarization channel
            
        Returns:
            Beta-nought in dB
        """
        if self.calibration_lut is not None:
            return self._calibrate_precise(dn_image, 'beta_nought', polarization, False)
        
        # For simplified, beta0 ≈ sigma0 + angle correction
        return self._calibrate_simplified(dn_image, polarization)
    
    def _calibrate_precise(
        self,
        dn_image: np.ndarray,
        cal_type: str,
        polarization: str,
        apply_noise_correction: bool
    ) -> np.ndarray:
        """Precise calibration using LUT."""
        
        # Get calibration array
        if cal_type == 'sigma_nought':
            cal_lut = self.calibration_lut.sigma_nought
        elif cal_type == 'gamma':
            cal_lut = self.calibration_lut.gamma
        elif cal_type == 'beta_nought':
            cal_lut = self.calibration_lut.beta_nought
        else:
            raise ValueError(f"Unknown calibration type: {cal_type}")
        
        # Interpolate LUT to image size
        from scipy.interpolate import RectBivariateSpline
        
        lines = self.calibration_lut.line
        pixels = self.calibration_lut.pixel
        
        # Create interpolator
        if len(lines) > 1 and len(pixels) > 1:
            try:
                interp = RectBivariateSpline(lines, pixels, cal_lut)
                
                # Generate full-res calibration grid
                img_lines = np.arange(dn_image.shape[0])
                img_pixels = np.arange(dn_image.shape[1])
                cal_grid = interp(img_lines, img_pixels)
                
            except Exception:
                # Fall back to nearest neighbor
                cal_grid = np.full(dn_image.shape, np.mean(cal_lut))
        else:
            cal_grid = np.full(dn_image.shape, np.mean(cal_lut))
        
        # Apply calibration
        # σ° = DN² / A²
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            dn_squared = dn_image.astype(np.float64) ** 2
            calibrated = dn_squared / (cal_grid ** 2 + 1e-10)
        
        # Noise correction
        if apply_noise_correction and self.noise_lut is not None:
            noise_power = self._interpolate_noise(dn_image.shape)
            calibrated = calibrated - noise_power
            calibrated = np.maximum(calibrated, 1e-10)  # Avoid negative values
        
        # Convert to dB
        calibrated_db = 10 * np.log10(calibrated + 1e-10)
        
        return np.clip(calibrated_db, -50, 50)
    
    def _calibrate_simplified(
        self,
        dn_image: np.ndarray,
        polarization: str
    ) -> np.ndarray:
        """Simplified calibration without annotation files."""
        
        mode = self.mode or 'IW'
        pol = polarization.upper()
        
        # Get calibration offset
        if mode in self.DEFAULT_CALIBRATION and pol in self.DEFAULT_CALIBRATION[mode]:
            offset = self.DEFAULT_CALIBRATION[mode][pol]
        else:
            offset = -20.0  # Default fallback
        
        # Simple calibration: σ° ≈ 10*log10(DN²) + offset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dn_squared = dn_image.astype(np.float64) ** 2
            calibrated_db = 10 * np.log10(dn_squared + 1e-10) + offset
        
        return np.clip(calibrated_db, -50, 50)
    
    def _interpolate_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """Interpolate noise LUT to image size."""
        if self.noise_lut is None:
            return np.zeros(shape)
        
        from scipy.interpolate import RectBivariateSpline
        
        lines = self.noise_lut.line
        pixels = self.noise_lut.pixel
        noise_values = self.noise_lut.noise_lut
        
        if len(lines) > 1 and len(pixels) > 1:
            try:
                interp = RectBivariateSpline(lines, pixels, noise_values)
                img_lines = np.arange(shape[0])
                img_pixels = np.arange(shape[1])
                return interp(img_lines, img_pixels)
            except Exception:
                pass
        
        return np.full(shape, np.mean(noise_values))
    
    def get_calibration_method(self) -> str:
        """Return description of calibration method being used."""
        if self.calibration_lut is not None:
            return "Precise (XML annotation LUT)"
        else:
            return "Simplified (default constants)"


def find_calibration_files(safe_path: Path) -> Dict[str, List[Path]]:
    """
    Find all calibration-related files in a .SAFE product.
    
    Args:
        safe_path: Path to .SAFE directory
        
    Returns:
        Dictionary with file categories and paths
    """
    safe_path = Path(safe_path)
    
    files = {
        'calibration': [],
        'noise': [],
        'annotation': [],
        'measurement': []
    }
    
    # Calibration files
    cal_dir = safe_path / 'annotation' / 'calibration'
    if cal_dir.exists():
        files['calibration'] = list(cal_dir.glob('calibration-*.xml'))
        files['noise'] = list(cal_dir.glob('noise-*.xml'))
    
    # Main annotation files
    ann_dir = safe_path / 'annotation'
    if ann_dir.exists():
        files['annotation'] = [f for f in ann_dir.glob('*.xml') 
                              if 'calibration' not in str(f)]
    
    # Measurement files (actual data)
    meas_dir = safe_path / 'measurement'
    if meas_dir.exists():
        files['measurement'] = list(meas_dir.glob('*.tiff')) + list(meas_dir.glob('*.tif'))
    
    return files


def apply_lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Lee speckle filter to SAR image.
    
    Args:
        image: Input SAR image (linear or dB)
        window_size: Filter window size (odd number)
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import uniform_filter
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Ensure float
    img = image.astype(np.float64)
    
    # Local mean and variance
    img_mean = uniform_filter(img, window_size)
    img_sqr_mean = uniform_filter(img ** 2, window_size)
    img_var = img_sqr_mean - img_mean ** 2
    
    # Overall variance
    overall_var = np.var(img)
    
    # Lee filter weight
    weight = img_var / (img_var + overall_var + 1e-10)
    
    # Apply filter
    filtered = img_mean + weight * (img - img_mean)
    
    return filtered


def apply_refined_lee_filter(image: np.ndarray, window_size: int = 7, 
                             num_looks: int = 1) -> np.ndarray:
    """
    Apply Refined Lee speckle filter with edge-preserving properties.
    
    Args:
        image: Input SAR image
        window_size: Filter window size
        num_looks: Number of looks (for ENL estimation)
        
    Returns:
        Filtered image
    """
    from scipy.ndimage import uniform_filter
    
    img = image.astype(np.float64)
    
    # Estimate local statistics
    local_mean = uniform_filter(img, window_size)
    local_sqr_mean = uniform_filter(img ** 2, window_size)
    local_var = local_sqr_mean - local_mean ** 2
    
    # Coefficient of variation
    cv = np.sqrt(local_var) / (local_mean + 1e-10)
    
    # Theoretical CV for speckle
    cv_speckle = 1.0 / np.sqrt(num_looks)
    
    # Weight factor
    weight = 1 - (cv_speckle ** 2) / (cv ** 2 + 1e-10)
    weight = np.clip(weight, 0, 1)
    
    # Apply weighted filter
    filtered = local_mean + weight * (img - local_mean)
    
    return filtered
