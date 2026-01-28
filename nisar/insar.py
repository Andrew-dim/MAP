"""
InSAR Processing Module
Interferometric SAR for deformation and change detection

Implements:
- Coherence estimation
- Phase unwrapping (simplified)
- Deformation velocity estimation
- Change detection
- DEM generation support
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import ndimage
from scipy.ndimage import uniform_filter
import json
from datetime import datetime, timedelta


@dataclass
class InSARResult:
    """Container for InSAR processing results."""
    # Coherence
    coherence: np.ndarray      # 0-1 (1=high correlation)
    
    # Interferometric phase
    phase: np.ndarray          # radians (-π to π)
    phase_unwrapped: np.ndarray  # radians (unwrapped)
    
    # Deformation
    deformation_los: np.ndarray  # Line-of-sight displacement (meters)
    deformation_vertical: np.ndarray  # Vertical displacement (meters)
    
    # Velocity (if time series)
    velocity: np.ndarray       # mm/year
    
    # Change detection
    change_map: np.ndarray     # Binary change mask
    change_intensity: np.ndarray  # Change magnitude
    
    # Metadata
    temporal_baseline: float   # days
    perpendicular_baseline: float  # meters
    wavelength: float          # meters
    incidence_angle: float     # degrees


class InSARProcessor:
    """
    Process InSAR data for deformation and change detection.
    
    Supports NISAR L2-InSAR products and SLC pairs.
    """
    
    # NISAR wavelengths
    WAVELENGTH = {
        'L-band': 0.238,  # 23.8 cm
        'S-band': 0.094,  # 9.4 cm
    }
    
    def __init__(self, frequency: str = 'L-band'):
        """
        Initialize InSAR processor.
        
        Args:
            frequency: 'L-band' or 'S-band'
        """
        self.frequency = frequency
        self.wavelength = self.WAVELENGTH.get(frequency, 0.238)
        self.results = None
    
    def process_nisar_insar(self, h5_path: Path) -> InSARResult:
        """
        Process NISAR L2-InSAR product.
        
        Args:
            h5_path: Path to NISAR InSAR HDF5 file
            
        Returns:
            InSARResult with all products
        """
        import h5py
        
        with h5py.File(h5_path, 'r') as f:
            # Navigate to InSAR data
            insar_path = self._find_insar_group(f)
            
            if not insar_path:
                raise ValueError("Could not find InSAR group in file")
            
            insar = f[insar_path]
            
            # Read wrapped interferogram
            phase = np.array(insar.get('wrappedInterferogram', 
                                       insar.get('unwrappedPhase', np.zeros((100,100)))))
            
            # Read coherence
            coherence = np.array(insar.get('coherenceMagnitude',
                                          insar.get('coherence', np.ones_like(phase) * 0.5)))
            
            # Read unwrapped phase if available
            if 'unwrappedPhase' in insar:
                phase_unwrapped = np.array(insar['unwrappedPhase'])
            else:
                phase_unwrapped = self._unwrap_phase(phase, coherence)
            
            # Get metadata
            meta = self._extract_metadata(f, insar_path)
        
        # Compute deformation
        deformation_los = self._phase_to_displacement(phase_unwrapped)
        deformation_vertical = self._los_to_vertical(deformation_los, meta['incidence_angle'])
        
        # Change detection
        change_map, change_intensity = self._detect_changes(coherence, phase)
        
        # No velocity from single pair
        velocity = np.zeros_like(deformation_los)
        
        self.results = InSARResult(
            coherence=coherence,
            phase=phase,
            phase_unwrapped=phase_unwrapped,
            deformation_los=deformation_los,
            deformation_vertical=deformation_vertical,
            velocity=velocity,
            change_map=change_map,
            change_intensity=change_intensity,
            temporal_baseline=meta.get('temporal_baseline', 12),
            perpendicular_baseline=meta.get('perpendicular_baseline', 0),
            wavelength=self.wavelength,
            incidence_angle=meta.get('incidence_angle', 35)
        )
        
        return self.results
    
    def process_slc_pair(
        self,
        slc1: np.ndarray,
        slc2: np.ndarray,
        temporal_baseline: float = 12,
        incidence_angle: float = 35
    ) -> InSARResult:
        """
        Process a pair of SLC images.
        
        Args:
            slc1: Primary SLC (complex)
            slc2: Secondary SLC (complex)
            temporal_baseline: Days between acquisitions
            incidence_angle: Incidence angle in degrees
            
        Returns:
            InSARResult
        """
        # Compute interferogram
        interferogram = slc1 * np.conj(slc2)
        
        # Extract phase
        phase = np.angle(interferogram)
        
        # Compute coherence
        coherence = self._estimate_coherence(slc1, slc2)
        
        # Unwrap phase
        phase_unwrapped = self._unwrap_phase(phase, coherence)
        
        # Compute deformation
        deformation_los = self._phase_to_displacement(phase_unwrapped)
        deformation_vertical = self._los_to_vertical(deformation_los, incidence_angle)
        
        # Change detection
        change_map, change_intensity = self._detect_changes(coherence, phase)
        
        # Velocity estimate (rough)
        velocity = (deformation_vertical / temporal_baseline) * 365 * 1000  # mm/year
        
        self.results = InSARResult(
            coherence=coherence,
            phase=phase,
            phase_unwrapped=phase_unwrapped,
            deformation_los=deformation_los,
            deformation_vertical=deformation_vertical,
            velocity=velocity,
            change_map=change_map,
            change_intensity=change_intensity,
            temporal_baseline=temporal_baseline,
            perpendicular_baseline=0,
            wavelength=self.wavelength,
            incidence_angle=incidence_angle
        )
        
        return self.results
    
    def _estimate_coherence(
        self,
        slc1: np.ndarray,
        slc2: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Estimate interferometric coherence.
        
        Coherence = |<S1 * S2*>| / sqrt(<|S1|²> * <|S2|²>)
        """
        # Compute correlation terms
        s1_s2_conj = slc1 * np.conj(slc2)
        s1_power = np.abs(slc1) ** 2
        s2_power = np.abs(slc2) ** 2
        
        # Spatial averaging
        numerator = uniform_filter(s1_s2_conj.real, window_size) + \
                   1j * uniform_filter(s1_s2_conj.imag, window_size)
        
        denom1 = uniform_filter(s1_power, window_size)
        denom2 = uniform_filter(s2_power, window_size)
        
        # Coherence magnitude
        coherence = np.abs(numerator) / (np.sqrt(denom1 * denom2) + 1e-10)
        coherence = np.clip(coherence, 0, 1)
        
        return coherence
    
    def _unwrap_phase(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        coherence_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Unwrap interferometric phase.
        
        Uses simplified quality-guided unwrapping.
        Real implementation would use SNAPHU or similar.
        """
        # Mask low coherence areas
        mask = coherence > coherence_threshold
        
        # Simple row-by-row unwrapping (not production quality)
        unwrapped = np.zeros_like(wrapped_phase)
        
        for i in range(wrapped_phase.shape[0]):
            unwrapped[i, :] = self._unwrap_1d(wrapped_phase[i, :])
        
        # Adjust with column continuity
        for j in range(wrapped_phase.shape[1]):
            col_unwrap = self._unwrap_1d(unwrapped[:, j])
            # Blend with row-unwrapped
            unwrapped[:, j] = 0.5 * (unwrapped[:, j] + col_unwrap)
        
        # Mask low coherence
        unwrapped[~mask] = np.nan
        
        return unwrapped
    
    def _unwrap_1d(self, phase_1d: np.ndarray) -> np.ndarray:
        """1D phase unwrapping."""
        unwrapped = phase_1d.copy()
        
        for i in range(1, len(phase_1d)):
            diff = phase_1d[i] - phase_1d[i-1]
            
            # Detect and correct wraps
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            
            unwrapped[i] = unwrapped[i-1] + diff
        
        return unwrapped
    
    def _phase_to_displacement(self, phase: np.ndarray) -> np.ndarray:
        """
        Convert unwrapped phase to line-of-sight displacement.
        
        Displacement = (wavelength / 4π) * phase
        """
        displacement = (self.wavelength / (4 * np.pi)) * phase
        return displacement  # meters
    
    def _los_to_vertical(
        self,
        los_displacement: np.ndarray,
        incidence_angle: float
    ) -> np.ndarray:
        """
        Convert LOS displacement to vertical displacement.
        
        Assumes purely vertical motion.
        """
        inc_rad = np.radians(incidence_angle)
        vertical = los_displacement / np.cos(inc_rad)
        return vertical
    
    def _detect_changes(
        self,
        coherence: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect changes between acquisitions.
        
        Low coherence indicates change (decorrelation).
        """
        # Change where coherence is low
        coherence_threshold = 0.3
        change_map = coherence < coherence_threshold
        
        # Change intensity based on coherence loss
        # Higher = more change
        change_intensity = 1.0 - coherence
        
        # Additional: phase variance indicates change
        phase_var = ndimage.generic_filter(
            phase, np.std, size=5
        )
        phase_var_norm = phase_var / (np.pi / 2)  # Normalize
        
        # Combine coherence and phase variance
        change_intensity = 0.7 * (1 - coherence) + 0.3 * phase_var_norm
        change_intensity = np.clip(change_intensity, 0, 1)
        
        # Update change map with combined metric
        change_map = change_map | (change_intensity > 0.6)
        
        # Morphological cleanup
        change_map = ndimage.binary_opening(change_map, iterations=1)
        
        return change_map, change_intensity
    
    def detect_subsidence(
        self,
        velocity_threshold: float = -10.0  # mm/year
    ) -> np.ndarray:
        """
        Detect subsidence (sinking ground).
        
        Args:
            velocity_threshold: Threshold for subsidence (negative = sinking)
            
        Returns:
            Binary subsidence mask
        """
        if self.results is None:
            raise ValueError("Run processing first")
        
        subsidence = self.results.velocity < velocity_threshold
        
        # Cleanup
        subsidence = ndimage.binary_opening(subsidence, iterations=2)
        
        return subsidence
    
    def detect_uplift(
        self,
        velocity_threshold: float = 10.0  # mm/year
    ) -> np.ndarray:
        """
        Detect uplift (rising ground).
        
        Args:
            velocity_threshold: Threshold for uplift (positive = rising)
            
        Returns:
            Binary uplift mask
        """
        if self.results is None:
            raise ValueError("Run processing first")
        
        uplift = self.results.velocity > velocity_threshold
        
        # Cleanup
        uplift = ndimage.binary_opening(uplift, iterations=2)
        
        return uplift
    
    def detect_underground_activity(
        self,
        sensitivity: float = 0.5
    ) -> np.ndarray:
        """
        Detect potential underground activity (tunneling, construction).
        
        Looks for localized deformation patterns.
        
        Args:
            sensitivity: Detection sensitivity (0-1)
            
        Returns:
            Anomaly probability map
        """
        if self.results is None:
            raise ValueError("Run processing first")
        
        deformation = self.results.deformation_vertical
        
        # Local deformation anomalies
        local_mean = uniform_filter(deformation, size=21)
        local_std = np.sqrt(uniform_filter(deformation**2, size=21) - local_mean**2)
        
        # Anomaly = deviation from local mean
        anomaly = np.abs(deformation - local_mean) / (local_std + 1e-6)
        
        # Threshold based on sensitivity
        threshold = 3.0 - 2.0 * sensitivity  # 1-3 sigma
        anomaly_prob = np.clip((anomaly - threshold) / threshold, 0, 1)
        
        # Filter by size (tunnels have specific spatial extent)
        anomaly_prob = ndimage.gaussian_filter(anomaly_prob, sigma=2)
        
        return anomaly_prob
    
    def compute_dem_error(
        self,
        perpendicular_baseline: float,
        range_distance: float = 850000  # meters (typical)
    ) -> np.ndarray:
        """
        Estimate DEM error from phase.
        
        Perpendicular baseline creates topographic phase.
        
        Args:
            perpendicular_baseline: Baseline in meters
            range_distance: Slant range distance
            
        Returns:
            DEM error in meters
        """
        if self.results is None:
            raise ValueError("Run processing first")
        
        phase = self.results.phase_unwrapped
        inc_rad = np.radians(self.results.incidence_angle)
        
        # Height sensitivity
        # dh = (λ * R * sin(θ)) / (4π * B_perp) * dφ
        height_per_rad = (self.wavelength * range_distance * np.sin(inc_rad)) / \
                        (4 * np.pi * perpendicular_baseline + 1e-6)
        
        dem_error = phase * height_per_rad
        
        return dem_error
    
    def _find_insar_group(self, f) -> Optional[str]:
        """Find InSAR group in HDF5 file."""
        possible_paths = [
            '/science/LSAR/RUNW/grids/frequencyA',
            '/science/LSAR/GUNW/grids/frequencyA',
            '/science/SSAR/RUNW/grids/frequencyA',
            '/INSAR/frequencyA',
        ]
        
        for path in possible_paths:
            if path in f:
                return path
        
        return None
    
    def _extract_metadata(self, f, insar_path: str) -> dict:
        """Extract metadata from file."""
        meta = {
            'temporal_baseline': 12,
            'perpendicular_baseline': 0,
            'incidence_angle': 35
        }
        
        try:
            # Try to find metadata groups
            id_path = '/identification'
            if id_path in f:
                ident = f[id_path]
                # Extract baseline info if available
                pass
        except:
            pass
        
        return meta
    
    def export_results(self, output_dir: Path) -> Dict[str, Path]:
        """Export all InSAR products."""
        if self.results is None:
            raise ValueError("Run processing first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Save each product
        products = {
            'coherence': self.results.coherence,
            'phase_wrapped': self.results.phase,
            'phase_unwrapped': self.results.phase_unwrapped,
            'deformation_los': self.results.deformation_los,
            'deformation_vertical': self.results.deformation_vertical,
            'velocity': self.results.velocity,
            'change_map': self.results.change_map.astype(np.uint8),
            'change_intensity': self.results.change_intensity
        }
        
        for name, data in products.items():
            path = output_dir / f"{name}.npy"
            np.save(path, data)
            paths[name] = path
        
        # Save metadata
        meta = {
            'temporal_baseline_days': self.results.temporal_baseline,
            'perpendicular_baseline_m': self.results.perpendicular_baseline,
            'wavelength_m': self.results.wavelength,
            'incidence_angle_deg': self.results.incidence_angle,
            'frequency': self.frequency
        }
        
        meta_path = output_dir / 'insar_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        paths['metadata'] = meta_path
        
        return paths
