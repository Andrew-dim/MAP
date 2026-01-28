"""
Terrain Classification Module
Land cover classification and trafficability analysis

Implements:
- Land cover classification (water, vegetation, urban, bare soil)
- Trafficability mapping for vehicle movement
- Soil moisture estimation (S-band)
- Water body detection
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import ndimage
from scipy.ndimage import label
import json


@dataclass
class TerrainClass:
    """Terrain classification class definition."""
    class_id: int
    name: str
    color: Tuple[int, int, int]  # RGB
    trafficability: float  # 0-1 (1=easy to traverse)
    description: str


class TerrainClassifier:
    """
    Classify terrain from polarimetric SAR data.
    
    Uses H-α classification and polarimetric indices.
    """
    
    # Standard terrain classes
    CLASSES = {
        0: TerrainClass(0, "No Data", (0, 0, 0), 0.0, "No data or masked"),
        1: TerrainClass(1, "Water", (0, 0, 255), 0.0, "Open water bodies"),
        2: TerrainClass(2, "Wetland", (0, 128, 255), 0.1, "Marshes, flooded areas"),
        3: TerrainClass(3, "Bare Soil", (139, 90, 43), 0.8, "Exposed soil, beaches"),
        4: TerrainClass(4, "Low Vegetation", (144, 238, 144), 0.7, "Grass, crops, shrubs"),
        5: TerrainClass(5, "Forest", (34, 139, 34), 0.3, "Dense forest/canopy"),
        6: TerrainClass(6, "Urban Low", (255, 165, 0), 0.5, "Low-density built-up"),
        7: TerrainClass(7, "Urban High", (255, 0, 0), 0.2, "High-density built-up"),
        8: TerrainClass(8, "Road/Runway", (128, 128, 128), 1.0, "Paved surfaces"),
        9: TerrainClass(9, "Rocky", (128, 128, 0), 0.4, "Rocky terrain"),
    }
    
    def __init__(self):
        """Initialize terrain classifier."""
        self.classification = None
        self.trafficability = None
        self.soil_moisture = None
        self.water_mask = None
    
    def classify(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray,
        alpha: np.ndarray,
        h_alpha_class: np.ndarray,
        fd_surface: np.ndarray = None,
        fd_volume: np.ndarray = None,
        fd_double: np.ndarray = None
    ) -> np.ndarray:
        """
        Perform terrain classification.
        
        Args:
            hh, vv, hv: Backscatter values (dB)
            entropy: H from H-α-A
            alpha: α from H-α-A
            h_alpha_class: 9-zone H-α classification
            fd_surface, fd_volume, fd_double: Freeman-Durden components
            
        Returns:
            Classification array (values 0-9)
        """
        shape = hh.shape
        classification = np.zeros(shape, dtype=np.uint8)
        
        # 1. Water detection (low backscatter, low entropy)
        water = self._detect_water(hh, vv, hv, entropy)
        classification[water] = 1
        self.water_mask = water
        
        # 2. Wetland (transitional)
        wetland = self._detect_wetland(hh, vv, hv, entropy, water)
        classification[wetland] = 2
        
        # 3. Urban detection (double-bounce dominant)
        urban_high, urban_low = self._detect_urban(
            hh, vv, alpha, entropy, h_alpha_class, fd_double
        )
        classification[urban_high] = 7
        classification[urban_low] = 6
        
        # 4. Forest (volume scattering dominant)
        forest = self._detect_forest(entropy, h_alpha_class, fd_volume, hv)
        classification[forest & (classification == 0)] = 5
        
        # 5. Low vegetation
        low_veg = self._detect_low_vegetation(entropy, alpha, h_alpha_class, fd_volume)
        classification[low_veg & (classification == 0)] = 4
        
        # 6. Bare soil (surface scattering, low entropy)
        bare = self._detect_bare_soil(entropy, alpha, h_alpha_class, fd_surface)
        classification[bare & (classification == 0)] = 3
        
        # 7. Roads/runways (smooth, linear features)
        roads = self._detect_roads(hh, entropy)
        classification[roads & (classification == 0)] = 8
        
        # 8. Rocky terrain
        rocky = self._detect_rocky(hh, vv, entropy, alpha)
        classification[rocky & (classification == 0)] = 9
        
        # Fill remaining with bare soil
        classification[classification == 0] = 3
        
        # Morphological cleanup
        classification = self._cleanup_classification(classification)
        
        self.classification = classification
        return classification
    
    def _detect_water(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray
    ) -> np.ndarray:
        """Detect open water bodies."""
        # Water is very dark in all polarizations
        # Low entropy (specular reflection)
        water = (
            (hh < -18) &
            (vv < -18) &
            (hv < -25) &
            (entropy < 0.5)
        )
        
        # Remove small isolated pixels
        water = ndimage.binary_opening(water, iterations=2)
        water = ndimage.binary_closing(water, iterations=1)
        
        return water
    
    def _detect_wetland(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray,
        water: np.ndarray
    ) -> np.ndarray:
        """Detect wetlands/marshes."""
        # Wetlands: moderate backscatter, higher than water
        wetland = (
            (hh > -25) & (hh < -12) &
            (entropy > 0.4) & (entropy < 0.7) &
            ~water
        )
        
        # Should be near water
        water_dilated = ndimage.binary_dilation(water, iterations=5)
        wetland = wetland & water_dilated
        
        return wetland
    
    def _detect_urban(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        alpha: np.ndarray,
        entropy: np.ndarray,
        h_alpha_class: np.ndarray,
        fd_double: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect urban areas (double-bounce scattering)."""
        # High alpha = double-bounce
        # Low entropy = deterministic scattering
        # Bright in HH and VV
        
        urban_base = (
            (alpha > 50) &
            (hh > -15) &
            (vv > -15) &
            (entropy < 0.6)
        )
        
        if fd_double is not None:
            urban_base = urban_base | (fd_double > 0.4)
        
        # Separate high/low density by intensity
        urban_high = urban_base & (hh > -8) & (vv > -8)
        urban_low = urban_base & ~urban_high
        
        # Cleanup
        urban_high = ndimage.binary_opening(urban_high, iterations=1)
        urban_low = ndimage.binary_opening(urban_low, iterations=1)
        
        return urban_high, urban_low
    
    def _detect_forest(
        self,
        entropy: np.ndarray,
        h_alpha_class: np.ndarray,
        fd_volume: np.ndarray = None,
        hv: np.ndarray = None
    ) -> np.ndarray:
        """Detect forested areas (volume scattering)."""
        # Forest: high entropy, volume scattering dominant
        # H-α zones 2, 5 are vegetation
        
        forest = (
            (entropy > 0.6) &
            ((h_alpha_class == 2) | (h_alpha_class == 5))
        )
        
        if fd_volume is not None:
            forest = forest | (fd_volume > 0.5)
        
        if hv is not None:
            # Forest has high HV (cross-pol from canopy)
            forest = forest | ((hv > -18) & (entropy > 0.5))
        
        forest = ndimage.binary_opening(forest, iterations=2)
        
        return forest
    
    def _detect_low_vegetation(
        self,
        entropy: np.ndarray,
        alpha: np.ndarray,
        h_alpha_class: np.ndarray,
        fd_volume: np.ndarray = None
    ) -> np.ndarray:
        """Detect low vegetation (grass, crops)."""
        # Lower entropy than forest, some volume scattering
        
        low_veg = (
            (entropy > 0.4) & (entropy < 0.7) &
            (alpha > 30) & (alpha < 55)
        )
        
        if fd_volume is not None:
            low_veg = low_veg & (fd_volume > 0.2) & (fd_volume < 0.5)
        
        return low_veg
    
    def _detect_bare_soil(
        self,
        entropy: np.ndarray,
        alpha: np.ndarray,
        h_alpha_class: np.ndarray,
        fd_surface: np.ndarray = None
    ) -> np.ndarray:
        """Detect bare soil (surface scattering)."""
        # Low entropy, low alpha (surface/Bragg scattering)
        # H-α zones 7, 4, 1 are surface
        
        bare = (
            (entropy < 0.5) &
            (alpha < 40) &
            ((h_alpha_class == 1) | (h_alpha_class == 4) | (h_alpha_class == 7))
        )
        
        if fd_surface is not None:
            bare = bare | (fd_surface > 0.6)
        
        return bare
    
    def _detect_roads(
        self,
        hh: np.ndarray,
        entropy: np.ndarray
    ) -> np.ndarray:
        """Detect roads and paved surfaces."""
        # Roads: low entropy, moderate backscatter, linear features
        
        road_candidate = (entropy < 0.3) & (hh > -20) & (hh < -5)
        
        # Linear feature detection (simplified)
        # In full implementation, use edge detection and Hough transform
        
        return road_candidate
    
    def _detect_rocky(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        entropy: np.ndarray,
        alpha: np.ndarray
    ) -> np.ndarray:
        """Detect rocky terrain."""
        # Rocky: rough surface, moderate entropy
        
        rocky = (
            (entropy > 0.3) & (entropy < 0.6) &
            (alpha > 25) & (alpha < 45) &
            (hh > -18)
        )
        
        return rocky
    
    def _cleanup_classification(self, classification: np.ndarray) -> np.ndarray:
        """Apply morphological cleanup to classification."""
        cleaned = classification.copy()
        
        # Remove isolated pixels for each class
        for class_id in range(1, 10):
            mask = (classification == class_id)
            mask = ndimage.binary_opening(mask, iterations=1)
            cleaned[mask] = class_id
        
        return cleaned
    
    def compute_trafficability(
        self,
        classification: np.ndarray = None,
        slope: np.ndarray = None,
        soil_moisture: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute trafficability map for vehicle movement.
        
        Args:
            classification: Terrain classification
            slope: Slope in degrees (optional)
            soil_moisture: Soil moisture 0-1 (optional)
            
        Returns:
            Trafficability map (0=impassable, 1=easy)
        """
        if classification is None:
            classification = self.classification
        
        if classification is None:
            raise ValueError("No classification available")
        
        # Base trafficability from land cover
        trafficability = np.zeros_like(classification, dtype=np.float32)
        
        for class_id, terrain_class in self.CLASSES.items():
            mask = (classification == class_id)
            trafficability[mask] = terrain_class.trafficability
        
        # Reduce trafficability on steep slopes
        if slope is not None:
            slope_factor = 1.0 - np.clip(slope / 30.0, 0, 1)  # >30° impassable
            trafficability = trafficability * slope_factor
        
        # Reduce trafficability on wet soil
        if soil_moisture is not None:
            moisture_factor = 1.0 - 0.5 * soil_moisture  # High moisture = harder
            trafficability = trafficability * moisture_factor
        
        self.trafficability = np.clip(trafficability, 0, 1)
        return self.trafficability
    
    def estimate_soil_moisture(
        self,
        s_band_hh: np.ndarray,
        s_band_vv: np.ndarray,
        vegetation_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Estimate soil moisture from S-band backscatter.
        
        S-band is sensitive to soil moisture.
        
        Args:
            s_band_hh, s_band_vv: S-band backscatter (dB)
            vegetation_mask: Mask where vegetation prevents soil sensing
            
        Returns:
            Soil moisture (0-1, volumetric)
        """
        # Empirical model (simplified)
        # Real implementation would use calibrated coefficients
        
        # Average of HH and VV
        sigma = (s_band_hh + s_band_vv) / 2
        
        # Map backscatter to moisture
        # Typical range: -25dB (dry) to -10dB (saturated)
        moisture = (sigma + 25) / 15  # Linear mapping
        moisture = np.clip(moisture, 0, 1)
        
        # Mask vegetation (cannot see soil)
        if vegetation_mask is not None:
            moisture[vegetation_mask] = np.nan
        
        self.soil_moisture = moisture
        return moisture
    
    def get_water_mask(self) -> np.ndarray:
        """Return binary water mask."""
        if self.water_mask is None:
            raise ValueError("Run classify() first")
        return self.water_mask
    
    def get_vegetation_mask(self) -> np.ndarray:
        """Return binary vegetation mask (forest + low veg)."""
        if self.classification is None:
            raise ValueError("Run classify() first")
        return (self.classification == 4) | (self.classification == 5)
    
    def get_urban_mask(self) -> np.ndarray:
        """Return binary urban mask."""
        if self.classification is None:
            raise ValueError("Run classify() first")
        return (self.classification == 6) | (self.classification == 7)
    
    def export_classification(
        self,
        output_path: Path,
        include_legend: bool = True
    ) -> Path:
        """Export classification as GeoTIFF with color table."""
        # Implementation would use rasterio/GDAL
        # For now, save as numpy
        np.save(output_path.with_suffix('.npy'), self.classification)
        
        # Save legend
        if include_legend:
            legend = {
                str(k): {
                    'name': v.name,
                    'color': v.color,
                    'trafficability': v.trafficability,
                    'description': v.description
                }
                for k, v in self.CLASSES.items()
            }
            legend_path = output_path.with_suffix('.legend.json')
            with open(legend_path, 'w') as f:
                json.dump(legend, f, indent=2)
        
        return output_path
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute area statistics for each class."""
        if self.classification is None:
            raise ValueError("Run classify() first")
        
        total_pixels = self.classification.size
        stats = {}
        
        for class_id, terrain_class in self.CLASSES.items():
            count = np.sum(self.classification == class_id)
            percentage = (count / total_pixels) * 100
            stats[terrain_class.name] = {
                'pixels': int(count),
                'percentage': round(percentage, 2)
            }
        
        return stats
