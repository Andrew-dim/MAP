"""
Target Detection Module
CFAR and classification algorithms for military applications

Implements:
- CFAR (Constant False Alarm Rate) detector
- Ship detection and classification
- Vehicle/convoy detection
- Building detection
- Camouflage/anomaly detection
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from scipy import ndimage
from scipy.ndimage import label, find_objects
import json
from datetime import datetime


@dataclass
class DetectedTarget:
    """Represents a detected target."""
    target_id: str
    target_type: str  # ship, vehicle, building, unknown
    confidence: float  # 0-1
    
    # Location
    center_lat: float
    center_lon: float
    bbox: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    
    # Size estimates
    length_m: float
    width_m: float
    area_m2: float
    
    # Polarimetric features
    hh_mean: float
    vv_mean: float
    hv_mean: float
    entropy: float
    alpha: float
    
    # Detection metadata
    cfar_value: float
    pixel_count: int
    
    # Classification details
    classification_scores: Dict[str, float] = field(default_factory=dict)
    
    # Optional image chip path
    chip_path: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
    
    def to_geojson_feature(self):
        """Convert to GeoJSON Feature."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.center_lon, self.center_lat]
            },
            "properties": {
                "target_id": self.target_id,
                "target_type": self.target_type,
                "confidence": round(self.confidence, 3),
                "length_m": round(self.length_m, 1),
                "width_m": round(self.width_m, 1),
                "area_m2": round(self.area_m2, 1),
                "hh_db": round(self.hh_mean, 1),
                "vv_db": round(self.vv_mean, 1),
                "hv_db": round(self.hv_mean, 1),
                "entropy": round(self.entropy, 3),
                "alpha_deg": round(self.alpha, 1),
                "cfar_value": round(self.cfar_value, 2),
                "classification_scores": {k: round(v, 3) for k, v in self.classification_scores.items()}
            }
        }


class TargetDetector:
    """
    Detect and classify targets in SAR imagery.
    
    Uses CFAR detection with polarimetric classification.
    """
    
    # Default detection parameters
    DEFAULT_CONFIG = {
        'cfar_threshold': 3.0,      # Standard deviations
        'cfar_guard_size': 3,       # Guard window radius
        'cfar_background_size': 10, # Background window radius
        'min_target_pixels': 5,     # Minimum target size
        'max_target_pixels': 5000,  # Maximum target size
        'min_aspect_ratio': 0.1,    # Minimum length/width ratio
        'max_aspect_ratio': 10.0,   # Maximum length/width ratio
    }
    
    # Classification thresholds (empirically derived)
    SHIP_FEATURES = {
        'min_hh': -15,     # dB
        'max_entropy': 0.7,
        'min_length': 20,  # meters
        'max_length': 400,
        'aspect_ratio': (2, 8),  # Ships are elongated
    }
    
    VEHICLE_FEATURES = {
        'min_hh': -18,
        'max_entropy': 0.5,
        'min_length': 3,
        'max_length': 30,
        'aspect_ratio': (1.5, 5),
    }
    
    BUILDING_FEATURES = {
        'min_hh': -12,
        'max_entropy': 0.4,
        'min_alpha': 50,  # Double-bounce dominant
        'aspect_ratio': (0.5, 2),  # More square
    }
    
    def __init__(self, config: dict = None, pixel_spacing: float = 10.0):
        """
        Initialize target detector.
        
        Args:
            config: Detection configuration dictionary
            pixel_spacing: Ground sample distance in meters
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.pixel_spacing = pixel_spacing
        self.detected_targets: List[DetectedTarget] = []
    
    def detect_targets(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray = None,
        alpha: np.ndarray = None,
        geotransform: tuple = None,
        land_mask: np.ndarray = None
    ) -> List[DetectedTarget]:
        """
        Detect all targets in the image.
        
        Args:
            hh, vv, hv: Backscatter images (dB)
            entropy, alpha: Polarimetric parameters (optional)
            geotransform: GDAL geotransform for coordinate conversion
            land_mask: Binary mask (1=land, 0=water) for context
            
        Returns:
            List of DetectedTarget objects
        """
        self.detected_targets = []
        
        # Create detection image (maximize contrast)
        detection_img = self._create_detection_image(hh, vv, hv)
        
        # Apply CFAR detection
        cfar_result = self._cfar_detector(detection_img)
        
        # Segment connected components
        labeled_array, num_features = label(cfar_result)
        
        if num_features == 0:
            return []
        
        # Process each connected component
        slices = find_objects(labeled_array)
        
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            target_mask = labeled_array[slc] == (i + 1)
            pixel_count = np.sum(target_mask)
            
            # Filter by size
            if pixel_count < self.config['min_target_pixels']:
                continue
            if pixel_count > self.config['max_target_pixels']:
                continue
            
            # Extract target features
            target = self._extract_target_features(
                i + 1, slc, target_mask,
                hh, vv, hv, entropy, alpha,
                cfar_result, geotransform, land_mask
            )
            
            if target:
                self.detected_targets.append(target)
        
        # Classify all targets
        self._classify_targets(land_mask)
        
        return self.detected_targets
    
    def detect_ships(
        self,
        hh: np.ndarray,
        water_mask: np.ndarray,
        geotransform: tuple = None
    ) -> List[DetectedTarget]:
        """
        Specialized ship detection on water.
        
        Args:
            hh: HH polarization (dB) - ships are bright in HH
            water_mask: Binary mask (1=water, 0=land)
            geotransform: Coordinate transform
            
        Returns:
            List of detected ships
        """
        # Mask land areas
        masked_hh = np.where(water_mask, hh, np.nan)
        
        # CFAR on water only
        cfar_result = self._cfar_detector(masked_hh)
        
        # Further filtering
        cfar_result = cfar_result & water_mask
        
        # Segment and extract
        labeled, num = label(cfar_result)
        ships = []
        
        slices = find_objects(labeled)
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            mask = labeled[slc] == (i + 1)
            count = np.sum(mask)
            
            if count < 10 or count > 2000:  # Ship size limits
                continue
            
            # Calculate dimensions
            rows, cols = np.where(mask)
            length = (rows.max() - rows.min() + 1) * self.pixel_spacing
            width = (cols.max() - cols.min() + 1) * self.pixel_spacing
            
            # Ships should be elongated
            aspect = max(length, width) / (min(length, width) + 0.1)
            if aspect < 2 or aspect > 10:
                continue
            
            # Create target
            target = self._create_target_from_mask(
                f"SHIP_{i+1:04d}", slc, mask,
                hh, hh, hh, None, None,
                cfar_result, geotransform
            )
            
            if target:
                target.target_type = 'ship'
                target.confidence = self._calculate_ship_confidence(target)
                ships.append(target)
        
        return ships
    
    def detect_vehicles(
        self,
        hh: np.ndarray,
        road_mask: np.ndarray = None,
        geotransform: tuple = None
    ) -> List[DetectedTarget]:
        """
        Detect vehicles (point targets on land).
        
        Args:
            hh: HH backscatter (vehicles are bright metallic targets)
            road_mask: Optional road network mask
            geotransform: Coordinate transform
            
        Returns:
            List of detected vehicles
        """
        # Use stricter CFAR for small targets
        original_threshold = self.config['cfar_threshold']
        self.config['cfar_threshold'] = max(original_threshold, 4.0)
        
        cfar_result = self._cfar_detector(hh)
        
        self.config['cfar_threshold'] = original_threshold
        
        # If road mask provided, boost confidence near roads
        if road_mask is not None:
            dilated_road = ndimage.binary_dilation(road_mask, iterations=5)
            cfar_result = cfar_result & dilated_road
        
        labeled, num = label(cfar_result)
        vehicles = []
        
        slices = find_objects(labeled)
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            mask = labeled[slc] == (i + 1)
            count = np.sum(mask)
            
            # Vehicles are small
            if count < 3 or count > 100:
                continue
            
            target = self._create_target_from_mask(
                f"VEH_{i+1:04d}", slc, mask,
                hh, hh, hh, None, None,
                cfar_result, geotransform
            )
            
            if target:
                # Check size matches vehicle
                if 3 < target.length_m < 30 and 2 < target.width_m < 10:
                    target.target_type = 'vehicle'
                    target.confidence = 0.7
                    vehicles.append(target)
        
        return vehicles
    
    def detect_buildings(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        alpha: np.ndarray,
        geotransform: tuple = None
    ) -> List[DetectedTarget]:
        """
        Detect buildings using double-bounce signature.
        
        Args:
            hh, vv: Co-pol backscatter
            alpha: Alpha angle (high for double-bounce)
            geotransform: Coordinate transform
            
        Returns:
            List of detected buildings
        """
        # Buildings have strong double-bounce (high alpha, bright HH and VV)
        building_indicator = (alpha > 50) & (hh > -15) & (vv > -15)
        
        # Morphological cleanup
        building_indicator = ndimage.binary_opening(building_indicator, iterations=1)
        building_indicator = ndimage.binary_closing(building_indicator, iterations=2)
        
        labeled, num = label(building_indicator)
        buildings = []
        
        slices = find_objects(labeled)
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            mask = labeled[slc] == (i + 1)
            count = np.sum(mask)
            
            if count < 10 or count > 1000:
                continue
            
            target = self._create_target_from_mask(
                f"BLD_{i+1:04d}", slc, mask,
                hh, vv, hh, None, alpha,
                building_indicator.astype(float), geotransform
            )
            
            if target:
                target.target_type = 'building'
                target.confidence = self._calculate_building_confidence(target)
                buildings.append(target)
        
        return buildings
    
    def detect_camouflage(
        self,
        hh: np.ndarray,
        entropy: np.ndarray,
        h_alpha_class: np.ndarray,
        vegetation_mask: np.ndarray
    ) -> List[DetectedTarget]:
        """
        Detect potential camouflaged targets.
        
        L-band penetrates camouflage nets and vegetation.
        Looks for artificial (low entropy) objects in vegetation.
        
        Args:
            hh: HH backscatter (L-band penetrates)
            entropy: Polarimetric entropy
            h_alpha_class: H-α classification zones
            vegetation_mask: Binary mask of vegetation areas
            
        Returns:
            List of potential camouflaged targets
        """
        # Artificial objects have low entropy in vegetation
        anomaly = (
            vegetation_mask &
            (entropy < 0.4) &  # Low entropy = man-made
            (hh > -20) &       # Bright enough
            (h_alpha_class < 7)  # Not pure vegetation
        )
        
        # Remove isolated pixels
        anomaly = ndimage.binary_opening(anomaly, iterations=1)
        
        labeled, num = label(anomaly)
        camo_targets = []
        
        slices = find_objects(labeled)
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            
            mask = labeled[slc] == (i + 1)
            count = np.sum(mask)
            
            if count < 5 or count > 500:
                continue
            
            target = self._create_target_from_mask(
                f"CAMO_{i+1:04d}", slc, mask,
                hh, hh, hh, entropy, None,
                anomaly.astype(float), None
            )
            
            if target:
                target.target_type = 'camouflage'
                target.confidence = 0.5 + 0.3 * (0.4 - target.entropy)
                camo_targets.append(target)
        
        return camo_targets
    
    def _create_detection_image(
        self,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray
    ) -> np.ndarray:
        """Create optimal detection image from polarimetric data."""
        # For general targets, use maximum of co-pols
        # Ships/vehicles are bright in HH, buildings bright in both HH and VV
        detection = np.maximum(hh, vv)
        
        # Add cross-pol for some targets
        detection = detection + 0.3 * hv
        
        return detection
    
    def _cfar_detector(self, image: np.ndarray) -> np.ndarray:
        """
        Constant False Alarm Rate detector.
        
        Compares each pixel to local background statistics.
        """
        guard = self.config['cfar_guard_size']
        background = self.config['cfar_background_size']
        threshold = self.config['cfar_threshold']
        
        # Replace NaN with median
        valid_mask = ~np.isnan(image)
        if not np.any(valid_mask):
            return np.zeros_like(image, dtype=bool)
        
        image_clean = np.where(valid_mask, image, np.nanmedian(image))
        
        # Calculate local statistics
        # Mean in background window
        kernel_bg = np.ones((2*background+1, 2*background+1))
        kernel_guard = np.ones((2*guard+1, 2*guard+1))
        
        # Subtract guard window from background
        kernel = kernel_bg.copy()
        g_start = background - guard
        g_end = background + guard + 1
        kernel[g_start:g_end, g_start:g_end] = 0
        kernel = kernel / np.sum(kernel)
        
        local_mean = ndimage.convolve(image_clean, kernel, mode='reflect')
        
        # Local variance (simplified)
        local_sq = ndimage.convolve(image_clean**2, kernel, mode='reflect')
        local_var = local_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0.01))
        
        # CFAR threshold
        cfar = (image_clean - local_mean) / local_std
        
        # Detect
        detections = (cfar > threshold) & valid_mask
        
        return detections
    
    def _extract_target_features(
        self,
        target_num: int,
        slc: tuple,
        mask: np.ndarray,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray,
        alpha: np.ndarray,
        cfar: np.ndarray,
        geotransform: tuple,
        land_mask: np.ndarray
    ) -> Optional[DetectedTarget]:
        """Extract features from detected target region."""
        return self._create_target_from_mask(
            f"TGT_{target_num:04d}", slc, mask,
            hh, vv, hv, entropy, alpha, cfar, geotransform
        )
    
    def _create_target_from_mask(
        self,
        target_id: str,
        slc: tuple,
        mask: np.ndarray,
        hh: np.ndarray,
        vv: np.ndarray,
        hv: np.ndarray,
        entropy: np.ndarray,
        alpha: np.ndarray,
        cfar: np.ndarray,
        geotransform: tuple
    ) -> Optional[DetectedTarget]:
        """Create DetectedTarget from mask and data."""
        
        # Get global coordinates
        row_offset = slc[0].start
        col_offset = slc[1].start
        
        # Find target pixels
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None
        
        global_rows = rows + row_offset
        global_cols = cols + col_offset
        
        # Calculate dimensions
        min_row, max_row = global_rows.min(), global_rows.max()
        min_col, max_col = global_cols.min(), global_cols.max()
        
        length_px = max_row - min_row + 1
        width_px = max_col - min_col + 1
        
        length_m = length_px * self.pixel_spacing
        width_m = width_px * self.pixel_spacing
        
        # Center pixel
        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2
        
        # Convert to geographic coordinates
        if geotransform:
            center_lon = geotransform[0] + center_col * geotransform[1]
            center_lat = geotransform[3] + center_row * geotransform[5]
            
            min_lon = geotransform[0] + min_col * geotransform[1]
            max_lon = geotransform[0] + max_col * geotransform[1]
            min_lat = geotransform[3] + max_row * geotransform[5]
            max_lat = geotransform[3] + min_row * geotransform[5]
        else:
            center_lat, center_lon = center_row, center_col
            min_lat, max_lat = min_row, max_row
            min_lon, max_lon = min_col, max_col
        
        # Extract polarimetric features
        hh_roi = hh[slc][mask]
        vv_roi = vv[slc][mask]
        hv_roi = hv[slc][mask]
        
        hh_mean = np.nanmean(hh_roi)
        vv_mean = np.nanmean(vv_roi)
        hv_mean = np.nanmean(hv_roi)
        
        if entropy is not None:
            ent_roi = entropy[slc][mask]
            ent_mean = np.nanmean(ent_roi)
        else:
            ent_mean = 0.5
        
        if alpha is not None:
            alpha_roi = alpha[slc][mask]
            alpha_mean = np.nanmean(alpha_roi)
        else:
            alpha_mean = 45.0
        
        # CFAR value
        cfar_roi = cfar[slc][mask] if cfar is not None else np.ones_like(mask[mask])
        cfar_mean = float(np.nanmean(cfar_roi))
        
        return DetectedTarget(
            target_id=target_id,
            target_type='unknown',
            confidence=0.5,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            bbox={
                'min_lat': float(min_lat),
                'max_lat': float(max_lat),
                'min_lon': float(min_lon),
                'max_lon': float(max_lon)
            },
            length_m=float(length_m),
            width_m=float(width_m),
            area_m2=float(len(rows) * self.pixel_spacing**2),
            hh_mean=float(hh_mean),
            vv_mean=float(vv_mean),
            hv_mean=float(hv_mean),
            entropy=float(ent_mean),
            alpha=float(alpha_mean),
            cfar_value=cfar_mean,
            pixel_count=int(len(rows))
        )
    
    def _classify_targets(self, land_mask: np.ndarray = None):
        """Classify detected targets based on features."""
        for target in self.detected_targets:
            scores = self._calculate_classification_scores(target, land_mask)
            target.classification_scores = scores
            
            # Assign type based on highest score
            if scores:
                best_type = max(scores, key=scores.get)
                best_score = scores[best_type]
                
                if best_score > 0.5:
                    target.target_type = best_type
                    target.confidence = best_score
    
    def _calculate_classification_scores(
        self,
        target: DetectedTarget,
        land_mask: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate classification scores for each target type."""
        scores = {}
        
        # Ship score
        ship_score = 0.0
        if target.length_m >= 20 and target.length_m <= 400:
            ship_score += 0.3
        if target.hh_mean > -15:
            ship_score += 0.2
        if target.entropy < 0.7:
            ship_score += 0.2
        aspect = target.length_m / (target.width_m + 0.1)
        if 2 < aspect < 8:
            ship_score += 0.3
        scores['ship'] = min(ship_score, 1.0)
        
        # Vehicle score
        veh_score = 0.0
        if 3 < target.length_m < 30:
            veh_score += 0.3
        if 2 < target.width_m < 10:
            veh_score += 0.2
        if target.hh_mean > -18:
            veh_score += 0.2
        if target.entropy < 0.5:
            veh_score += 0.3
        scores['vehicle'] = min(veh_score, 1.0)
        
        # Building score
        bld_score = 0.0
        if target.alpha > 50:  # Double-bounce
            bld_score += 0.4
        if target.hh_mean > -12:
            bld_score += 0.2
        if target.entropy < 0.4:
            bld_score += 0.2
        aspect = target.length_m / (target.width_m + 0.1)
        if 0.5 < aspect < 2:
            bld_score += 0.2
        scores['building'] = min(bld_score, 1.0)
        
        return scores
    
    def _calculate_ship_confidence(self, target: DetectedTarget) -> float:
        """Calculate confidence score for ship classification."""
        conf = 0.5
        
        if target.length_m > 50:
            conf += 0.2
        if target.hh_mean > -10:
            conf += 0.15
        if target.entropy < 0.5:
            conf += 0.15
        
        return min(conf, 0.95)
    
    def _calculate_building_confidence(self, target: DetectedTarget) -> float:
        """Calculate confidence score for building classification."""
        conf = 0.5
        
        if target.alpha > 60:
            conf += 0.2
        if target.entropy < 0.3:
            conf += 0.15
        if target.hh_mean > -10 and target.vv_mean > -10:
            conf += 0.15
        
        return min(conf, 0.95)
    
    def export_geojson(self, output_path: Path, targets: List[DetectedTarget] = None):
        """Export detected targets to GeoJSON."""
        if targets is None:
            targets = self.detected_targets
        
        features = [t.to_geojson_feature() for t in targets]
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "detection_time": datetime.utcnow().isoformat(),
                "detector_config": self.config,
                "total_targets": len(targets)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return output_path
    
    def export_by_type(self, output_dir: Path, targets: List[DetectedTarget] = None):
        """Export targets separated by type."""
        if targets is None:
            targets = self.detected_targets
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        by_type = {}
        for t in targets:
            if t.target_type not in by_type:
                by_type[t.target_type] = []
            by_type[t.target_type].append(t)
        
        paths = {}
        for target_type, type_targets in by_type.items():
            path = output_dir / f"{target_type}s.geojson"
            self.export_geojson(path, type_targets)
            paths[target_type] = path
        
        return paths
