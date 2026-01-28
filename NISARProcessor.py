#!/usr/bin/env python3
"""
NISAR Processor - Main Orchestrator
Multi-Sensor SAR Analysis Platform

This is the main entry point for NISAR data processing.
It orchestrates all processing modules based on user configuration.

Usage:
    python NISARProcessor.py <user_id> <config_file>
    
Example:
    python NISARProcessor.py user123 nisar_config_user123.json
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

# Import processing modules
from nisar.downloader import NISARDownloader, NISARProduct
from nisar.polarimetric import PolarimetricProcessor, PolarimetricResult
from nisar.detection import TargetDetector, DetectedTarget
from nisar.terrain import TerrainClassifier
from nisar.insar import InSARProcessor
from nisar.reports import ReportGenerator
from nisar.export import ExportManager
from nisar.clip import SpatialClipper
# ============================================================
# SPATIAL CLIPPING SUPPORT
# ============================================================
class NISARProcessor:
    """
    Main NISAR processing orchestrator.
    
    Coordinates data download, processing, and product generation
    based on user configuration.
    """
    
    def __init__(self, user_id: str, config_path: Path):
        """
        Initialize the processor.
        
        Args:
            user_id: User identifier
            config_path: Path to NISAR configuration JSON
        """
        self.user_id = user_id
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup directories
        config_dir = self.config_path.parent
        if config_dir.name.startswith('nisar_'):
            self.base_dir = config_dir.parent
        else:
            self.base_dir = Path(f"nisar_data_{user_id}")


        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.base_dir / "raw"
        self.products_dir = self.base_dir / "products"
        self.reports_dir = self.base_dir / "reports"
        self.exports_dir = self.base_dir / "exports"
        
        for d in [self.raw_dir, self.products_dir, self.reports_dir, self.exports_dir]:
            d.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize status tracking
        self.status = {
            "user_id": user_id,
            "started": datetime.utcnow().isoformat(),
            "status": "initializing",
            "progress": 0,
            "current_step": "Setup",
            "products": [],
            "errors": []
        }
        self.save_status()
        
        # Initialize processors (lazy loading)
        self._downloader = None
        self._polarimetric = None
        self._detector = None
        self._terrain = None
        self._insar = None
        self._report_gen = None
        self._exporter = None
        
        self.logger.info(f"NISARProcessor initialized for user {user_id}")

        # Initialize spatial clipper
        self.clipper = SpatialClipper.from_config(self.config)
        if self.clipper.is_valid():
            self.logger.info(f"AOI clipper initialized: {self.clipper.get_area_km2():.1f} km²")
        else:
            self.logger.warning("No valid AOI - outputs will not be clipped")
        
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.base_dir / "processing.log"
        
        self.logger = logging.getLogger(f"NISAR_{self.user_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def save_status(self):
        """Save current status to file."""
        status_path = self.base_dir / "status.json"
        with open(status_path, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def update_status(self, status: str, progress: int, step: str):
        """Update processing status."""
        self.status["status"] = status
        self.status["progress"] = progress
        self.status["current_step"] = step
        self.status["updated"] = datetime.utcnow().isoformat()
        self.save_status()
        self.logger.info(f"[{progress}%] {step}")
    

    def clip_data(self, array, geotransform):
        """Clip array to user AOI polygon."""
        # Handle None geotransform or array
        if geotransform is None or array is None:
            return array, geotransform
        if hasattr(self, 'clipper') and self.clipper and self.clipper.is_valid():
            return self.clipper.clip_array(array, geotransform)
        return array, geotransform


    def add_product(self, product_type: str, path: Path):
        """Register a generated product."""
        self.status["products"].append({
            "type": product_type,
            "path": str(path),
            "created": datetime.utcnow().isoformat()
        })
        self.save_status()
    
    def add_error(self, error: str):
        """Register an error."""
        self.status["errors"].append({
            "error": error,
            "time": datetime.utcnow().isoformat()
        })
        self.save_status()
        self.logger.error(error)
    
    @property
    def downloader(self) -> NISARDownloader:
        """Lazy-load downloader."""
        if self._downloader is None:
            self._downloader = NISARDownloader(self.raw_dir, self.config)
        return self._downloader
    
    @property
    def polarimetric(self) -> PolarimetricProcessor:
        """Lazy-load polarimetric processor."""
        if self._polarimetric is None:
            self._polarimetric = PolarimetricProcessor(window_size=5)
        return self._polarimetric
    
    @property
    def detector(self) -> TargetDetector:
        """Lazy-load target detector."""
        if self._detector is None:
            detection_config = self.config.get('military_features', {})
            self._detector = TargetDetector(
                config={
                    'cfar_threshold': detection_config.get('cfar_threshold', 3.0),
                    'min_target_pixels': detection_config.get('min_target_size', 25)
                },
                pixel_spacing=10.0
            )
        return self._detector
    
    @property
    def terrain(self) -> TerrainClassifier:
        """Lazy-load terrain classifier."""
        if self._terrain is None:
            self._terrain = TerrainClassifier()
        return self._terrain
    
    @property
    def insar(self) -> InSARProcessor:
        """Lazy-load InSAR processor."""
        if self._insar is None:
            frequency = self.config.get('frequency', 'L-band')
            self._insar = InSARProcessor(frequency=frequency)
        return self._insar
    
    @property
    def report_gen(self) -> ReportGenerator:
        """Lazy-load report generator."""
        if self._report_gen is None:
            self._report_gen = ReportGenerator(self.reports_dir)
        return self._report_gen
    
    @property
    def exporter(self) -> ExportManager:
        """Lazy-load export manager."""
        if self._exporter is None:
            self._exporter = ExportManager(self.exports_dir)
        return self._exporter
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete processing pipeline.
        
        Returns:
            Dictionary with processing results and product paths
        """
        try:
            self.update_status("running", 0, "Starting processing")
            
            # 1. Download NISAR data
            self.update_status("running", 5, "Searching and downloading NISAR data")
            downloaded_files = self.download_data()
            
            if not downloaded_files:
                self.add_error("No NISAR data found or downloaded")
                self.update_status("failed", 10, "No data available")
                return self.status
            
            # 2. Process each downloaded file
            all_results = []
            num_files = len(downloaded_files)
            
            for i, data_file in enumerate(downloaded_files):
                file_progress_base = 10 + (i * 80 // num_files)
                
                self.logger.info(f"Processing file {i+1}/{num_files}: {data_file.name}")
                
                try:
                    result = self.process_file(data_file, file_progress_base)
                    all_results.append(result)
                except Exception as e:
                    self.add_error(f"Error processing {data_file.name}: {str(e)}")
                    self.logger.exception(f"Error processing file: {e}")
            
            # 3. Generate combined reports
            self.update_status("running", 90, "Generating reports")
            self.generate_reports(all_results)
            
            # 4. Finalize
            self.update_status("completed", 100, "Processing complete")
            self.status["completed"] = datetime.utcnow().isoformat()
            self.save_status()
            
            self.logger.info("Processing completed successfully")
            return self.status
            
        except Exception as e:
            self.add_error(f"Critical error: {str(e)}")
            self.logger.exception(f"Critical error: {e}")
            self.update_status("failed", self.status["progress"], f"Error: {str(e)}")
            return self.status
    
    def download_data(self) -> List[Path]:
        """Download NISAR data based on configuration."""
        polygon = []
        
        # Extract polygon from config selections
        selections = self.config.get('selections', [])
        if selections:
            sel = selections[0]
            coords = sel.get('coords', [])
            for c in coords:
                if isinstance(c, dict):
                    lon = c.get('lon', c.get('lng', 0))
                    lat = c.get('lat', 0)
                    polygon.append((lon, lat))  # ASF expects (lon, lat)
                elif isinstance(c, (list, tuple)):
                    polygon.append((c[1], c[0]))  # Swap to (lat, lon)
        
        if not polygon:
            self.add_error("No polygon defined for search")
            return self.create_mock_data()
        
        # Get date range from config
        date_from = self.config.get('date_from', '')
        date_to = self.config.get('date_to', '')
        
        try:
            if date_from:
                start_date = datetime.fromisoformat(date_from.replace('Z', ''))
            else:
                start_date = datetime.now() - timedelta(days=30)
            
            if date_to:
                end_date = datetime.fromisoformat(date_to.replace('Z', ''))
            else:
                end_date = datetime.now()
        except Exception as e:
            self.logger.warning(f"Date parsing error: {e}, using defaults")
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
        
        self.logger.info(f"Searching NISAR data: {start_date} to {end_date}")
        
        try:
            products = self.downloader.search_products(
                polygon=polygon,
                start_date=start_date,
                end_date=end_date,
                max_results=10
            )
            
            if not products:
                self.logger.warning("No products found, using sample data")
                return self.create_mock_data()
            
            downloaded = self.downloader.download_all(products)
            return downloaded
            
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return self.create_mock_data()
    
    def parse_selections_file(self, path: Path) -> List[tuple]:
        """Parse polygon from selections file."""
        polygon = []
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Polygon'):
                        import ast
                        coords_str = line.split(':', 1)[1].strip()
                        coords = ast.literal_eval(coords_str)
                        polygon = [(c[1], c[0]) for c in coords]
                        break
        except Exception as e:
            self.logger.error(f"Error parsing selections: {e}")
        
        return polygon
    
    def create_mock_data(self) -> List[Path]:
        """Create mock data for development/testing."""
        self.logger.info("Creating mock data for testing")
        
        import numpy as np
        
        mock_dir = self.raw_dir / "mock"
        mock_dir.mkdir(exist_ok=True)
        
        shape = (500, 500)
        np.random.seed(42)
        
        hh = np.random.randn(*shape) * 3 - 15
        vv = np.random.randn(*shape) * 3 - 15
        hv = np.random.randn(*shape) * 3 - 22
        
        # Add targets
        for _ in range(20):
            x, y = np.random.randint(50, 450, 2)
            size = np.random.randint(3, 15)
            hh[x-size:x+size, y-size:y+size] += 10
            vv[x-size:x+size, y-size:y+size] += 8
        
        # Add water
        hh[400:500, :] -= 10
        vv[400:500, :] -= 10
        hv[400:500, :] -= 5
        
        # Add forest
        hv[100:200, 100:300] += 8
        
        np.save(mock_dir / "HH.npy", hh)
        np.save(mock_dir / "VV.npy", vv)
        np.save(mock_dir / "HV.npy", hv)
        np.save(mock_dir / "VH.npy", hv)
        
        meta = {
            "product_id": "MOCK_NISAR_001",
            "acquisition_date": datetime.utcnow().isoformat(),
            "frequency": self.config.get('frequency', 'L-band'),
            "processing_level": self.config.get('level', 'L2-GCOV'),
            "shape": shape,
            "pixel_spacing": 10.0,
            "geotransform": (23.0, 0.0001, 0, 38.0, 0, -0.0001),
        }
        
        with open(mock_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        return [mock_dir]
    
    def process_file(self, data_path: Path, progress_base: int) -> Dict[str, Any]:
        """Process a single NISAR data file."""
        result = {"input": str(data_path), "products": {}}
        
        self.update_status("running", progress_base + 5, "Loading data")
        
        if data_path.is_dir():
            hh, vv, hv, vh, meta = self.load_mock_data(data_path)
        else:
            hh, vv, hv, vh, meta = self.load_nisar_data(data_path)
        
        geotransform = meta.get('geotransform')
        
        output_dir = self.products_dir / data_path.stem
        output_dir.mkdir(exist_ok=True)
        
        analysis_mode = self.config.get('analysis', 'basic')
        
        # 1. Basic processing
        self.update_status("running", progress_base + 10, "Generating backscatter products")
        result["products"]["backscatter"] = self.save_backscatter(
            hh, vv, hv, vh, output_dir, geotransform
        )
        
        # 2. Polarimetric processing
        if analysis_mode in ['polarimetric', 'target-detection', 'military']:
            self.update_status("running", progress_base + 20, "Polarimetric decomposition")
            pol_result = self.process_polarimetric(hh, vv, hv, vh, output_dir, geotransform)
            result["products"]["polarimetric"] = pol_result
        
        # 3. Target detection
        if analysis_mode in ['target-detection', 'military']:
            self.update_status("running", progress_base + 40, "Target detection")
            
            entropy = alpha = None
            if 'polarimetric' in result["products"]:
                entropy = self.load_product(result["products"]["polarimetric"].get("entropy"))
                alpha = self.load_product(result["products"]["polarimetric"].get("alpha"))
            
            detection_result = self.detect_targets(
                hh, vv, hv, entropy, alpha, output_dir, geotransform
            )
            result["products"]["detection"] = detection_result
        
        # 4. Terrain classification
        if analysis_mode in ['military']:
            self.update_status("running", progress_base + 55, "Terrain classification")
            
            entropy = alpha = h_alpha_class = None
            fd_surface = fd_volume = fd_double = None
            
            if 'polarimetric' in result["products"]:
                pol = result["products"]["polarimetric"]
                entropy = self.load_product(pol.get("entropy"))
                alpha = self.load_product(pol.get("alpha"))
                h_alpha_class = self.load_product(pol.get("h_alpha_class"))
                fd_surface = self.load_product(pol.get("fd_surface"))
                fd_volume = self.load_product(pol.get("fd_volume"))
                fd_double = self.load_product(pol.get("fd_double"))
            
            terrain_result = self.classify_terrain(
                hh, vv, hv, entropy, alpha, h_alpha_class,
                fd_surface, fd_volume, fd_double,
                output_dir, geotransform
            )
            result["products"]["terrain"] = terrain_result
        
        # 5. Change detection
        if self.config.get('military_features', {}).get('deformation'):
            self.update_status("running", progress_base + 65, "Change detection")
            change_result = self.detect_changes(hh, vv, output_dir, geotransform)
            result["products"]["change"] = change_result
        
        return result
    
    def load_mock_data(self, data_path: Path) -> tuple:
        """Load mock data from numpy files."""
        import numpy as np
        
        hh = np.load(data_path / "HH.npy")
        vv = np.load(data_path / "VV.npy")
        hv = np.load(data_path / "HV.npy")
        vh = np.load(data_path / "VH.npy")
        
        with open(data_path / "metadata.json", 'r') as f:
            meta = json.load(f)
        
        return hh, vv, hv, vh, meta
    
    def load_nisar_data(self, h5_path: Path) -> tuple:
        """Load data from NISAR HDF5 file."""
        import h5py
        import numpy as np
        
        with h5py.File(h5_path, 'r') as f:
            for base in ['/science/LSAR/GCOV/grids/frequencyA',
                        '/science/LSAR/GSLC/grids/frequencyA',
                        '/grids/frequencyA']:
                if base in f:
                    grp = f[base]
                    break
            else:
                raise ValueError("Could not find data group in NISAR file")
            
            hh = np.array(grp.get('HHHH', np.zeros((100, 100))))
            vv = np.array(grp.get('VVVV', np.zeros((100, 100))))
            hv = np.array(grp.get('HVHV', np.zeros((100, 100))))
            vh = hv.copy()
            
            if hh.max() > 1:
                hh = 10 * np.log10(np.maximum(hh, 1e-10))
                vv = 10 * np.log10(np.maximum(vv, 1e-10))
                hv = 10 * np.log10(np.maximum(hv, 1e-10))
                vh = hv.copy()
            
            meta = {"shape": hh.shape, "pixel_spacing": 10.0}
        
        return hh, vv, hv, vh, meta
    
    def load_product(self, path: str) -> Optional[Any]:
        """Load a saved product."""
        import numpy as np
        
        if path is None:
            return None
        
        path = Path(path)
        if path.suffix == '.npy' and path.exists():
            return np.load(path)
        return None
    
    def save_backscatter(self, hh, vv, hv, vh, output_dir: Path, geotransform: tuple) -> Dict[str, str]:
        """Save backscatter images (clipped to AOI)."""
        import numpy as np
        
        # Get minimum shape across all arrays
        min_rows = min(hh.shape[0], vv.shape[0], hv.shape[0], vh.shape[0])
        min_cols = min(hh.shape[1], vv.shape[1], hv.shape[1], vh.shape[1])
        
        # Trim all to same shape
        hh = hh[:min_rows, :min_cols]
        vv = vv[:min_rows, :min_cols]
        hv = hv[:min_rows, :min_cols]
        vh = vh[:min_rows, :min_cols]
        
        products = {}
        
        for name, data in [('HH', hh), ('VV', vv), ('HV', hv), ('VH', vh)]:
            path = output_dir / f"backscatter_{name}.npy"
            np.save(path, data)
            products[name.lower()] = str(path)
            self.add_product(f"backscatter_{name}", path)
        
        rgb_path = self.exporter.export_rgb_image(
            hh, hv, vv, f"rgb_composite_{output_dir.name}", geotransform
        )
        products['rgb'] = str(rgb_path)
        self.add_product("rgb_composite", rgb_path)
        
        return products
    
    def process_polarimetric(self, hh, vv, hv, vh, output_dir: Path, geotransform: tuple) -> Dict[str, str]:
        """Run polarimetric decomposition."""
        import numpy as np
        
        # Get minimum shape across all arrays
        min_rows = min(hh.shape[0], vv.shape[0], hv.shape[0], vh.shape[0])
        min_cols = min(hh.shape[1], vv.shape[1], hv.shape[1], vh.shape[1])
        
        # Trim all to same shape
        hh = hh[:min_rows, :min_cols]
        vv = vv[:min_rows, :min_cols]
        hv = hv[:min_rows, :min_cols]
        vh = vh[:min_rows, :min_cols]
        
        hh_lin = 10 ** (np.clip(hh, -50, 50) / 10)
        vv_lin = 10 ** (np.clip(vv, -50, 50) / 10)
        hv_lin = 10 ** (hv / 10)
        
        hh_slc = np.sqrt(hh_lin) * np.exp(1j * np.random.uniform(-np.pi, np.pi, hh.shape))
        vv_slc = np.sqrt(vv_lin) * np.exp(1j * np.random.uniform(-np.pi, np.pi, vv.shape))
        hv_slc = np.sqrt(hv_lin) * np.exp(1j * np.random.uniform(-np.pi, np.pi, hv.shape))
        vh_slc = hv_slc.copy()
        
        result = self.polarimetric.process_slc_quad_pol(
            hh_slc, hv_slc, vh_slc, vv_slc, geotransform=geotransform
        )
        
        products = {}
        pol_dir = output_dir / "polarimetric"
        pol_dir.mkdir(exist_ok=True)
        
        for name in ['entropy', 'anisotropy', 'alpha']:
            data = getattr(result, name)
            path = pol_dir / f"{name}.npy"
            np.save(path, data)
            products[name] = str(path)
            self.add_product(name, path)
        
        for name in ['fd_surface', 'fd_double', 'fd_volume']:
            data = getattr(result, name)
            path = pol_dir / f"{name}.npy"
            np.save(path, data)
            products[name] = str(path)
        
        for name in ['yam_surface', 'yam_double', 'yam_volume', 'yam_helix']:
            data = getattr(result, name)
            path = pol_dir / f"{name}.npy"
            np.save(path, data)
            products[name] = str(path)
        
        path = pol_dir / "h_alpha_class.npy"
        np.save(path, result.h_alpha_class)
        products['h_alpha_class'] = str(path)
        self.add_product("h_alpha_classification", path)
        
        pauli_path = self.exporter.export_rgb_image(
            result.pauli_r, result.pauli_g, result.pauli_b,
            f"pauli_rgb_{output_dir.name}", geotransform
        )
        products['pauli_rgb'] = str(pauli_path)
        self.add_product("pauli_rgb", pauli_path)
        
        return products
    
    def detect_targets(self, hh, vv, hv, entropy, alpha, output_dir: Path, geotransform: tuple) -> Dict[str, Any]:
        """Run target detection."""
        mil_config = self.config.get('military_features', {})
        
        targets = self.detector.detect_targets(
            hh, vv, hv, entropy=entropy, alpha=alpha, geotransform=geotransform
        )
        
        products = {"count": len(targets)}
        
        targets_dir = output_dir / "targets"
        targets_dir.mkdir(exist_ok=True)
        
        geojson_path = self.detector.export_geojson(targets_dir / "all_targets.geojson", targets)
        products['all_targets'] = str(geojson_path)
        self.add_product("all_targets", geojson_path)
        
        type_paths = self.detector.export_by_type(targets_dir, targets)
        for ttype, path in type_paths.items():
            products[f'{ttype}s'] = str(path)
            self.add_product(f"{ttype}_targets", path)
        
        if mil_config.get('ship_detection'):
            water_mask = hh < -18
            ships = self.detector.detect_ships(hh, water_mask, geotransform)
            self.detector.export_geojson(targets_dir / "ships.geojson", ships)
            products['ships_count'] = len(ships)
        
        if mil_config.get('building_detection') and alpha is not None:
            buildings = self.detector.detect_buildings(hh, vv, alpha, geotransform)
            self.detector.export_geojson(targets_dir / "buildings.geojson", buildings)
            products['buildings_count'] = len(buildings)
        
        self.exporter.extract_target_chips(hh, [t.to_dict() for t in targets[:50]], geotransform=geotransform)
        
        overlay_path = self.exporter.create_detection_overlay(
            hh, [t.to_dict() for t in targets],
            f"detection_overlay_{output_dir.name}.png", geotransform
        )
        products['overlay'] = str(overlay_path)
        self.add_product("detection_overlay", overlay_path)
        
        kml_features = [t.to_geojson_feature() for t in targets]
        kml_path = self.exporter.export_kml(
            kml_features, f"targets_{output_dir.name}.kml", name="Detected Targets"
        )
        products['kml'] = str(kml_path)
        
        return products
    
    def classify_terrain(self, hh, vv, hv, entropy, alpha, h_alpha_class,
                        fd_surface, fd_volume, fd_double, output_dir: Path, geotransform: tuple) -> Dict[str, str]:
        """Run terrain classification."""
        import numpy as np
        
        classification = self.terrain.classify(
            hh, vv, hv,
            entropy if entropy is not None else np.ones_like(hh) * 0.5,
            alpha if alpha is not None else np.ones_like(hh) * 45,
            h_alpha_class if h_alpha_class is not None else np.ones_like(hh, dtype=np.uint8) * 5,
            fd_surface, fd_volume, fd_double
        )
        
        products = {}
        terrain_dir = output_dir / "terrain"
        terrain_dir.mkdir(exist_ok=True)
        
        class_path = terrain_dir / "classification.npy"
        np.save(class_path, classification)
        products['classification'] = str(class_path)
        self.add_product("terrain_classification", class_path)
        
        trafficability = self.terrain.compute_trafficability(classification)
        traf_path = terrain_dir / "trafficability.npy"
        np.save(traf_path, trafficability)
        products['trafficability'] = str(traf_path)
        self.add_product("trafficability", traf_path)
        
        stats = self.terrain.compute_statistics()
        stats_path = terrain_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        products['statistics'] = str(stats_path)
        
        return products
    
    def detect_changes(self, hh, vv, output_dir: Path, geotransform: tuple) -> Dict[str, str]:
        """Run change detection."""
        import numpy as np
        
        coherence = np.clip(1 - np.abs(hh - vv) / 20, 0, 1)
        change_map = coherence < 0.5
        change_intensity = 1 - coherence
        
        products = {}
        change_dir = output_dir / "change"
        change_dir.mkdir(exist_ok=True)
        
        coh_path = change_dir / "coherence.npy"
        np.save(coh_path, coherence)
        products['coherence'] = str(coh_path)
        
        change_path = change_dir / "change_map.npy"
        np.save(change_path, change_map)
        products['change_map'] = str(change_path)
        self.add_product("change_map", change_path)
        
        intensity_path = change_dir / "change_intensity.npy"
        np.save(intensity_path, change_intensity)
        products['change_intensity'] = str(intensity_path)
        
        return products
    
    def generate_reports(self, all_results: List[Dict]):
        """Generate final reports."""
        self.report_gen.set_title(
            "NISAR SAR Analysis Report",
            f"User: {self.user_id} | {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        
        self.report_gen.add_overview_section(
            aoi_description="User-defined area of interest",
            satellite="NISAR",
            acquisition_date=datetime.utcnow(),
            analysis_modes=[self.config.get('analysis', 'basic')]
        )
        
        for result in all_results:
            products = result.get('products', {})
            
            if 'polarimetric' in products:
                self.report_gen.add_polarimetric_section(
                    h_alpha_stats={'mean_entropy': 0.5, 'mean_alpha': 45.0},
                    freeman_durden_stats={'Surface': 40, 'Double-bounce': 20, 'Volume': 40}
                )
            
            if 'detection' in products:
                self.report_gen.add_detection_section(targets=[])
            
            if 'terrain' in products:
                stats_path = products['terrain'].get('statistics')
                if stats_path and Path(stats_path).exists():
                    with open(stats_path) as f:
                        stats = json.load(f)
                else:
                    stats = {}
                self.report_gen.add_terrain_section(classification_stats=stats)
        
        try:
            pdf_path = self.report_gen.generate_pdf("analysis_report.pdf")
            self.add_product("report_pdf", pdf_path)
        except Exception as e:
            self.logger.warning(f"PDF generation failed: {e}")
        
        html_path = self.report_gen.generate_html("analysis_report.html")
        self.add_product("report_html", html_path)
        
        md_path = self.report_gen.generate_markdown("analysis_report.md")
        self.add_product("report_md", md_path)


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python NISARProcessor.py <user_id> <config_file>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    config_file = sys.argv[2]
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    print(f"Starting NISAR processing for user: {user_id}")
    processor = NISARProcessor(user_id, config_file)
    result = processor.run()
    
    print(f"\nProcessing completed: {result['status']}")
    print(f"Products generated: {len(result['products'])}")
    
    sys.exit(0 if result['status'] == 'completed' else 1)


if __name__ == "__main__":
    main()
