"""
Export Module
Export analysis results to various formats

Supports:
- GeoJSON export
- KML/KMZ export
- GeoTIFF export
- Target chip extraction
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import zipfile
from io import BytesIO
import base64


class ExportManager:
    """
    Export analysis results to various GIS formats.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize export manager.
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        
        self.chips_dir = self.output_dir / "chips"
        self.chips_dir.mkdir(exist_ok=True)
    
    def export_geojson(
        self,
        features: List[Dict],
        filename: str,
        properties: Dict = None
    ) -> Path:
        """
        Export features to GeoJSON.
        
        Args:
            features: List of GeoJSON feature dicts
            filename: Output filename
            properties: Additional properties for FeatureCollection
            
        Returns:
            Path to exported file
        """
        if not filename.endswith('.geojson'):
            filename += '.geojson'
        
        output_path = self.exports_dir / filename
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": properties or {}
        }
        
        # Add metadata
        geojson["properties"]["export_time"] = datetime.utcnow().isoformat()
        geojson["properties"]["feature_count"] = len(features)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        return output_path
    
    def export_kml(
        self,
        features: List[Dict],
        filename: str,
        name: str = "SAR Analysis Results",
        description: str = ""
    ) -> Path:
        """
        Export features to KML.
        
        Args:
            features: List of feature dicts with geometry and properties
            filename: Output filename
            name: KML document name
            description: KML document description
            
        Returns:
            Path to exported KML file
        """
        if not filename.endswith('.kml'):
            filename += '.kml'
        
        output_path = self.exports_dir / filename
        
        # Build KML
        kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <description>{description}</description>
'''
        
        # Define styles
        kml += self._kml_styles()
        
        # Add features
        for feature in features:
            kml += self._feature_to_kml_placemark(feature)
        
        kml += '''  </Document>
</kml>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(kml)
        
        return output_path
    
    def export_kmz(
        self,
        features: List[Dict],
        filename: str,
        name: str = "SAR Analysis Results",
        images: List[Path] = None
    ) -> Path:
        """
        Export features to KMZ (zipped KML with images).
        
        Args:
            features: List of feature dicts
            filename: Output filename
            name: KML document name
            images: List of image paths to include
            
        Returns:
            Path to exported KMZ file
        """
        if not filename.endswith('.kmz'):
            filename += '.kmz'
        
        output_path = self.exports_dir / filename
        
        # Create KML content
        kml_content = self._create_kml_content(features, name, images)
        
        # Create KMZ (zip file)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
            # Add KML
            kmz.writestr('doc.kml', kml_content)
            
            # Add images
            if images:
                for img_path in images:
                    if img_path.exists():
                        kmz.write(img_path, f'images/{img_path.name}')
        
        return output_path
    
    def _kml_styles(self) -> str:
        """Generate KML styles for different target types."""
        return '''
    <Style id="ship">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/sailing.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="vehicle">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.0</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/truck.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="building">
      <IconStyle>
        <color>ffff0000</color>
        <scale>1.0</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="unknown">
      <IconStyle>
        <color>ffffffff</color>
        <scale>0.8</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>
      </IconStyle>
    </Style>
    <Style id="polygon">
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
      <PolyStyle>
        <color>400000ff</color>
      </PolyStyle>
    </Style>
'''
    
    def _feature_to_kml_placemark(self, feature: Dict) -> str:
        """Convert a GeoJSON feature to KML Placemark."""
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        geom_type = geom.get('type', 'Point')
        coords = geom.get('coordinates', [0, 0])
        
        # Determine style
        target_type = props.get('target_type', 'unknown')
        style_url = f"#{target_type}"
        
        # Build description
        desc_parts = []
        for key, value in props.items():
            if key not in ['target_id', 'target_type']:
                desc_parts.append(f"<b>{key}:</b> {value}")
        description = "<br/>".join(desc_parts)
        
        name = props.get('target_id', 'Feature')
        
        if geom_type == 'Point':
            lon, lat = coords
            return f'''
    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{description}]]></description>
      <styleUrl>{style_url}</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
'''
        elif geom_type == 'Polygon':
            # coords is [[[lon, lat], ...]]
            coord_str = ' '.join([f"{c[0]},{c[1]},0" for c in coords[0]])
            return f'''
    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{description}]]></description>
      <styleUrl>#polygon</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coord_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
'''
        
        return ''
    
    def _create_kml_content(
        self,
        features: List[Dict],
        name: str,
        images: List[Path] = None
    ) -> str:
        """Create full KML document content."""
        kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
'''
        
        kml += self._kml_styles()
        
        # Add ground overlays for images
        if images:
            for img_path in images:
                # Would need bounds for proper overlay
                pass
        
        for feature in features:
            kml += self._feature_to_kml_placemark(feature)
        
        kml += '''  </Document>
</kml>'''
        
        return kml
    
    def export_geotiff(
        self,
        data: np.ndarray,
        filename: str,
        geotransform: tuple = None,
        projection: str = None,
        nodata: float = None,
        dtype: str = 'float32'
    ) -> Path:
        """
        Export array as GeoTIFF.
        
        Args:
            data: 2D or 3D numpy array
            filename: Output filename
            geotransform: GDAL geotransform (x_min, x_res, 0, y_max, 0, -y_res)
            projection: WKT or EPSG string
            nodata: NoData value
            dtype: Output data type
            
        Returns:
            Path to exported GeoTIFF
        """
        if not filename.endswith('.tif') and not filename.endswith('.tiff'):
            filename += '.tif'
        
        output_path = self.exports_dir / filename
        
        try:
            from osgeo import gdal, osr
            
            # Handle dimensions
            if data.ndim == 2:
                height, width = data.shape
                bands = 1
                data = data[np.newaxis, :, :]
            else:
                bands, height, width = data.shape
            
            # Create driver
            driver = gdal.GetDriverByName('GTiff')
            
            # Map dtype
            gdal_dtype = {
                'uint8': gdal.GDT_Byte,
                'int16': gdal.GDT_Int16,
                'uint16': gdal.GDT_UInt16,
                'int32': gdal.GDT_Int32,
                'uint32': gdal.GDT_UInt32,
                'float32': gdal.GDT_Float32,
                'float64': gdal.GDT_Float64,
            }.get(dtype, gdal.GDT_Float32)
            
            # Create dataset
            ds = driver.Create(
                str(output_path), width, height, bands, gdal_dtype,
                options=['COMPRESS=LZW', 'TILED=YES']
            )
            
            # Set geotransform
            if geotransform:
                ds.SetGeoTransform(geotransform)
            
            # Set projection
            if projection:
                if projection.startswith('EPSG:'):
                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(int(projection.split(':')[1]))
                    ds.SetProjection(srs.ExportToWkt())
                else:
                    ds.SetProjection(projection)
            
            # Write bands
            for i in range(bands):
                band = ds.GetRasterBand(i + 1)
                band.WriteArray(data[i])
                if nodata is not None:
                    band.SetNoDataValue(nodata)
                band.FlushCache()
            
            ds = None  # Close dataset
            
        except ImportError:
            # Fall back to numpy save if GDAL not available
            np.save(output_path.with_suffix('.npy'), data)
            
            # Save metadata
            meta = {
                'geotransform': geotransform,
                'projection': projection,
                'nodata': nodata,
                'shape': data.shape
            }
            with open(output_path.with_suffix('.meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
        
        return output_path
    
    def export_rgb_image(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        filename: str,
        geotransform: tuple = None,
        projection: str = None,
        normalize: bool = True
    ) -> Path:
        """
        Export RGB composite image.
        
        Args:
            r, g, b: Red, green, blue channel arrays
            filename: Output filename
            geotransform: GDAL geotransform
            projection: Projection string
            normalize: Whether to normalize to 0-255
            
        Returns:
            Path to exported image
        """
        if normalize:
            def norm(arr):
                arr = arr.copy()
                p2, p98 = np.nanpercentile(arr, [2, 98])
                arr = np.clip((arr - p2) / (p98 - p2 + 1e-10), 0, 1)
                return (arr * 255).astype(np.uint8)
            
            r = norm(r)
            g = norm(g)
            b = norm(b)
        
        rgb = np.stack([r, g, b], axis=0)
        
        return self.export_geotiff(
            rgb, filename, geotransform, projection, dtype='uint8'
        )
    
    def extract_target_chips(
        self,
        image: np.ndarray,
        targets: List[Dict],
        chip_size: int = 64,
        geotransform: tuple = None
    ) -> List[Path]:
        """
        Extract image chips around detected targets.
        
        Args:
            image: Source image array
            targets: List of target dicts with location info
            chip_size: Size of chip in pixels
            geotransform: For coordinate conversion
            
        Returns:
            List of paths to saved chips
        """
        chip_paths = []
        half_size = chip_size // 2
        
        for target in targets:
            target_id = target.get('target_id', 'unknown')
            
            # Get pixel coordinates
            if geotransform:
                lon = target.get('center_lon', 0)
                lat = target.get('center_lat', 0)
                col = int((lon - geotransform[0]) / geotransform[1])
                row = int((lat - geotransform[3]) / geotransform[5])
            else:
                row = int(target.get('center_lat', 0))
                col = int(target.get('center_lon', 0))
            
            # Extract chip with boundary checking
            row_start = max(0, row - half_size)
            row_end = min(image.shape[0], row + half_size)
            col_start = max(0, col - half_size)
            col_end = min(image.shape[1], col + half_size)
            
            chip = image[row_start:row_end, col_start:col_end]
            
            if chip.size == 0:
                continue
            
            # Save chip
            chip_path = self.chips_dir / f"{target_id}.png"
            
            try:
                from PIL import Image
                
                # Normalize to 0-255
                if chip.dtype != np.uint8:
                    p2, p98 = np.nanpercentile(chip, [2, 98])
                    chip = np.clip((chip - p2) / (p98 - p2 + 1e-10), 0, 1)
                    chip = (chip * 255).astype(np.uint8)
                
                img = Image.fromarray(chip)
                img.save(chip_path)
                
            except ImportError:
                # Fall back to numpy
                np.save(chip_path.with_suffix('.npy'), chip)
            
            chip_paths.append(chip_path)
            
            # Update target with chip path
            target['chip_path'] = str(chip_path)
        
        return chip_paths
    
    def create_detection_overlay(
        self,
        base_image: np.ndarray,
        targets: List[Dict],
        filename: str = "detection_overlay.png",
        geotransform: tuple = None
    ) -> Path:
        """
        Create image with target detection overlay.
        
        Args:
            base_image: Background image
            targets: List of detected targets
            filename: Output filename
            geotransform: For coordinate conversion
            
        Returns:
            Path to overlay image
        """
        output_path = self.exports_dir / filename
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Normalize base image
            if base_image.dtype != np.uint8:
                p2, p98 = np.nanpercentile(base_image, [2, 98])
                base = np.clip((base_image - p2) / (p98 - p2 + 1e-10), 0, 1)
                base = (base * 255).astype(np.uint8)
            else:
                base = base_image.copy()
            
            # Convert to RGB if grayscale
            if base.ndim == 2:
                base = np.stack([base, base, base], axis=-1)
            
            img = Image.fromarray(base)
            draw = ImageDraw.Draw(img)
            
            # Color by type
            colors = {
                'ship': (255, 0, 0),      # Red
                'vehicle': (0, 255, 0),   # Green
                'building': (0, 0, 255),  # Blue
                'camouflage': (255, 255, 0),  # Yellow
                'unknown': (255, 255, 255)  # White
            }
            
            for target in targets:
                target_type = target.get('target_type', 'unknown')
                color = colors.get(target_type, (255, 255, 255))
                
                # Get pixel coordinates
                if geotransform:
                    lon = target.get('center_lon', 0)
                    lat = target.get('center_lat', 0)
                    col = int((lon - geotransform[0]) / geotransform[1])
                    row = int((lat - geotransform[3]) / geotransform[5])
                else:
                    row = int(target.get('center_lat', 0))
                    col = int(target.get('center_lon', 0))
                
                # Draw bounding box
                length = max(5, int(target.get('length_m', 10) / 10))  # Approximate
                width = max(5, int(target.get('width_m', 10) / 10))
                
                bbox = [
                    col - width//2, row - length//2,
                    col + width//2, row + length//2
                ]
                
                draw.rectangle(bbox, outline=color, width=2)
                
                # Draw label
                label = f"{target.get('target_id', '')}"
                draw.text((bbox[0], bbox[1] - 12), label, fill=color)
            
            img.save(output_path)
            
        except ImportError:
            # Fall back to simple numpy save
            np.save(output_path.with_suffix('.npy'), base_image)
        
        return output_path
    
    def export_all_formats(
        self,
        targets: List[Dict],
        base_name: str = "targets"
    ) -> Dict[str, Path]:
        """
        Export targets to all supported formats.
        
        Args:
            targets: List of target feature dicts
            base_name: Base filename
            
        Returns:
            Dict of format -> path
        """
        paths = {}
        
        # Convert to GeoJSON features if not already
        features = []
        for t in targets:
            if 'type' in t and t['type'] == 'Feature':
                features.append(t)
            elif 'to_geojson_feature' in dir(t):
                features.append(t.to_geojson_feature())
            else:
                # Assume it's a dict with lat/lon
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [t.get('center_lon', 0), t.get('center_lat', 0)]
                    },
                    "properties": t
                })
        
        # Export GeoJSON
        paths['geojson'] = self.export_geojson(features, f"{base_name}.geojson")
        
        # Export KML
        paths['kml'] = self.export_kml(features, f"{base_name}.kml")
        
        # Export KMZ
        paths['kmz'] = self.export_kmz(features, f"{base_name}.kmz")
        
        return paths
