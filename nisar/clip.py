"""
Spatial Clipping Module for NISAR Processing
"""
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from shapely.geometry import Polygon, mapping
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class SpatialClipper:
    """Clips raster data to a polygon AOI."""
    
    def __init__(self, polygon_coords: List[Tuple[float, float]]):
        self.coords = polygon_coords
        self.polygon = None
        self.bounds = None
        if SHAPELY_AVAILABLE and polygon_coords and len(polygon_coords) >= 3:
            self._create_polygon()
    
    def _create_polygon(self):
        coords = list(self.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            self.polygon = Polygon(coords)
            if not self.polygon.is_valid:
                self.polygon = self.polygon.buffer(0)
            self.bounds = self.polygon.bounds
        except Exception as e:
            print(f"Warning: Could not create polygon: {e}")
    
    @classmethod
    def from_config(cls, config: dict) -> 'SpatialClipper':
        coords = []
        selections = config.get("selections", [])
        if selections:
            sel = selections[0]
            if "coords" in sel:
                for c in sel["coords"]:
                    if isinstance(c, dict):
                        lon = c.get("lon", c.get("lng", 0))
                        lat = c.get("lat", 0)
                        coords.append((lon, lat))
                    elif isinstance(c, (list, tuple)):
                        coords.append((c[0], c[1]))
        return cls(coords)
    
    def is_valid(self) -> bool:
        return self.polygon is not None and self.polygon.is_valid
    
    def get_pixel_bounds(self, geotransform: tuple, raster_shape: tuple) -> Tuple[int, int, int, int]:
        if not self.bounds:
            return (0, raster_shape[0], 0, raster_shape[1])
        minx, miny, maxx, maxy = self.bounds
        origin_x, pixel_width = geotransform[0], geotransform[1]
        origin_y, pixel_height = geotransform[3], geotransform[5]
        col_start = int((minx - origin_x) / pixel_width)
        col_end = int((maxx - origin_x) / pixel_width)
        row_start = int((maxy - origin_y) / pixel_height)
        row_end = int((miny - origin_y) / pixel_height)
        height, width = raster_shape
        col_start = max(0, min(col_start, width))
        col_end = max(0, min(col_end, width))
        row_start = max(0, min(row_start, height))
        row_end = max(0, min(row_end, height))
        if col_start > col_end: col_start, col_end = col_end, col_start
        if row_start > row_end: row_start, row_end = row_end, row_start
        return (row_start, row_end, col_start, col_end)
    
    def clip_array(self, array: np.ndarray, geotransform: tuple) -> Tuple[np.ndarray, tuple]:
        if not self.is_valid() or array is None:
            return array, geotransform
        row_start, row_end, col_start, col_end = self.get_pixel_bounds(geotransform, array.shape)
        if row_end <= row_start or col_end <= col_start:
            return array, geotransform
        clipped = array[row_start:row_end, col_start:col_end].copy()
        new_gt = (
            geotransform[0] + col_start * geotransform[1],
            geotransform[1], geotransform[2],
            geotransform[3] + row_start * geotransform[5],
            geotransform[4], geotransform[5]
        )
        return clipped, new_gt
    
    def get_area_km2(self) -> float:
        if not self.is_valid():
            return 0.0
        minx, miny, maxx, maxy = self.bounds
        center_lat = (miny + maxy) / 2
        return self.polygon.area * 111.0 * 111.0 * np.cos(np.radians(center_lat))
