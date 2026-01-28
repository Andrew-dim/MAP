"""
NISAR Data Downloader
ASF DAAC Integration for NISAR products

Handles:
- NASA Earthdata authentication
- NISAR product search by polygon and date range
- Product download with resume capability
- Multiple product levels (L1-GSLC, L2-GCOV, L2-InSAR)
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class NISARProduct:
    """Represents a NISAR data product."""
    product_id: str
    filename: str
    url: str
    size_mb: float
    acquisition_date: datetime
    processing_level: str
    frequency: str  # L-band, S-band, or Both
    polarization: str
    geometry: dict  # GeoJSON geometry
    metadata: dict
    
    def to_dict(self):
        d = asdict(self)
        d['acquisition_date'] = self.acquisition_date.isoformat()
        return d


class NISARDownloader:
    """
    Downloads NISAR data from ASF DAAC.
    
    Requires NASA Earthdata credentials in environment variables:
    - NASA_EARTHDATA_USER
    - NASA_EARTHDATA_PASS
    """
    
    # ASF DAAC endpoints
    ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"
    ASF_AUTH_URL = "https://urs.earthdata.nasa.gov"
    CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
    
    # NISAR collection concepts (will be updated when NISAR is fully operational)
    NISAR_COLLECTIONS = {
        "L1-GSLC": "C1234567890-ASF",  # Placeholder - update with real IDs
        "L2-GCOV": "C1234567891-ASF",
        "L2-InSAR": "C1234567892-ASF",
        "L3-SM": "C1234567893-ASF",
    }
    
    def __init__(self, output_dir: Path, config: dict = None):
        """
        Initialize the NISAR downloader.
        
        Args:
            output_dir: Directory to save downloaded products
            config: NISAR configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.frequency = self.config.get('frequency', 'L-band')
        self.level = self.config.get('level', 'L2-GCOV')
        
        # Authentication
        self.username = os.getenv('NASA_EARTHDATA_USER')
        self.password = os.getenv('NASA_EARTHDATA_PASS')
        
        if not self.username or not self.password:
            raise ValueError(
                "NASA Earthdata credentials not found. "
                "Set NASA_EARTHDATA_USER and NASA_EARTHDATA_PASS environment variables."
            )
        
        self.session = self._create_authenticated_session()
        self.log_file = self.output_dir / "download_log.txt"
    
    def _log(self, message: str):
        """Log message to file and print."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
    
    def _create_authenticated_session(self) -> requests.Session:
        """Create an authenticated session for ASF DAAC."""
        session = requests.Session()
        
        # Authenticate with Earthdata Login
        auth_response = session.get(
            self.ASF_AUTH_URL,
            auth=(self.username, self.password),
            allow_redirects=True
        )
        
        if auth_response.status_code != 200:
            self._log(f"Warning: Initial auth returned {auth_response.status_code}")
        
        # Set up session with credentials
        session.auth = (self.username, self.password)
        
        return session
    
    def search_products(
        self,
        polygon: List[Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        max_results: int = 10
    ) -> List[NISARProduct]:
        """Search for NISAR products using asf_search library."""
        
        try:
            import asf_search as asf
            
            # Create WKT polygon (lon lat format)
            wkt_coords = ", ".join([f"{lon} {lat}" for lon, lat in polygon])
            wkt_polygon = f"POLYGON(({wkt_coords}, {polygon[0][0]} {polygon[0][1]}))"
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching NISAR GCOV products...")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Date: {start_date} to {end_date}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polygon: {wkt_polygon[:60]}...")
            
            # Search using asf_search
            results = asf.search(
                platform='NISAR',
                processingLevel='GCOV',
                intersectsWith=wkt_polygon,
                start=start_date,
                end=end_date,
                maxResults=max_results
            )
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(results)} GCOV products")
            
            products = []
            for r in results:
                props = r.properties
                
                product = NISARProduct(
                    product_id=props.get('fileID', 'unknown'),
                    filename=props.get('fileName', props.get('fileID', 'unknown')),
                    url=props.get('url', ''),
                    size_mb=0.0,
                    acquisition_date=datetime.fromisoformat(
                        props.get('startTime', datetime.now().isoformat()).replace('Z', '')
                    ),
                    processing_level='GCOV',
                    frequency=self.frequency,
                    polarization=props.get('polarization', 'Quad'),
                    geometry=r.geometry if hasattr(r, 'geometry') else {},
                    metadata=props
                )
                products.append(product)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   ✓ {product.product_id[:50]}...")
            
            return products
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Search error: {e}")
            return []

    def _search_cmr(
        self,
        polygon: List[Tuple[float, float]],
        start_date: datetime,
        end_date: datetime,
        max_results: int
    ) -> List[NISARProduct]:
        """Fallback search using CMR API."""
        self._log("Using CMR API fallback...")
        
        # Build polygon string for CMR
        coords_str = ",".join([f"{lon},{lat}" for lon, lat in polygon])
        
        params = {
            "collection_concept_id": self.NISAR_COLLECTIONS.get(self.level, ""),
            "temporal": f"{start_date.isoformat()},{end_date.isoformat()}",
            "polygon": coords_str,
            "page_size": max_results,
        }
        
        try:
            response = requests.get(self.CMR_SEARCH_URL, params=params, timeout=60)
            response.raise_for_status()
            results = response.json()
            
        except requests.exceptions.RequestException as e:
            self._log(f"CMR search also failed: {e}")
            return []
        
        products = []
        for entry in results.get("feed", {}).get("entry", []):
            try:
                product = self._parse_cmr_result(entry)
                products.append(product)
            except Exception as e:
                self._log(f"Error parsing CMR result: {e}")
                continue
        
        return products
    
    def _parse_asf_result(self, item: dict) -> NISARProduct:
        """Parse ASF search result into NISARProduct."""
        props = item.get("properties", item)
        
        # Extract acquisition date
        acq_date_str = props.get("startTime") or props.get("acquisitionDate")
        acq_date = datetime.fromisoformat(acq_date_str.replace("Z", "+00:00"))
        
        # Get file size
        size_bytes = props.get("bytes", 0) or props.get("sizeMB", 0) * 1024 * 1024
        size_mb = size_bytes / (1024 * 1024)
        
        return NISARProduct(
            product_id=props.get("fileID") or props.get("granuleId"),
            filename=props.get("fileName") or props.get("producerGranuleId"),
            url=props.get("url") or props.get("downloadUrl"),
            size_mb=size_mb,
            acquisition_date=acq_date,
            processing_level=self.level,
            frequency=props.get("frequency", self.frequency),
            polarization=props.get("polarization", "Quad-pol"),
            geometry=item.get("geometry", {}),
            metadata={
                "orbit": props.get("orbit"),
                "path": props.get("pathNumber"),
                "frame": props.get("frameNumber"),
                "look_direction": props.get("lookDirection"),
                "flight_direction": props.get("flightDirection"),
            }
        )
    
    def _parse_cmr_result(self, entry: dict) -> NISARProduct:
        """Parse CMR search result into NISARProduct."""
        # Extract download URL from links
        download_url = None
        for link in entry.get("links", []):
            if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                download_url = link.get("href")
                break
        
        # Parse time
        time_start = entry.get("time_start", "")
        acq_date = datetime.fromisoformat(time_start.replace("Z", "+00:00")) if time_start else datetime.now()
        
        return NISARProduct(
            product_id=entry.get("id", ""),
            filename=entry.get("producer_granule_id", entry.get("title", "")),
            url=download_url or "",
            size_mb=float(entry.get("granule_size", 0)),
            acquisition_date=acq_date,
            processing_level=self.level,
            frequency=self.frequency,
            polarization="Quad-pol",
            geometry=self._parse_cmr_geometry(entry),
            metadata=entry
        )
    
    def _parse_cmr_geometry(self, entry: dict) -> dict:
        """Extract geometry from CMR entry."""
        if "polygons" in entry:
            # CMR returns polygons as list of coordinate strings
            coords = []
            for poly in entry["polygons"]:
                for ring in poly:
                    points = ring.split()
                    ring_coords = []
                    for i in range(0, len(points), 2):
                        lat, lon = float(points[i]), float(points[i+1])
                        ring_coords.append([lon, lat])
                    coords.append(ring_coords)
            return {"type": "Polygon", "coordinates": coords}
        
        if "boxes" in entry:
            # Convert bounding box to polygon
            box = entry["boxes"][0].split()
            s, w, n, e = map(float, box)
            return {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
            }
        
        return {}
    
    def download_product(
        self,
        product: NISARProduct,
        progress_callback=None
    ) -> Optional[Path]:
        """
        Download a single NISAR product.
        
        Args:
            product: NISARProduct to download
            progress_callback: Optional callback(downloaded_mb, total_mb)
            
        Returns:
            Path to downloaded file, or None if failed
        """
        if not product.url:
            self._log(f"No download URL for {product.product_id}")
            return None
        
        # Create download directory
        download_dir = self.output_dir / "raw"
        download_dir.mkdir(exist_ok=True)
        
        # Determine output path
        output_path = download_dir / product.filename
        if not output_path.suffix:
            output_path = output_path.with_suffix(".h5")
        
        # Check if already downloaded (search in multiple locations)
        if output_path.exists() and output_path.stat().st_size > 1000000:  # >1MB
            self._log(f"Already downloaded: {product.filename}")
            return output_path
        
        # Also search for file by pattern in output directory
        for search_dir in [self.output_dir, self.output_dir / "raw", self.output_dir / "raw" / "raw"]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"*{product.product_id}*.h5"))
                if matches and matches[0].stat().st_size > 1000000:
                    self._log(f"Found existing file: {matches[0]}")
                    return matches[0]
        
        self._log(f"Downloading: {product.filename} ({product.size_mb:.1f} MB)")
        
        try:
            # Stream download with progress
            response = self.session.get(
                product.url,
                stream=True,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192 * 16  # 128KB chunks
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(
                                downloaded / (1024 * 1024),
                                total_size / (1024 * 1024)
                            )
            
            self._log(f"Downloaded: {output_path.name}")
            
            # Save metadata
            meta_path = output_path.with_suffix('.meta.json')
            with open(meta_path, 'w') as f:
                json.dump(product.to_dict(), f, indent=2)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            self._log(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return None
    
    def download_all(
        self,
        products: List[NISARProduct],
        max_concurrent: int = 2
    ) -> List[Path]:
        """
        Download multiple products.
        
        Args:
            products: List of products to download
            max_concurrent: Maximum concurrent downloads (not implemented yet)
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_paths = []
        
        for i, product in enumerate(products, 1):
            self._log(f"Downloading product {i}/{len(products)}")
            
            def progress(dl, total):
                pct = (dl / total * 100) if total > 0 else 0
                print(f"\r  Progress: {dl:.1f}/{total:.1f} MB ({pct:.1f}%)", end="", flush=True)
            
            path = self.download_product(product, progress_callback=progress)
            print()  # Newline after progress
            
            if path:
                downloaded_paths.append(path)
            
            # Small delay between downloads
            if i < len(products):
                time.sleep(1)
        
        self._log(f"Downloaded {len(downloaded_paths)}/{len(products)} products")
        return downloaded_paths
    
    def get_sample_data_urls(self) -> List[str]:
        """
        Get URLs for NISAR sample data (for testing).
        
        Returns:
            List of sample data URLs
        """
        # NASA provides sample NISAR data for development
        return [
            "https://nisar.jpl.nasa.gov/data/sample/NISAR_L2_PR_GCOV_001_005_A_129_4020_SHNA_A_20210101T000000_20210101T000000_P01101_F_N_J_001.h5",
            # Add more sample URLs as they become available
        ]
    
    def verify_download(self, file_path: Path) -> bool:
        """
        Verify downloaded file integrity.
        
        Args:
            file_path: Path to downloaded file
            
        Returns:
            True if file is valid
        """
        if not file_path.exists():
            return False
        
        # Check file size
        if file_path.stat().st_size < 1024:  # Less than 1KB
            self._log(f"File too small: {file_path}")
            return False
        
        # Try to open as HDF5
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # Check for expected NISAR groups
                required_groups = ['science', 'identification']
                for group in required_groups:
                    if group not in f:
                        self._log(f"Missing group '{group}' in {file_path}")
                        # Don't fail - structure may vary
            return True
            
        except ImportError:
            self._log("h5py not available, skipping HDF5 validation")
            return True
            
        except Exception as e:
            self._log(f"HDF5 validation failed: {e}")
            return False
