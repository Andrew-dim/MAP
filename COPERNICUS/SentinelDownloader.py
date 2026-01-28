"""
Enhanced Sentinel-1 & Sentinel-2 Data Downloader and Processor
Version: 2.0 - Precise SAR Calibration

Sentinel-2 (Optical):
- Max cloud cover filtering (API-compliant)
- True color RGB composites
- 13 bands statistics
- NDVI calculation and visualization
- TCIs saved as PNG

Sentinel-1 (SAR):
- GRD and SLC product support
- VV, VH, HH, HV polarizations
- PRECISE radiometric calibration using annotation XML (with fallback to simplified)
- Noise correction (NESZ subtraction)
- Speckle filtering (Lee filter)
- Grayscale and false-color RGB outputs
- Comprehensive backscatter statistics

Common Features:
- Automatic ZIP deletion after extraction
- Automatic .SAFE folder cleanup
- Statistics and histograms
- Clipped outputs to user AOI
"""

from COPERNICUS.CopernicusDataSpace import CopernicusDataSpace
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from PIL import Image
import json
import warnings

# Import SAR calibration module (optional - graceful fallback if not available)
try:
    from sar_calibration import SARCalibrator, find_calibration_files
    SAR_CALIBRATION_AVAILABLE = True
    print("✓ SAR Calibration module loaded (precise calibration enabled)")
except ImportError:
    SAR_CALIBRATION_AVAILABLE = False
    warnings.warn("⚠ sar_calibration module not found. Will use simplified calibration.")

# Fix PIL decompression bomb warning
Image.MAX_IMAGE_PIXELS = None


class SentinelDownloader:
    def __init__(self, username, password, output_dir):
        print("Connecting to Copernicus Data Space...")
        self.api = CopernicusDataSpace(username, password)

        if not self.api.get_access_token():
            raise Exception("Failed to connect. Please check your credentials.")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delete_safe_folders = True
        print("✓ Connected successfully\n")

    def parse_selections_file(self, filepath):
        selections = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Updated pattern to capture Constellation field
        pattern = r'Selection\s+(\d+)\s*-\s*(\w+)\s*\n\s*Constellation:\s*(Sentinel-[12])\s*\n\s*Time Range:\s*([^\n]+?)\s+to\s+([^\n]+?)\s*\n\s*Max Cloud Cover:\s*(\d+)%\s*\n((?:\s*[-\d.]+\s*,\s*[-\d.]+\s*\n)+)'
        matches = re.findall(pattern, content)

        if not matches:
            print("  ⚠ Warning: Using legacy format (no Constellation field)")
            pattern2 = r'Selection\s+(\d+)\s*-\s*(\w+)\s*\n\s*Time Range:\s*([^\n]+?)\s+to\s+([^\n]+?)\s*\n\s*Max Cloud Cover:\s*(\d+)%\s*\n((?:\s*[-\d.]+\s*,\s*[-\d.]+\s*\n)+)'
            matches = re.findall(pattern2, content)
            legacy_format = True
        else:
            legacy_format = False

        for match in matches:
            if not legacy_format and len(match) == 7:
                sel_num, map_type, constellation, time_from, time_to, cloud_max, coords_block = match
            elif legacy_format and len(match) == 6:
                sel_num, map_type, time_from, time_to, cloud_max, coords_block = match
                constellation = "Sentinel-2"
            else:
                continue

            coords = []
            for line in coords_block.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    lon, lat = map(float, line.split(','))
                    coords.append((lon, lat))
                except ValueError:
                    continue

            if not coords:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                selections.append({
                    'number': int(sel_num),
                    'map_type': map_type,
                    'constellation': constellation,
                    'time_from': datetime.strptime(time_from.strip(), '%Y-%m-%d %H:%M'),
                    'time_to': datetime.strptime(time_to.strip(), '%Y-%m-%d %H:%M'),
                    'cloud_max': int(cloud_max),
                    'coordinates': coords,
                    'polygon': Polygon(coords)
                })
            except Exception as e:
                print(f"  ⚠ Error parsing selection: {e}")
                continue

        return selections

    def extract_product_metadata(self, product_name):
        """Extract metadata from Sentinel product name"""

        tile_code = "Unknown"
        date_str = "unknown"
        level = "Unknown"
        satellite = "Unknown"
        time_str = "unknown"

        # Detect satellite type
        if product_name.startswith('S1'):
            s1_meta = self.extract_sentinel1_metadata(product_name)
            return (
                s1_meta.get('mode', 'Unknown'),
                s1_meta.get('date_str', 'unknown'),
                s1_meta.get('product_type', 'Unknown'),
                s1_meta.get('satellite', 'Unknown'),
                s1_meta.get('time_str', 'unknown')
            )

        # Sentinel-2 processing
        tile_match = re.search(r'_T(\d{2}[A-Z]{3})_', product_name)
        if tile_match:
            tile_code = "T" + tile_match.group(1)

        datetime_match = re.search(r'_(\d{8})T(\d{6})_', product_name)
        if datetime_match:
            date_str = datetime_match.group(1)
            time_str = datetime_match.group(2)

        if 'MSIL1C' in product_name:
            level = "L1C"
        elif 'MSIL2A' in product_name:
            level = "L2A"

        if product_name.startswith('S2A'):
            satellite = "S2A"
        elif product_name.startswith('S2B'):
            satellite = "S2B"
        elif product_name.startswith('S2C'):
            satellite = "S2C"

        if tile_code == "Unknown" or date_str == "unknown":
            print(f"      ⚠ Warning: Could not extract complete metadata from: {product_name[:60]}...")
            print(f"         Extracted: tile={tile_code}, date={date_str}, satellite={satellite}")

        return tile_code, date_str, level, satellite, time_str

    def extract_sentinel1_metadata(self, product_name):
        """Extract metadata from Sentinel-1 product name"""

        metadata = {
            'satellite': 'Unknown',
            'mode': 'Unknown',
            'product_type': 'Unknown',
            'polarization': 'Unknown',
            'date_str': 'unknown',
            'time_str': 'unknown'
        }

        parts = product_name.split('_')

        if len(parts) >= 6:
            metadata['satellite'] = parts[0]
            metadata['mode'] = parts[1]

            if parts[2] == 'SLC':
                metadata['product_type'] = 'SLC'
                pol_part = parts[4] if len(parts) > 4 else parts[3]
            else:
                metadata['product_type'] = parts[2][:3]
                pol_part = parts[3]

            if 'DV' in pol_part:
                metadata['polarization'] = 'VV+VH'
            elif 'DH' in pol_part:
                metadata['polarization'] = 'HH+HV'
            elif 'SV' in pol_part:
                metadata['polarization'] = 'VV'
            elif 'SH' in pol_part:
                metadata['polarization'] = 'HH'
            else:
                metadata['polarization'] = 'VV'

            for part in parts:
                if 'T' in part and len(part) >= 15:
                    date_time = part
                    if 'T' in date_time:
                        metadata['date_str'] = date_time.split('T')[0]
                        metadata['time_str'] = date_time.split('T')[1]
                    break

        return metadata

    def create_output_archive(self, selection_num, tile_code, date_str, level, satellite, time_str, previews_dir):
        """Create ZIP archive with all output files"""
        import zipfile

        print(f"      Creating output archive...")

        base_name = f"selection_{selection_num}_{tile_code}_{date_str}_{level}"

        files_to_archive = [
            previews_dir / f"{base_name}_clipped_RGB.png",
            previews_dir / f"{base_name}_NDVI.png",
            previews_dir / f"{base_name}_all_bands_statistics.json",
            previews_dir / f"{base_name}_statistics.json",
            previews_dir / f"{base_name}_NDVI_statistics.json",
            previews_dir / f"{base_name}_histograms.png"
        ]

        png_preview = previews_dir / f"{base_name}_quick_preview.png"
        jp2_preview = previews_dir / f"{base_name}_quick_preview.jp2"
        if png_preview.exists():
            files_to_archive.insert(0, png_preview)
        elif jp2_preview.exists():
            files_to_archive.insert(0, jp2_preview)

        existing_files = [f for f in files_to_archive if f.exists()]

        if not existing_files:
            print(f"      ⚠ No output files found to archive")
            return None

        zip_filename = f"{satellite}_{date_str}_{time_str}_{tile_code}_{level}.zip"
        zip_path = previews_dir / zip_filename

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in existing_files:
                    zipf.write(file_path, file_path.name)

            individual_size = sum(f.stat().st_size for f in existing_files) / (1024 * 1024)
            zip_size = zip_path.stat().st_size / (1024 * 1024)
            compression_ratio = (1 - zip_size / individual_size) * 100 if individual_size > 0 else 0

            print(f"      ✓ Archive created: {zip_filename}")
            print(f"        Files: {len(existing_files)}")
            print(f"        Size: {zip_size:.1f} MB (compression: {compression_ratio:.1f}%)")

            for file_path in existing_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"      ⚠ Could not delete {file_path.name}: {e}")

            print(f"      ✓ Original files removed")
            return zip_path

        except Exception as e:
            print(f"      ✗ Error creating archive: {e}")
            return None

    def query_sentinel2(self, polygon, date_from, date_to, cloud_cover_max=100, satellite="SENTINEL-2"):
        """Query Sentinel products"""

        satellite_name = "Sentinel-1 SAR" if satellite == "SENTINEL-1" else "Sentinel-2 Optical"
        print(f"Querying {satellite_name} data...")
        print(f"  Date range: {date_from.date()} to {date_to.date()}")
        if satellite == "SENTINEL-2":
            print(f"  Max cloud cover: {cloud_cover_max}%")
        else:
            print(f"  All-weather SAR (no cloud filtering)")

        with tqdm(total=1, desc="  Searching", bar_format='{desc}... ', leave=False) as pbar:
            products = self.api.search_products(polygon, date_from, date_to, cloud_cover_max, satellite)
            pbar.update(1)

        print(f"  ✓ Found {len(products)} product(s)")
        return products

    def download_products(self, products, selection_num):
        if not products:
            print("  No products to download")
            return []

        download_dir = self.output_dir / f"selection_{selection_num}_raw"
        download_dir.mkdir(exist_ok=True)

        print(f"\n  Downloading {len(products)} product(s)...")
        downloaded = []

        for idx, product in enumerate(tqdm(products, desc="  Overall progress", unit="product"), 1):
            try:
                product_name = product['Name']
                product_id = product['Id']

                if '_RAW_' in product_name or '_OCN_' in product_name:
                    print(f"\n  [{idx}/{len(products)}] Skipping non-processable product:")
                    print(f"      {product_name}")
                    print(f"      ⊘ RAW/OCN products cannot be processed")
                    continue

                size_mb = product.get('ContentLength', 0) / (1024 * 1024)

                print(f"\n  [{idx}/{len(products)}] {product_name}")
                if size_mb > 0:
                    print(f"      Size: {size_mb:.1f} MB")

                output_path = download_dir / f"{product_name}.zip"
                print(f"      Expected file: {output_path.name}")

                if output_path.exists():
                    print(f"      ✓ Already downloaded (skipping download)")
                    print(f"      → Will proceed to processing")
                    downloaded.append({'path': output_path, 'name': product_name})
                    continue

                success = self.api.download_product(product_id, output_path, max_retries=3)

                if success:
                    downloaded.append({'path': output_path, 'name': product_name})
                    print(f"      ✓ Complete")
                else:
                    print(f"      ✗ Failed after all retries")

            except Exception as e:
                print(f"      ✗ Error: {e}")
                continue

        print(f"\n  Download summary:")
        print(f"    Total products ready for processing: {len(downloaded)}")
        if len(downloaded) < len(products):
            print(f"    Failed downloads: {len(products) - len(downloaded)}")

        return downloaded

    def check_output_exists(self, product_name, previews_dir):
        """Check if processed output already exists"""

        if not previews_dir.exists():
            return None

        if product_name.startswith('S2'):
            tile_code, date_str, level, satellite, time_str = self.extract_product_metadata(product_name)

            if date_str == "unknown" or tile_code == "Unknown":
                print(f"      ⚠ Could not determine output filename (metadata extraction failed)")
                return None

            expected_output = previews_dir / f"{satellite}_{date_str}_{time_str}_{tile_code}_{level}.zip"

        elif product_name.startswith('S1'):
            metadata = self.extract_sentinel1_metadata(product_name)
            satellite = metadata['satellite']
            date_str = metadata['date_str']
            time_str = metadata['time_str']
            mode = metadata['mode']
            product_type = metadata['product_type']

            if date_str == "unknown":
                print(f"      ⚠ Could not determine output filename (metadata extraction failed)")
                return None

            if 'DV' in product_name:
                pol_str = 'VV-VH'
            elif 'DH' in product_name:
                pol_str = 'HH-HV'
            elif 'SV' in product_name:
                pol_str = 'VV'
            elif 'SH' in product_name:
                pol_str = 'HH'
            else:
                pol_str = 'VV'

            expected_output = previews_dir / f"{satellite}_{date_str}_{time_str}_{mode}_{product_type}_{pol_str}.zip"
        else:
            return None

        if expected_output.exists():
            return expected_output

        return None

    def extract_and_process_product(self, product_path, polygon, selection_num):
        import zipfile

        print(f"      Extracting archive...")

        extract_base = product_path.parent
        product_stem = product_path.stem

        safe_folders = [f for f in extract_base.glob("*.SAFE") if f.is_dir() and product_stem in f.name]

        if not safe_folders:
            try:
                zip_size_mb = product_path.stat().st_size / (1024 * 1024)

                with zipfile.ZipFile(product_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_base)
                    print(f"      ✓ Extracted")

                product_path.unlink()
                print(f"      ✓ Deleted ZIP file (freed {zip_size_mb:.0f} MB)")

            except Exception as e:
                print(f"      ✗ Extraction error: {e}")
                return None

            safe_folders = [f for f in extract_base.glob("*.SAFE") if f.is_dir() and product_stem in f.name]

        if not safe_folders:
            print(f"      ✗ No .SAFE folder found after extraction")
            return None

        safe_folder = safe_folders[0]
        print(f"      ✓ Using: {safe_folder.name}")

        if safe_folder.name.startswith('S1'):
            result = self.process_sentinel1_product(safe_folder, polygon, selection_num)
        elif safe_folder.name.startswith('S2'):
            result = self.process_sentinel2_product(safe_folder, polygon, selection_num)
        else:
            print(f"      ✗ Unknown product type: {safe_folder.name}")
            result = None

        if safe_folder.exists():
            try:
                total_size = sum(f.stat().st_size for f in safe_folder.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)

                import shutil
                shutil.rmtree(safe_folder)
                print(f"      ✓ Deleted .SAFE folder (freed {size_mb:.0f} MB)")
            except Exception as e:
                print(f"      ⚠ Could not delete .SAFE folder: {e}")

        return result

    # ==================== SENTINEL-2 PROCESSING ====================

    def process_sentinel2_product(self, safe_folder, polygon, selection_num):
        """Process Sentinel-2 optical product"""

        tile_code, date_str, level, satellite, time_str = self.extract_product_metadata(safe_folder.name)

        previews_dir = self.output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        try:
            self.extract_quick_preview(safe_folder, selection_num, tile_code, date_str, level, previews_dir)
        except Exception as e:
            print(f"      ⚠ Could not extract TCI preview: {e}")

        try:
            self.extract_all_bands_statistics(safe_folder, polygon, selection_num, tile_code, date_str, level,
                                              previews_dir)
        except Exception as e:
            print(f"      ⚠ Could not extract all bands statistics: {e}")

        try:
            result = self.create_true_color_rgb(safe_folder, polygon, selection_num, tile_code, date_str, level,
                                                previews_dir)
        except Exception as e:
            print(f"      ✗ Error creating RGB: {e}")
            result = None

        try:
            ndvi_result = self.calculate_and_visualize_ndvi(safe_folder, polygon, selection_num, tile_code, date_str,
                                                            level, previews_dir)
        except Exception as e:
            print(f"      ⚠ Could not create NDVI visualization: {e}")
            ndvi_result = None

        try:
            archive_path = self.create_output_archive(selection_num, tile_code, date_str, level, satellite, time_str,
                                                      previews_dir)
        except Exception as e:
            print(f"      ⚠ Could not create output archive: {e}")
            archive_path = None

        return result

    def extract_quick_preview(self, safe_folder, selection_num, tile_code, date_str, level, previews_dir):
        """Extract TCI.jp2 and convert to PNG"""
        print(f"      Extracting TCI preview...")

        tci_files = list(safe_folder.glob("**/IMG_DATA/*_TCI_*.jp2"))
        if not tci_files:
            tci_files = list(safe_folder.glob("**/IMG_DATA/*_TCI.jp2"))
        if not tci_files:
            tci_files = list(safe_folder.glob("**/IMG_DATA/R10m/*_TCI*.jp2"))

        if not tci_files:
            print(f"      ⚠ TCI file not found in {safe_folder.name}")
            return None

        tci_source = tci_files[0]
        print(f"      ✓ Found TCI: {tci_source.name}")

        try:
            with rasterio.open(tci_source) as src:
                data = src.read()
                rgb = np.transpose(data, (1, 2, 0))

                if rgb.max() > 255:
                    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)

                preview_path = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_quick_preview.png"

                img = Image.fromarray(rgb)
                img.save(preview_path, compress_level=6, optimize=True)

                original_size_mb = tci_source.stat().st_size / (1024 * 1024)
                new_size_mb = preview_path.stat().st_size / (1024 * 1024)

                print(f"      ✓ Quick preview saved: {preview_path.name}")
                print(f"      File size: {original_size_mb:.1f} MB → {new_size_mb:.1f} MB")
                print(f"      📁 Location: {preview_path.absolute()}")

                return preview_path

        except Exception as e:
            print(f"      ⚠ Error converting TCI to PNG: {e}")
            print(f"      ⚠ Falling back to simple copy as JP2...")

            preview_path = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_quick_preview.jp2"

            import shutil
            shutil.copy2(tci_source, preview_path)

            print(f"      ✓ Quick preview saved: {preview_path.name}")
            print(f"      📁 Location: {preview_path.absolute()}")

            return preview_path

    def extract_all_bands_statistics(self, safe_folder, polygon, selection_num, tile_code, date_str, level,
                                     previews_dir):
        """Extract percentiles for ALL Sentinel-2 bands"""
        print(f"      Extracting statistics for all bands...")

        img_data_paths = list(safe_folder.glob("**/IMG_DATA"))
        if not img_data_paths:
            print(f"      ⚠ Could not find IMG_DATA folder")
            return None

        img_data_root = img_data_paths[0]

        band_patterns = {
            'B01': ['**/R60m/*_B01_60m.jp2', '**/*_B01.jp2'],
            'B02': ['**/R10m/*_B02_10m.jp2', '**/*_B02.jp2'],
            'B03': ['**/R10m/*_B03_10m.jp2', '**/*_B03.jp2'],
            'B04': ['**/R10m/*_B04_10m.jp2', '**/*_B04.jp2'],
            'B05': ['**/R20m/*_B05_20m.jp2', '**/*_B05.jp2'],
            'B06': ['**/R20m/*_B06_20m.jp2', '**/*_B06.jp2'],
            'B07': ['**/R20m/*_B07_20m.jp2', '**/*_B07.jp2'],
            'B08': ['**/R10m/*_B08_10m.jp2', '**/*_B08.jp2'],
            'B8A': ['**/R20m/*_B8A_20m.jp2', '**/*_B8A.jp2'],
            'B09': ['**/R60m/*_B09_60m.jp2', '**/*_B09.jp2'],
            'B10': ['**/R60m/*_B10_60m.jp2', '**/*_B10.jp2'],
            'B11': ['**/R20m/*_B11_20m.jp2', '**/*_B11.jp2'],
            'B12': ['**/R20m/*_B12_20m.jp2', '**/*_B12.jp2']
        }

        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")

        all_bands_stats = {}
        percentiles_to_calc = [0.5, 1, 2, 3, 97, 98, 99, 99.5]

        for band_name, patterns in band_patterns.items():
            band_file = None
            for pattern in patterns:
                files = list(img_data_root.glob(pattern))
                if files:
                    band_file = files[0]
                    break

            if not band_file:
                print(f"      ⚠ {band_name} not found, skipping")
                continue

            try:
                with rasterio.open(band_file) as src:
                    if src.crs != gdf.crs:
                        gdf_reproj = gdf.to_crs(src.crs)
                    else:
                        gdf_reproj = gdf

                    out_image, _ = mask(src, gdf_reproj.geometry, crop=True)
                    band_data = out_image[0]

                    valid_pixels = band_data[band_data > 0]

                    if len(valid_pixels) == 0:
                        print(f"      ⚠ {band_name}: No valid pixels")
                        continue

                    percentile_values = np.percentile(valid_pixels, percentiles_to_calc)

                    all_bands_stats[band_name] = {
                        'resolution': src.res[0],
                        'min': float(np.min(valid_pixels)),
                        'max': float(np.max(valid_pixels)),
                        'mean': float(np.mean(valid_pixels)),
                        'median': float(np.median(valid_pixels)),
                        'std': float(np.std(valid_pixels)),
                        'percentiles': {
                            'p0.5': float(percentile_values[0]),
                            'p1': float(percentile_values[1]),
                            'p2': float(percentile_values[2]),
                            'p3': float(percentile_values[3]),
                            'p97': float(percentile_values[4]),
                            'p98': float(percentile_values[5]),
                            'p99': float(percentile_values[6]),
                            'p99.5': float(percentile_values[7])
                        },
                        'pixel_count': int(len(valid_pixels)),
                        'nodata_count': int(np.sum(band_data == 0)),
                        'file_path': str(band_file.name)
                    }

                    print(f"      ✓ {band_name}: {len(valid_pixels):,} pixels")

            except Exception as e:
                print(f"      ✗ Error processing {band_name}: {e}")
                continue

        if previews_dir:
            output_json = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_all_bands_statistics.json"
        else:
            output_json = self.output_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_all_bands_statistics.json"

        with open(output_json, 'w') as f:
            json.dump(all_bands_stats, f, indent=2)

        print(f"      ✓ All bands statistics saved: {output_json.name}")
        print(f"      📁 Location: {output_json.absolute()}")

        return all_bands_stats

    def generate_band_statistics(self, bands_dict, rgb_array, output_path_base):
        """Generate statistics, histograms and percentiles for each band"""
        print(f"      Generating statistics and histograms...")

        stats = {}

        for band_name, band_data in bands_dict.items():
            valid_pixels = band_data[band_data > 0]

            if len(valid_pixels) > 0:
                stats[band_name] = {
                    'min': float(np.min(valid_pixels)),
                    'max': float(np.max(valid_pixels)),
                    'mean': float(np.mean(valid_pixels)),
                    'median': float(np.median(valid_pixels)),
                    'std': float(np.std(valid_pixels)),
                    'percentiles': {
                        'p2': float(np.percentile(valid_pixels, 2)),
                        'p5': float(np.percentile(valid_pixels, 5)),
                        'p25': float(np.percentile(valid_pixels, 25)),
                        'p50': float(np.percentile(valid_pixels, 50)),
                        'p75': float(np.percentile(valid_pixels, 75)),
                        'p95': float(np.percentile(valid_pixels, 95)),
                        'p98': float(np.percentile(valid_pixels, 98))
                    },
                    'pixel_count': int(len(valid_pixels)),
                    'nodata_count': int(np.sum(band_data == 0))
                }

        for i, color_name in enumerate(['R_normalized', 'G_normalized', 'B_normalized']):
            band_data = rgb_array[:, :, i]
            valid_pixels = band_data[band_data > 0]

            if len(valid_pixels) > 0:
                stats[color_name] = {
                    'min': float(np.min(valid_pixels)),
                    'max': float(np.max(valid_pixels)),
                    'mean': float(np.mean(valid_pixels)),
                    'median': float(np.median(valid_pixels)),
                    'std': float(np.std(valid_pixels))
                }

        stats_json_path = str(output_path_base) + "_statistics.json"
        with open(stats_json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"      ✓ Statistics saved: {Path(stats_json_path).name}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Band Statistics and Histograms', fontsize=16, fontweight='bold')

        colors = {'R': 'red', 'G': 'green', 'B': 'blue'}

        for idx, (band_name, band_data) in enumerate(bands_dict.items()):
            ax = axes[0, idx]
            valid_pixels = band_data[band_data > 0]

            if len(valid_pixels) > 0:
                ax.hist(valid_pixels, bins=100, color=colors[band_name], alpha=0.7, edgecolor='black')
                ax.set_title(f'{band_name} Band (Original)', fontweight='bold')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

                p2, p98 = stats[band_name]['percentiles']['p2'], stats[band_name]['percentiles']['p98']
                ax.axvline(p2, color='orange', linestyle='--', linewidth=2, label=f'p2: {p2:.0f}')
                ax.axvline(p98, color='purple', linestyle='--', linewidth=2, label=f'p98: {p98:.0f}')
                ax.legend()

                textstr = f"Mean: {stats[band_name]['mean']:.0f}\nStd: {stats[band_name]['std']:.0f}\nMedian: {stats[band_name]['median']:.0f}"
                ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        norm_names = ['R_normalized', 'G_normalized', 'B_normalized']
        for idx, color_name in enumerate(norm_names):
            ax = axes[1, idx]
            band_data = rgb_array[:, :, idx]
            valid_pixels = band_data[band_data > 0]

            if len(valid_pixels) > 0:
                color_short = color_name.split('_')[0]
                ax.hist(valid_pixels, bins=50, color=colors[color_short], alpha=0.7, edgecolor='black')
                ax.set_title(f'{color_short} Band (Normalized 0-255)', fontweight='bold')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.set_xlim(0, 255)
                ax.grid(True, alpha=0.3)

                textstr = f"Mean: {stats[color_name]['mean']:.1f}\nStd: {stats[color_name]['std']:.1f}"
                ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        hist_png_path = str(output_path_base) + "_histograms.png"
        plt.savefig(hist_png_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"      ✓ Histograms saved: {Path(hist_png_path).name}")

        return stats

    def calculate_and_visualize_ndvi(self, safe_folder, polygon, selection_num, tile_code, date_str, level,
                                     previews_dir=None):
        """Calculate NDVI from NIR and Red bands"""
        print(f"      Calculating NDVI...")

        img_data_paths = list(safe_folder.glob("**/IMG_DATA/R10m"))
        if not img_data_paths:
            img_data_paths = list(safe_folder.glob("**/IMG_DATA"))

        if not img_data_paths:
            print(f"      ✗ Could not find IMG_DATA folder")
            return None

        img_data = img_data_paths[0]

        b08_files = list(img_data.glob("**/*B08_10m.jp2"))
        b04_files = list(img_data.glob("**/*B04_10m.jp2"))

        if not b08_files:
            b08_files = list(img_data.glob("**/*B08.jp2"))
        if not b04_files:
            b04_files = list(img_data.glob("**/*B04.jp2"))

        if not (b08_files and b04_files):
            print(f"      ⚠ Could not find B08 (NIR) or B04 (Red) bands for NDVI")
            return None

        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")

        print(f"      Loading NIR (B08)...")
        with rasterio.open(b08_files[0]) as src:
            if src.crs != gdf.crs:
                gdf_reproj = gdf.to_crs(src.crs)
            else:
                gdf_reproj = gdf

            nir_clipped, _ = mask(src, gdf_reproj.geometry, crop=True)
            nir = nir_clipped[0].astype(np.float32)

        print(f"      Loading Red (B04)...")
        with rasterio.open(b04_files[0]) as src:
            if src.crs != gdf.crs:
                gdf_reproj = gdf.to_crs(src.crs)
            else:
                gdf_reproj = gdf

            red_clipped, _ = mask(src, gdf_reproj.geometry, crop=True)
            red = red_clipped[0].astype(np.float32)

        print(f"      Computing NDVI = (NIR - Red) / (NIR + Red)...")

        denominator = nir + red
        denominator[denominator == 0] = 1e-10

        ndvi = (nir - red) / denominator

        nodata_mask = (nir == 0) & (red == 0)
        ndvi[nodata_mask] = np.nan

        valid_ndvi = ndvi[~np.isnan(ndvi)]
        if len(valid_ndvi) > 0:
            ndvi_stats = {
                'min': float(np.min(valid_ndvi)),
                'max': float(np.max(valid_ndvi)),
                'mean': float(np.mean(valid_ndvi)),
                'median': float(np.median(valid_ndvi)),
                'std': float(np.std(valid_ndvi)),
                'percentiles': {
                    'p2': float(np.percentile(valid_ndvi, 2)),
                    'p25': float(np.percentile(valid_ndvi, 25)),
                    'p50': float(np.percentile(valid_ndvi, 50)),
                    'p75': float(np.percentile(valid_ndvi, 75)),
                    'p98': float(np.percentile(valid_ndvi, 98))
                },
                'valid_pixels': int(len(valid_ndvi)),
                'nodata_pixels': int(np.sum(nodata_mask))
            }

            print(f"      NDVI range: {ndvi_stats['min']:.3f} to {ndvi_stats['max']:.3f}")
            print(f"      NDVI mean: {ndvi_stats['mean']:.3f}")

        print(f"      Creating NDVI visualization...")
        ndvi_colored = self.colorize_ndvi(ndvi)

        if previews_dir:
            output_path = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_NDVI.png"
        else:
            output_path = self.output_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_NDVI.png"

        img = Image.fromarray(ndvi_colored)
        img.save(output_path)

        print(f"      ✓ NDVI visualization saved: {output_path.name}")
        print(f"      📁 Location: {output_path.absolute()}")

        if previews_dir and len(valid_ndvi) > 0:
            stats_path = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_NDVI_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(ndvi_stats, f, indent=2)
            print(f"      ✓ NDVI statistics saved: {stats_path.name}")

        return output_path

    def colorize_ndvi(self, ndvi):
        """Apply color scale to NDVI values"""
        h, w = ndvi.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        color_scale = [
            (-1.0, 0, 0, 255),
            (-0.1, 70, 130, 255),
            (0.0, 165, 42, 42),
            (0.1, 210, 180, 140),
            (0.2, 255, 250, 160),
            (0.3, 200, 255, 100),
            (0.4, 100, 200, 50),
            (0.5, 50, 180, 50),
            (0.6, 30, 150, 30),
            (0.7, 10, 120, 10),
            (1.0, 0, 90, 0),
        ]

        for i in range(h):
            for j in range(w):
                value = ndvi[i, j]

                if np.isnan(value):
                    rgb[i, j] = [255, 255, 255]
                    continue

                value = np.clip(value, -1.0, 1.0)

                for idx in range(len(color_scale) - 1):
                    lower_thresh, lower_r, lower_g, lower_b = color_scale[idx]
                    upper_thresh, upper_r, upper_g, upper_b = color_scale[idx + 1]

                    if lower_thresh <= value <= upper_thresh:
                        t = (value - lower_thresh) / (upper_thresh - lower_thresh)
                        r = int(lower_r + t * (upper_r - lower_r))
                        g = int(lower_g + t * (upper_g - lower_g))
                        b = int(lower_b + t * (upper_b - lower_b))
                        rgb[i, j] = [r, g, b]
                        break

        return rgb

    def create_true_color_rgb(self, safe_folder, polygon, selection_num, tile_code, date_str, level, previews_dir=None):
        print(f"      Creating clipped RGB...")

        img_data_paths = list(safe_folder.glob("**/IMG_DATA/R10m"))
        if not img_data_paths:
            img_data_paths = list(safe_folder.glob("**/IMG_DATA"))

        if not img_data_paths:
            print(f"      ✗ Could not find IMG_DATA folder")
            return None

        img_data = img_data_paths[0]

        b04_files = list(img_data.glob("**/*B04_10m.jp2"))
        b03_files = list(img_data.glob("**/*B03_10m.jp2"))
        b02_files = list(img_data.glob("**/*B02_10m.jp2"))

        if not b04_files:
            b04_files = list(img_data.glob("**/*B04.jp2"))
        if not b03_files:
            b03_files = list(img_data.glob("**/*B03.jp2"))
        if not b02_files:
            b02_files = list(img_data.glob("**/*B02.jp2"))

        if not (b04_files and b03_files and b02_files):
            print(f"      ✗ Could not find RGB bands")
            return None

        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")

        bands = []
        bands_dict = {}
        with tqdm(total=3, desc="      Processing bands", leave=False) as pbar:
            for idx, band_file in enumerate([b04_files[0], b03_files[0], b02_files[0]]):
                with rasterio.open(band_file) as src:
                    if src.crs != gdf.crs:
                        gdf_reproj = gdf.to_crs(src.crs)
                    else:
                        gdf_reproj = gdf

                    out_image, _ = mask(src, gdf_reproj.geometry, crop=True)
                    band_data = out_image[0]
                    bands.append(band_data)

                    band_names = ['R', 'G', 'B']
                    bands_dict[band_names[idx]] = band_data
                pbar.update(1)

        rgb = np.stack(bands, axis=-1)

        print(f"      Normalizing...")
        rgb_normalized = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[:, :, i]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                band_stretched = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                rgb_normalized[:, :, i] = band_stretched.astype(np.uint8)

        if previews_dir:
            output_path = previews_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_clipped_RGB.png"
        else:
            output_path = self.output_dir / f"selection_{selection_num}_{tile_code}_{date_str}_{level}_RGB.png"

        base_name = f"selection_{selection_num}_{tile_code}_{date_str}_{level}"
        output_path_base = previews_dir / base_name if previews_dir else self.output_dir / base_name
        self.generate_band_statistics(bands_dict, rgb_normalized, output_path_base)

        img = Image.fromarray(rgb_normalized)
        img.save(output_path)
        print(f"      ✓ Clipped RGB saved: {output_path.name}")
        print(f"      📁 Location: {output_path.absolute()}")

        return output_path

    # ==================== SENTINEL-1 SAR PROCESSING ====================
    # ==================== LEGACY SIMPLIFIED CALIBRATION METHODS ====================

    def calibrate_sar_simplified(self, img, calibration_vector=None):
        """Apply SIMPLIFIED radiometric calibration to SAR image

        Args:
            img: Raw DN values
            calibration_vector: Optional calibration LUT (not used in simplified method)

        Returns:
            Calibrated sigma0 in linear scale

        NOTE: This is a SIMPLIFIED approximation (±2-3 dB error).
        For accurate results, use SARCalibrator from sar_calibration module.
        """
        # Simplified calibration constant
        K = 1e6

        # Apply calibration: sigma0 ≈ DN^2 / K
        sigma0 = (img.astype(np.float32) ** 2) / K

        return sigma0

    def calibrate_sar(self, img, calibration_vector):
        """Legacy method - redirects to simplified calibration

        DEPRECATED: Use SARCalibrator from sar_calibration module for precise calibration.
        """
        warnings.warn("calibrate_sar is deprecated. Use SARCalibrator for precise calibration.", DeprecationWarning)
        return self.calibrate_sar_simplified(img, calibration_vector)

    def read_calibration_vector(self, annotation_file, polarization):
        """Legacy method - returns None (use SARCalibrator instead)

        DEPRECATED: Use SARCalibrator from sar_calibration module for precise calibration.
        """
        warnings.warn("read_calibration_vector is deprecated. Use SARCalibrator for precise calibration.", DeprecationWarning)
        return None

    def linear_to_db(self, img):
        """Convert linear scale to dB"""
        return 10 * np.log10(np.clip(img, 1e-10, None))

    def db_to_linear(self, img_db):
        """Convert dB to linear scale"""
        return 10 ** (img_db / 10)

    def lee_filter(self, img, size=5):
        """Apply Lee speckle filter to SAR image"""
        img_mean = uniform_filter(img, size)
        img_sqr_mean = uniform_filter(img ** 2, size)
        img_variance = img_sqr_mean - img_mean ** 2

        overall_variance = np.var(img)

        img_weights = img_variance / (img_variance + overall_variance + 1e-10)
        img_output = img_mean + img_weights * (img - img_mean)

        return img_output

    # ==================== ENHANCED SENTINEL-1 PROCESSING ====================

    def process_sentinel1_product(self, safe_folder, polygon, selection_num):
        """Process Sentinel-1 SAR product with PRECISE or SIMPLIFIED calibration

        Priority:
        1. Try PRECISE calibration (using sar_calibration.SARCalibrator)
        2. Fall back to SIMPLIFIED calibration if precise fails or unavailable

        The calibration method used is stored in output statistics.
        """

        print(f"      Processing Sentinel-1 SAR data...")

        metadata = self.extract_sentinel1_metadata(safe_folder.name)
        satellite = metadata['satellite']
        mode = metadata['mode']
        product_type = metadata['product_type']
        polarization = metadata['polarization']
        date_str = metadata['date_str']
        time_str = metadata['time_str']

        print(f"      Satellite: {satellite}, Mode: {mode}, Type: {product_type}")
        print(f"      Polarization: {polarization}")

        is_cog = '_COG' in safe_folder.name
        if is_cog:
            print(f"      Format: COG (Cloud Optimized GeoTIFF)")

        measurement_dir = safe_folder / "measurement"
        if not measurement_dir.exists():
            print(f"      ✗ No measurement directory found")
            print(f"      ℹ Searching for alternative paths...")
            all_tiffs = list(safe_folder.glob("**/*.tiff")) + list(safe_folder.glob("**/*.tif"))
            if all_tiffs:
                print(f"      ✓ Found {len(all_tiffs)} TIFF files in product")
                for tiff in all_tiffs[:3]:
                    print(f"         - {tiff.relative_to(safe_folder)}")
                measurement_dir = all_tiffs[0].parent
            else:
                if product_type == 'RAW':
                    print(f"      ℹ RAW (Level-0) products contain unprocessed sensor data")
                    print(f"      ℹ Only GRD and SLC products can be processed")
                return None

        vv_files = list(measurement_dir.glob("*-vv-*.tiff"))
        vh_files = list(measurement_dir.glob("*-vh-*.tiff"))
        hh_files = list(measurement_dir.glob("*-hh-*.tiff"))
        hv_files = list(measurement_dir.glob("*-hv-*.tiff"))

        available_pols = []
        pol_files = {}

        if vv_files:
            available_pols.append('VV')
            pol_files['VV'] = vv_files[0]
        if vh_files:
            available_pols.append('VH')
            pol_files['VH'] = vh_files[0]
        if hh_files:
            available_pols.append('HH')
            pol_files['HH'] = hh_files[0]
        if hv_files:
            available_pols.append('HV')
            pol_files['HV'] = hv_files[0]

        if not available_pols:
            print(f"      ✗ No polarization files found")
            return None

        print(f"      Found polarizations: {', '.join(available_pols)}")

        # Process each polarization
        processed_bands = {}
        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")

        for pol_name, pol_file in pol_files.items():
            print(f"      Processing {pol_name}...")

            try:
                # ==================== CALIBRATION STRATEGY ====================

                use_precise_calibration = False

                if SAR_CALIBRATION_AVAILABLE:
                    try:
                        calib_files = find_calibration_files(safe_folder, pol_name)

                        print(f"         Annotation: {calib_files['annotation'].name}")
                        if calib_files['calibration']:
                            print(f"         Calibration: {calib_files['calibration'].name}")
                        if calib_files['noise']:
                            print(f"         Noise: {calib_files['noise'].name}")

                        use_precise_calibration = True

                    except FileNotFoundError as e:
                        print(f"         ⚠ Calibration files not found: {e}")
                        print(f"         ⚠ Falling back to simplified calibration")
                        use_precise_calibration = False
                else:
                    print(f"         ℹ SAR calibration module not available")
                    print(f"         ℹ Using simplified calibration")

                # ==================== LOAD & CLIP IMAGE ====================

                with rasterio.open(pol_file) as src:
                    print(f"         File: {pol_file.name}")
                    print(f"         CRS: {src.crs}")
                    print(f"         Shape: {src.shape}")
                    print(f"         Bounds: {src.bounds}")

                    if src.crs is None:
                        print(f"      ⚠ {pol_name}: No CRS found in file")
                        print(f"      ℹ Processing full scene (cannot clip to AOI without CRS)")
                        raw_data = src.read(1)
                    elif src.crs != gdf.crs:
                        try:
                            gdf_reproj = gdf.to_crs(src.crs)
                            out_image, _ = mask(src, gdf_reproj.geometry, crop=True)
                            raw_data = out_image[0]
                        except Exception as crs_error:
                            print(f"      ⚠ {pol_name}: CRS reprojection failed, using full scene")
                            print(f"         Source CRS: {src.crs}")
                            print(f"         Error: {crs_error}")
                            raw_data = src.read(1)
                    else:
                        gdf_reproj = gdf
                        out_image, _ = mask(src, gdf_reproj.geometry, crop=True)
                        raw_data = out_image[0]

                    # Handle complex data (SLC products)
                    if np.iscomplexobj(raw_data):
                        print(f"      ℹ {pol_name}: Complex data detected (SLC)")
                        band_data_dn = raw_data.astype(np.complex64)
                    else:
                        band_data_dn = raw_data.astype(np.float32)

                # Check for valid data
                if np.iscomplexobj(band_data_dn):
                    valid_mask = np.abs(band_data_dn) > 0
                else:
                    valid_mask = band_data_dn > 0

                if not np.any(valid_mask):
                    print(f"      ⚠ {pol_name}: No valid pixels in AOI")
                    continue

                # ==================== RADIOMETRIC CALIBRATION ====================

                if use_precise_calibration:
                    # PRECISE CALIBRATION using annotation XML

                    print(f"         ℹ Applying PRECISE calibration...")

                    try:
                        # Initialize calibrator
                        calibrator = SARCalibrator(calib_files['annotation'])

                        # Load calibration LUT
                        if calib_files['calibration']:
                            calibrator.load_calibration_lut(
                                calib_files['calibration'],
                                calibration_type='sigma0'
                            )
                        else:
                            print(f"         ⚠ No calibration XML found")
                            use_precise_calibration = False

                        # Load noise LUT (optional)
                        if calib_files['noise'] and use_precise_calibration:
                            calibrator.load_noise_lut(calib_files['noise'])

                        if use_precise_calibration:
                            # Apply calibration
                            sigma0_linear = calibrator.calibrate(
                                band_data_dn,
                                calibration_type='sigma0',
                                apply_noise_correction=True
                            )

                            print(f"         ✓ Precise calibration complete")

                    except Exception as calib_error:
                        print(f"         ✗ Precise calibration failed: {calib_error}")
                        print(f"         ⚠ Falling back to simplified calibration")
                        use_precise_calibration = False

                # Fallback to simplified calibration
                if not use_precise_calibration:
                    print(f"         ℹ Using simplified calibration")

                    # Handle complex data
                    if np.iscomplexobj(band_data_dn):
                        intensity = np.abs(band_data_dn) ** 2
                    else:
                        intensity = band_data_dn ** 2

                    # Simplified calibration
                    sigma0_linear = self.calibrate_sar_simplified(band_data_dn)

                # ==================== SPECKLE FILTERING ====================

                print(f"      Applying Lee speckle filter to {pol_name}...")
                filtered_linear = self.lee_filter(sigma0_linear, size=5)

                # Convert to dB
                sigma0_db = self.linear_to_db(filtered_linear)

                # Mask nodata
                sigma0_db[~valid_mask] = np.nan

                processed_bands[pol_name] = {
                    'db': sigma0_db,
                    'linear': filtered_linear,
                    'valid_mask': valid_mask,
                    'calibration_method': 'precise' if use_precise_calibration else 'simplified'
                }

                print(f"      ✓ {pol_name} processed ({processed_bands[pol_name]['calibration_method']} calibration)")

                # Show calibration statistics
                valid_db = sigma0_db[~np.isnan(sigma0_db)]
                if len(valid_db) > 0:
                    print(f"         Backscatter range: {np.min(valid_db):.1f} to {np.max(valid_db):.1f} dB")
                    print(f"         Mean: {np.mean(valid_db):.1f} dB")

            except Exception as e:
                print(f"      ✗ Error processing {pol_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not processed_bands:
            print(f"      ✗ No bands successfully processed")
            return None

        # Create output directory
        previews_dir = self.output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        # Generate visualizations and statistics
        output_files = []

        try:
            # 1. Grayscale images
            for pol_name, pol_data in processed_bands.items():
                gray_path = self.create_sar_grayscale(
                    pol_data['db'],
                    pol_name,
                    selection_num,
                    satellite,
                    date_str,
                    mode,
                    product_type,
                    previews_dir
                )
                if gray_path:
                    output_files.append(gray_path)

            # 2. False color RGB
            if len(processed_bands) >= 2:
                rgb_path = self.create_sar_rgb_composite(
                    processed_bands,
                    selection_num,
                    satellite,
                    date_str,
                    mode,
                    product_type,
                    previews_dir
                )
                if rgb_path:
                    output_files.append(rgb_path)

            # 3. Enhanced statistics
            stats_path = self.extract_sar_statistics(
                processed_bands,
                selection_num,
                satellite,
                date_str,
                mode,
                product_type,
                previews_dir
            )
            if stats_path:
                output_files.append(stats_path)

            # 4. Histograms
            hist_path = self.create_sar_histograms(
                processed_bands,
                selection_num,
                satellite,
                date_str,
                mode,
                product_type,
                previews_dir
            )
            if hist_path:
                output_files.append(hist_path)

        except Exception as e:
            print(f"      ⚠ Error creating outputs: {e}")
            import traceback
            traceback.print_exc()

        # Create ZIP archive
        if output_files:
            zip_path = self.create_sar_archive(
                output_files,
                selection_num,
                satellite,
                date_str,
                time_str,
                mode,
                product_type,
                polarization,
                previews_dir
            )
            return zip_path

        return None

    def create_sar_grayscale(self, sigma0_db, pol_name, selection_num, satellite, date_str, mode, product_type,
                             output_dir):
        """Create grayscale visualization of SAR band"""
        print(f"      Creating {pol_name} grayscale...")

        try:
            vmin, vmax = -25, 5

            sigma0_normalized = np.clip((sigma0_db - vmin) / (vmax - vmin) * 255, 0, 255)
            sigma0_normalized = np.nan_to_num(sigma0_normalized, nan=0).astype(np.uint8)

            img = Image.fromarray(sigma0_normalized, mode='L')

            output_path = output_dir / f"selection_{selection_num}_{satellite}_{date_str}_{mode}_{product_type}_{pol_name}_grayscale.png"
            img.save(output_path)

            print(f"      ✓ {pol_name} grayscale saved: {output_path.name}")
            return output_path

        except Exception as e:
            print(f"      ✗ Error creating {pol_name} grayscale: {e}")
            return None

    def create_sar_rgb_composite(self, processed_bands, selection_num, satellite, date_str, mode, product_type,
                                 output_dir):
        """Create false color RGB composite"""
        print(f"      Creating false color RGB...")

        try:
            if 'VV' in processed_bands and 'VH' in processed_bands:
                vv_db = processed_bands['VV']['db']
                vh_db = processed_bands['VH']['db']
                ratio_db = vv_db - vh_db
                channels = [vv_db, vh_db, ratio_db]
                combo_name = "VV-VH-Ratio"

            elif 'HH' in processed_bands and 'HV' in processed_bands:
                hh_db = processed_bands['HH']['db']
                hv_db = processed_bands['HV']['db']
                ratio_db = hh_db - hv_db
                channels = [hh_db, hv_db, ratio_db]
                combo_name = "HH-HV-Ratio"
            else:
                print(f"      ⚠ Insufficient polarizations for RGB composite")
                return None

            rgb_array = np.zeros((*channels[0].shape, 3), dtype=np.uint8)

            for i, channel in enumerate(channels):
                if i < 2:
                    vmin, vmax = -25, 5
                else:
                    vmin, vmax = -10, 10

                normalized = np.clip((channel - vmin) / (vmax - vmin) * 255, 0, 255)
                normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)
                rgb_array[:, :, i] = normalized

            img = Image.fromarray(rgb_array, mode='RGB')

            output_path = output_dir / f"selection_{selection_num}_{satellite}_{date_str}_{mode}_{product_type}_RGB_composite.png"
            img.save(output_path)

            print(f"      ✓ RGB composite saved: {output_path.name}")
            print(f"        Channels: R={combo_name.split('-')[0]}, G={combo_name.split('-')[1]}, B={combo_name.split('-')[2]}")
            return output_path

        except Exception as e:
            print(f"      ✗ Error creating RGB composite: {e}")
            return None

    def extract_sar_statistics(self, processed_bands, selection_num, satellite, date_str, mode, product_type,
                               output_dir):
        """Extract ENHANCED statistics from SAR bands"""
        print(f"      Extracting SAR statistics...")

        try:
            stats = {
                'product': {
                    'satellite': satellite,
                    'date': date_str,
                    'mode': mode,
                    'type': product_type
                },
                'polarizations': {}
            }

            for pol_name, pol_data in processed_bands.items():
                sigma0_db = pol_data['db']
                valid_mask = pol_data['valid_mask']

                valid_pixels = sigma0_db[valid_mask & ~np.isnan(sigma0_db)]

                if len(valid_pixels) > 0:
                    stats['polarizations'][pol_name] = {
                        'calibration_method': pol_data.get('calibration_method', 'unknown'),
                        'backscatter_db': {
                            'mean': float(np.mean(valid_pixels)),
                            'std': float(np.std(valid_pixels)),
                            'min': float(np.min(valid_pixels)),
                            'max': float(np.max(valid_pixels)),
                            'median': float(np.median(valid_pixels)),
                            'percentiles': {
                                'p1': float(np.percentile(valid_pixels, 1)),
                                'p5': float(np.percentile(valid_pixels, 5)),
                                'p10': float(np.percentile(valid_pixels, 10)),
                                'p25': float(np.percentile(valid_pixels, 25)),
                                'p50': float(np.percentile(valid_pixels, 50)),
                                'p75': float(np.percentile(valid_pixels, 75)),
                                'p90': float(np.percentile(valid_pixels, 90)),
                                'p95': float(np.percentile(valid_pixels, 95)),
                                'p99': float(np.percentile(valid_pixels, 99))
                            }
                        },
                        'coverage': {
                            'valid_pixels': int(np.sum(valid_mask)),
                            'nodata_pixels': int(np.sum(~valid_mask)),
                            'coverage_percent': float(np.sum(valid_mask) / valid_mask.size * 100)
                        }
                    }

            # Add ratio statistics if dual-pol
            if 'VV' in processed_bands and 'VH' in processed_bands:
                vv_db = processed_bands['VV']['db']
                vh_db = processed_bands['VH']['db']
                valid_mask = processed_bands['VV']['valid_mask'] & processed_bands['VH']['valid_mask']

                ratio_db = vv_db - vh_db
                valid_ratio = ratio_db[valid_mask & ~np.isnan(ratio_db)]

                if len(valid_ratio) > 0:
                    stats['polarizations']['VV_VH_ratio'] = {
                        'ratio_db': {
                            'mean': float(np.mean(valid_ratio)),
                            'std': float(np.std(valid_ratio)),
                            'min': float(np.min(valid_ratio)),
                            'max': float(np.max(valid_ratio)),
                            'median': float(np.median(valid_ratio)),
                            'percentiles': {
                                'p25': float(np.percentile(valid_ratio, 25)),
                                'p50': float(np.percentile(valid_ratio, 50)),
                                'p75': float(np.percentile(valid_ratio, 75))
                            }
                        },
                        'interpretation': {
                            'high_ratio_gt_10db': 'Surface scattering (water, smooth surfaces)',
                            'medium_ratio_3_10db': 'Mixed scattering (urban, rough surfaces)',
                            'low_ratio_lt_3db': 'Volume scattering (vegetation, forests)'
                        }
                    }

            output_path = output_dir / f"selection_{selection_num}_{satellite}_{date_str}_{mode}_{product_type}_statistics.json"
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"      ✓ Statistics saved: {output_path.name}")
            return output_path

        except Exception as e:
            print(f"      ✗ Error extracting statistics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_sar_histograms(self, processed_bands, selection_num, satellite, date_str, mode, product_type,
                              output_dir):
        """Create histograms for SAR bands"""
        print(f"      Creating SAR histograms...")

        try:
            n_pols = len(processed_bands)
            fig, axes = plt.subplots(1, n_pols, figsize=(5 * n_pols, 4))

            if n_pols == 1:
                axes = [axes]

            colors = {'VV': 'blue', 'VH': 'green', 'HH': 'red', 'HV': 'orange'}

            for idx, (pol_name, pol_data) in enumerate(processed_bands.items()):
                ax = axes[idx]
                sigma0_db = pol_data['db']
                valid_mask = pol_data['valid_mask']

                valid_pixels = sigma0_db[valid_mask & ~np.isnan(sigma0_db)]

                if len(valid_pixels) > 0:
                    color = colors.get(pol_name, 'gray')
                    ax.hist(valid_pixels, bins=100, color=color, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{pol_name} Backscatter', fontweight='bold')
                    ax.set_xlabel('Backscatter (dB)')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)

                    mean_val = np.mean(valid_pixels)
                    std_val = np.std(valid_pixels)
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f} dB')
                    ax.legend()

                    textstr = f"Mean: {mean_val:.1f} dB\nStd: {std_val:.1f} dB\nPixels: {len(valid_pixels):,}"
                    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()

            output_path = output_dir / f"selection_{selection_num}_{satellite}_{date_str}_{mode}_{product_type}_histograms.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"      ✓ Histograms saved: {output_path.name}")
            return output_path

        except Exception as e:
            print(f"      ✗ Error creating histograms: {e}")
            return None

    def create_sar_archive(self, output_files, selection_num, satellite, date_str, time_str, mode, product_type,
                           polarization, output_dir):
        """Create ZIP archive for SAR outputs"""
        import zipfile

        print(f"      Creating SAR output archive...")

        try:
            pol_str = polarization.replace('+', '-')
            zip_filename = f"{satellite}_{date_str}_{time_str}_{mode}_{product_type}_{pol_str}.zip"
            zip_path = output_dir / zip_filename

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_files:
                    if file_path.exists():
                        zipf.write(file_path, file_path.name)

            individual_size = sum(f.stat().st_size for f in output_files if f.exists()) / (1024 * 1024)
            zip_size = zip_path.stat().st_size / (1024 * 1024)
            compression_ratio = (1 - zip_size / individual_size) * 100 if individual_size > 0 else 0

            print(f"      ✓ Archive created: {zip_filename}")
            print(f"        Files: {len(output_files)}")
            print(f"        Size: {zip_size:.1f} MB (compression: {compression_ratio:.1f}%)")

            for file_path in output_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    print(f"      ⚠ Could not delete {file_path.name}: {e}")

            print(f"      ✓ Original files removed")

            return zip_path

        except Exception as e:
            print(f"      ✗ Error creating archive: {e}")
            return None

    # ==================== MAIN PROCESSING ====================

    def process_selection(self, selection):
        print(f"\n{'=' * 70}")
        print(f"Processing Selection {selection['number']}")
        print(f"{'=' * 70}")
        print(f"Constellation: {selection.get('constellation', 'Sentinel-2')}")
        print(f"Coordinates: {len(selection['coordinates'])} vertices")
        print(f"Time range: {selection['time_from']} to {selection['time_to']}")
        print(f"Max cloud cover: {selection['cloud_max']}%")

        now = datetime.now()
        if selection['time_from'] > now:
            print(f"\n  ⚠ WARNING: Start date is in the future!")
            print(f"  Current date: {now.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Adjusting search to past 30 days...")

            from datetime import timedelta
            selection['time_to'] = now
            selection['time_from'] = now - timedelta(days=30)

            print(f"  New range: {selection['time_from'].strftime('%Y-%m-%d')} to {selection['time_to'].strftime('%Y-%m-%d')}")

        constellation = selection.get('constellation', 'Sentinel-2')

        if constellation == "Sentinel-2":
            print(f"\n  Querying Sentinel-2 (optical)...")
            products_s2 = self.query_sentinel2(
                selection['polygon'],
                selection['time_from'],
                selection['time_to'],
                selection['cloud_max'],
                satellite="SENTINEL-2"
            )
            products_s1 = []

        elif constellation == "Sentinel-1":
            print(f"\n  Querying Sentinel-1 (SAR)...")
            products_s1 = self.query_sentinel2(
                selection['polygon'],
                selection['time_from'],
                selection['time_to'],
                100,
                satellite="SENTINEL-1"
            )
            products_s2 = []

        else:
            print(f"\n  ✗ Unknown constellation: {constellation}")
            return

        products = products_s2 + products_s1

        # Filter S1 products
        if products_s1:
            filtered_s1 = []
            excluded_count = 0
            for product in products_s1:
                name = product.get('Name', '')
                if '_GRD' in name or '_SLC' in name:
                    filtered_s1.append(product)
                else:
                    excluded_count += 1

            if excluded_count > 0:
                print(f"  ⊘ Excluded {excluded_count} non-processable product(s) (RAW/OCN)")

            products = products_s2 + filtered_s1
            print(f"\n  ✓ Total usable: {len(products)} product(s) (S2: {len(products_s2)}, S1: {len(filtered_s1)})")
        else:
            print(f"\n  ✓ Total found: {len(products)} product(s)")

        if not products:
            print("\n  ✗ No products found")
            return

        print("\n  Available products:")
        for idx, product in enumerate(products[:10], 1):
            name = product.get('Name', 'Unknown')
            tile_code, date_str, level, satellite_name, time_str = self.extract_product_metadata(name)

            cloud_cover = None
            for attr in product.get('Attributes', []):
                if attr.get('Name') == 'cloudCover':
                    cloud_cover = attr.get('Value')
                    break

            print(f"    [{idx}] {name}")
            print(f"        Tile: {tile_code}, Level: {level}")
            if cloud_cover is not None:
                print(f"        Cloud: {cloud_cover:.1f}%")

        if len(products) > 10:
            print(f"    ... and {len(products) - 10} more")

        downloaded = self.download_products(products, selection['number'])

        if not downloaded:
            print("\n  ✗ No products downloaded")
            return

        print(f"\n  Processing downloaded products...")
        previews_dir = self.output_dir / "previews"
        previews_dir.mkdir(exist_ok=True)

        processed_count = 0
        skipped_count = 0

        for product_info in downloaded:
            product_name = product_info['name']
            print(f"\n    Product: {product_name}")

            existing_output = self.check_output_exists(product_name, previews_dir)
            if existing_output:
                print(f"      ✓ Output already exists: {existing_output.name}")
                print(f"      ⊙ Skipping processing (already completed)")
                skipped_count += 1
                continue

            self.extract_and_process_product(
                product_info['path'],
                selection['polygon'],
                selection['number']
            )
            processed_count += 1

        print(f"\n  Processing summary:")
        print(f"    Processed: {processed_count}")
        print(f"    Skipped (already done): {skipped_count}")
        print(f"    Total: {len(downloaded)}")

        print(f"\n✓ Selection {selection['number']} completed")

    def process_all_selections(self, filepath):
        print("=" * 70)
        print(" " * 15 + "Sentinel-1 & Sentinel-2 Data Processor v2.0")
        print(" " * 20 + "(Precise SAR Calibration)")
        print("=" * 70)

        print(f"\nReading selections from: {filepath}")
        selections = self.parse_selections_file(filepath)
        print(f"✓ Found {len(selections)} selection(s)\n")

        for idx, selection in enumerate(selections, 1):
            print(f"\n[{idx}/{len(selections)}] Starting selection {selection['number']}...")
            try:
                self.process_selection(selection)
            except Exception as e:
                print(f"\n✗ Error processing selection {selection['number']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "=" * 70)
        print("✓ All selections processed!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("=" * 70)
        print("\n💡 Output files:")
        print(f"   Location: {self.output_dir.absolute()}/previews/")
        print("   Format: ZIP archives")
        print("   Contents per ZIP:")
        print("     Sentinel-2:")
        print("       - Quick preview PNG, RGB PNG, NDVI PNG")
        print("       - All bands statistics, RGB statistics, NDVI statistics")
        print("       - Histograms")
        print("     Sentinel-1:")
        print("       - VV/VH grayscale PNGs, RGB composite PNG")
        print("       - Backscatter statistics (with calibration method)")
        print("       - Histograms")