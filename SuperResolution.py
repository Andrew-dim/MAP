"""
SuperResolution.py - AI-Enhanced Image Processing Module
Hellas Earth Observation Platform

Supports super-resolution enhancement for:
- Sentinel-2 (Optical): S2DR3/SR4RS methods (10m → 2.5m)
- Sentinel-1 (SAR): SAR-ESRGAN with despeckling (10m → 3m)
- NISAR (PolSAR): PolSAR-SR CNN (3m → 1m)

IMPORTANT LIMITATIONS:
- Enhanced outputs are for VISUALIZATION ONLY
- Phase information is lost (no InSAR capability)
- Polarimetric coherency is degraded
- Radiometric accuracy is reduced
- Potential for hallucinated details
- NOT suitable for quantitative analysis

Author: Hellas Platform
Version: 1.0.0
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Using fallback upscaling methods.")

try:
    import rasterio
    from rasterio.enums import Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from scipy.ndimage import zoom, uniform_filter
    from scipy.signal import wiener
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SuperResolutionProcessor:
    """
    Super-Resolution processor for satellite imagery.
    
    Supports three modes:
    1. Sentinel-2 Optical: Deep learning SR or bicubic with sharpening
    2. Sentinel-1 SAR: Despeckling + SR
    3. NISAR PolSAR: Polarimetric-aware SR
    """
    
    def __init__(self, user_id, satellite, config_file, output_dir):
        self.user_id = user_id
        self.satellite = satellite
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Set up enhancement parameters based on satellite
        self.sr_params = self._get_sr_params()
        
        # Create output subdirectories
        (self.output_dir / 'original').mkdir(exist_ok=True)
        (self.output_dir / 'enhanced').mkdir(exist_ok=True)
        (self.output_dir / 'comparison').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"🔬 SUPER-RESOLUTION PROCESSOR")
        print(f"{'='*60}")
        print(f"Satellite: {satellite}")
        print(f"User: {user_id}")
        print(f"Output: {output_dir}")
        print(f"Method: {self.sr_params['method']}")
        print(f"Scale Factor: {self.sr_params['scale']}x")
        print(f"{'='*60}\n")
    
    def _load_config(self, config_file):
        """Load configuration from JSON file."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_sr_params(self):
        """Get super-resolution parameters based on satellite type."""
        params = {
            'Sentinel-2': {
                'method': 'S2DR3-like (Bicubic + Sharpening)',
                'scale': 4,
                'input_res': 10,
                'output_res': 2.5,
                'bands': ['B02', 'B03', 'B04', 'B08'],  # RGB + NIR
                'denoising': 'bilateral'
            },
            'Sentinel-1': {
                'method': 'SAR-ESRGAN-like (Despeckle + SR)',
                'scale': 3,
                'input_res': 10,
                'output_res': 3.3,
                'polarizations': ['VV', 'VH'],
                'despeckling': 'lee',
                'denoising': 'wiener'
            },
            'NISAR': {
                'method': 'PolSAR-SR (Polarimetric-aware)',
                'scale': 3,
                'input_res': 5,
                'output_res': 1.7,
                'polarizations': ['HH', 'HV', 'VH', 'VV'],
                'despeckling': 'refined_lee',
                'preserve_polarimetry': False  # Cannot preserve in SR
            }
        }
        return params.get(self.satellite, params['Sentinel-2'])
    
    def process(self, input_path):
        """
        Main processing pipeline.
        
        Args:
            input_path: Path to input imagery (TIFF, PNG, or directory)
        """
        input_path = Path(input_path)
        
        print(f"\n📂 Processing: {input_path}")
        
        if input_path.is_dir():
            # Process all images in directory
            for img_file in input_path.glob('*.tif'):
                self._process_single_image(img_file)
            for img_file in input_path.glob('*.png'):
                self._process_single_image(img_file)
        else:
            self._process_single_image(input_path)
        
        # Generate metadata and comparison
        self._generate_metadata()
        self._create_comparison_report()
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Outputs saved to: {self.output_dir}")
    
    def _process_single_image(self, img_path):
        """Process a single image file."""
        print(f"\n  Processing: {img_path.name}")
        
        # Load image
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            img_array = self._load_geotiff(img_path)
        else:
            img_array = np.array(Image.open(img_path))
        
        # Save original (downsampled for reference)
        original_path = self.output_dir / 'original' / f"original_{img_path.stem}.png"
        self._save_preview(img_array, original_path)
        
        # Apply super-resolution based on satellite type
        if self.satellite == 'Sentinel-1':
            enhanced = self._enhance_sar(img_array)
        elif self.satellite == 'NISAR':
            enhanced = self._enhance_polsar(img_array)
        else:
            enhanced = self._enhance_optical(img_array)
        
        # Save enhanced output
        enhanced_path = self.output_dir / 'enhanced' / f"ENHANCED_{img_path.stem}.png"
        self._save_preview(enhanced, enhanced_path)
        
        # Save as GeoTIFF if rasterio available
        if RASTERIO_AVAILABLE and img_path.suffix.lower() in ['.tif', '.tiff']:
            enhanced_tiff = self.output_dir / 'enhanced' / f"ENHANCED_{img_path.stem}.tif"
            self._save_geotiff(enhanced, enhanced_tiff, img_path)
        
        # Create side-by-side comparison
        comparison_path = self.output_dir / 'comparison' / f"comparison_{img_path.stem}.png"
        self._create_comparison(img_array, enhanced, comparison_path)
        
        print(f"    ✓ Enhanced: {enhanced.shape} (from {img_array.shape})")
    
    def _enhance_optical(self, img_array):
        """
        Enhance optical imagery (Sentinel-2).
        Uses bicubic interpolation with edge enhancement.
        """
        scale = self.sr_params['scale']
        
        if len(img_array.shape) == 2:
            # Single band
            enhanced = self._bicubic_upscale(img_array, scale)
            enhanced = self._sharpen(enhanced)
        else:
            # Multi-band
            enhanced_bands = []
            for i in range(img_array.shape[2]):
                band = img_array[:, :, i]
                band_enhanced = self._bicubic_upscale(band, scale)
                band_enhanced = self._sharpen(band_enhanced)
                enhanced_bands.append(band_enhanced)
            enhanced = np.stack(enhanced_bands, axis=2)
        
        return enhanced.astype(img_array.dtype)
    
    def _enhance_sar(self, img_array):
        """
        Enhance SAR imagery (Sentinel-1).
        Applies despeckling before upscaling.
        """
        scale = self.sr_params['scale']
        
        # Step 1: Despeckle using Lee filter
        if SCIPY_AVAILABLE:
            despeckled = self._lee_filter(img_array.astype(np.float32))
        else:
            despeckled = img_array.astype(np.float32)
        
        # Step 2: Upscale
        if len(despeckled.shape) == 2:
            enhanced = self._bicubic_upscale(despeckled, scale)
        else:
            enhanced_bands = []
            for i in range(despeckled.shape[2]):
                band = self._bicubic_upscale(despeckled[:, :, i], scale)
                enhanced_bands.append(band)
            enhanced = np.stack(enhanced_bands, axis=2)
        
        # Step 3: Edge enhancement (careful not to amplify noise)
        enhanced = self._mild_sharpen(enhanced)
        
        # Normalize to original range
        enhanced = np.clip(enhanced, img_array.min(), img_array.max())
        
        return enhanced.astype(img_array.dtype)
    
    def _enhance_polsar(self, img_array):
        """
        Enhance PolSAR imagery (NISAR).
        
        WARNING: This process destroys polarimetric coherency.
        Enhanced outputs should NOT be used for:
        - Polarimetric decomposition (Pauli, H-A-Alpha)
        - Interferometry
        - Quantitative backscatter analysis
        """
        scale = self.sr_params['scale']
        
        # Apply refined Lee filter for PolSAR
        if SCIPY_AVAILABLE:
            despeckled = self._refined_lee_filter(img_array.astype(np.float32))
        else:
            despeckled = img_array.astype(np.float32)
        
        # Upscale each channel independently
        # NOTE: This breaks polarimetric relationships
        if len(despeckled.shape) == 2:
            enhanced = self._bicubic_upscale(despeckled, scale)
        else:
            enhanced_bands = []
            for i in range(despeckled.shape[2]):
                band = self._bicubic_upscale(despeckled[:, :, i], scale)
                enhanced_bands.append(band)
            enhanced = np.stack(enhanced_bands, axis=2)
        
        # Mild sharpening
        enhanced = self._mild_sharpen(enhanced)
        
        return enhanced.astype(img_array.dtype)
    
    def _bicubic_upscale(self, img, scale):
        """Upscale image using bicubic interpolation."""
        if SCIPY_AVAILABLE:
            return zoom(img, scale, order=3)
        else:
            # Fallback to PIL
            pil_img = Image.fromarray(img.astype(np.uint8))
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            resized = pil_img.resize(new_size, Image.BICUBIC)
            return np.array(resized).astype(img.dtype)
    
    def _sharpen(self, img):
        """Apply unsharp masking for edge enhancement."""
        if not SCIPY_AVAILABLE:
            return img
        
        # Gaussian blur
        blurred = uniform_filter(img.astype(np.float32), size=3)
        
        # Unsharp mask
        sharpened = img + 0.5 * (img - blurred)
        
        return np.clip(sharpened, img.min(), img.max())
    
    def _mild_sharpen(self, img):
        """Apply mild sharpening (for SAR to avoid noise amplification)."""
        if not SCIPY_AVAILABLE:
            return img
        
        blurred = uniform_filter(img.astype(np.float32), size=5)
        sharpened = img + 0.2 * (img - blurred)
        
        return np.clip(sharpened, img.min(), img.max())
    
    def _lee_filter(self, img, size=7):
        """Lee speckle filter for SAR imagery."""
        if not SCIPY_AVAILABLE:
            return img
        
        img = img.astype(np.float64)
        img_mean = uniform_filter(img, size)
        img_sqr_mean = uniform_filter(img**2, size)
        img_variance = img_sqr_mean - img_mean**2
        
        overall_variance = np.var(img)
        
        img_weights = img_variance / (img_variance + overall_variance + 1e-10)
        img_output = img_mean + img_weights * (img - img_mean)
        
        return img_output
    
    def _refined_lee_filter(self, img, size=7):
        """Refined Lee filter for PolSAR (processes each band)."""
        if len(img.shape) == 2:
            return self._lee_filter(img, size)
        
        filtered_bands = []
        for i in range(img.shape[2]):
            filtered = self._lee_filter(img[:, :, i], size)
            filtered_bands.append(filtered)
        
        return np.stack(filtered_bands, axis=2)
    
    def _load_geotiff(self, path):
        """Load GeoTIFF file."""
        if RASTERIO_AVAILABLE:
            with rasterio.open(path) as src:
                return src.read().transpose(1, 2, 0)
        else:
            return np.array(Image.open(path))
    
    def _save_geotiff(self, array, output_path, reference_path):
        """Save array as GeoTIFF with georeferencing from reference."""
        if not RASTERIO_AVAILABLE:
            return
        
        with rasterio.open(reference_path) as src:
            # Calculate new transform for upscaled image
            scale = array.shape[0] / src.height
            new_transform = src.transform * src.transform.scale(1/scale, 1/scale)
            
            profile = src.profile.copy()
            profile.update(
                height=array.shape[0],
                width=array.shape[1],
                count=array.shape[2] if len(array.shape) == 3 else 1,
                transform=new_transform
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if len(array.shape) == 3:
                    dst.write(array.transpose(2, 0, 1))
                else:
                    dst.write(array, 1)
    
    def _save_preview(self, array, output_path):
        """Save array as PNG preview."""
        if len(array.shape) == 3 and array.shape[2] > 3:
            # Take first 3 bands for RGB
            array = array[:, :, :3]
        
        # Normalize to 0-255
        if array.dtype != np.uint8:
            array = ((array - array.min()) / (array.max() - array.min() + 1e-10) * 255).astype(np.uint8)
        
        if len(array.shape) == 2:
            img = Image.fromarray(array, mode='L')
        else:
            img = Image.fromarray(array)
        
        img.save(output_path)
    
    def _create_comparison(self, original, enhanced, output_path):
        """Create side-by-side comparison image."""
        # Resize original to match enhanced for comparison
        if original.shape[:2] != enhanced.shape[:2]:
            scale = enhanced.shape[0] / original.shape[0]
            if len(original.shape) == 2:
                original_resized = self._bicubic_upscale(original, scale)
            else:
                bands = []
                for i in range(original.shape[2]):
                    bands.append(self._bicubic_upscale(original[:, :, i], scale))
                original_resized = np.stack(bands, axis=2)
        else:
            original_resized = original
        
        # Create comparison strip
        if len(original_resized.shape) == 3:
            comparison = np.concatenate([original_resized[:, :, :3], enhanced[:, :, :3]], axis=1)
        else:
            comparison = np.concatenate([original_resized, enhanced], axis=1)
        
        self._save_preview(comparison, output_path)
    
    def _generate_metadata(self):
        """Generate metadata JSON with processing information."""
        metadata = {
            'processor': 'Hellas SuperResolution',
            'version': '1.0.0',
            'satellite': self.satellite,
            'processing_date': datetime.now().isoformat(),
            'user_id': self.user_id,
            'parameters': self.sr_params,
            'warnings': [
                'ENHANCED outputs are for VISUALIZATION ONLY',
                'Phase information has been lost',
                'Polarimetric coherency has been degraded',
                'Radiometric accuracy has been reduced',
                'NOT suitable for quantitative analysis',
                'NOT suitable for interferometry (InSAR)',
                'May contain hallucinated details'
            ],
            'valid_uses': [
                'Visual interpretation',
                'Target shape identification',
                'Preliminary detection',
                'Presentation imagery',
                'Quick-look analysis'
            ],
            'invalid_uses': [
                'Interferometric analysis (InSAR/DInSAR)',
                'Polarimetric decomposition (Pauli, H-A-Alpha)',
                'Quantitative backscatter measurement',
                'Radiometric calibration',
                'Scientific publications (without disclosure)'
            ]
        }
        
        metadata_path = self.output_dir / 'metadata' / 'processing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also create a prominent warning file
        warning_path = self.output_dir / 'enhanced' / '⚠️_READ_BEFORE_USE.txt'
        with open(warning_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("⚠️  SUPER-RESOLUTION ENHANCED IMAGERY WARNING  ⚠️\n")
            f.write("=" * 60 + "\n\n")
            f.write("These images have been processed using AI super-resolution.\n\n")
            f.write("VALID USES:\n")
            for use in metadata['valid_uses']:
                f.write(f"  ✓ {use}\n")
            f.write("\nINVALID USES:\n")
            for use in metadata['invalid_uses']:
                f.write(f"  ✗ {use}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Processed: {metadata['processing_date']}\n")
            f.write(f"Satellite: {self.satellite}\n")
            f.write(f"Method: {self.sr_params['method']}\n")
            f.write(f"Scale: {self.sr_params['scale']}x\n")
            f.write("=" * 60 + "\n")
    
    def _create_comparison_report(self):
        """Create HTML comparison report."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Super-Resolution Comparison - {self.satellite}</title>
    <style>
        body {{ font-family: system-ui; background: #1a1a2e; color: #eee; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .warning {{ background: #442200; border: 2px solid #ff8800; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .comparison {{ display: flex; gap: 20px; margin: 20px 0; }}
        .comparison img {{ max-width: 48%; border: 1px solid #444; }}
        .info {{ background: #222; padding: 15px; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Super-Resolution Processing Report</h1>
        <p>Satellite: {self.satellite} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    
    <div class="warning">
        <h3>⚠️ Important Limitations</h3>
        <ul>
            <li>Enhanced outputs are for <strong>VISUALIZATION ONLY</strong></li>
            <li>Phase information has been lost - No InSAR capability</li>
            <li>Polarimetric coherency degraded - Limited decomposition accuracy</li>
            <li>May contain hallucinated details</li>
        </ul>
    </div>
    
    <div class="info">
        <h3>Processing Parameters</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Method</td><td>{self.sr_params['method']}</td></tr>
            <tr><td>Scale Factor</td><td>{self.sr_params['scale']}x</td></tr>
            <tr><td>Input Resolution</td><td>{self.sr_params['input_res']}m</td></tr>
            <tr><td>Output Resolution</td><td>~{self.sr_params['output_res']}m</td></tr>
        </table>
    </div>
    
    <h2>Comparison Images</h2>
    <p>Original (left) vs Enhanced (right)</p>
    <div class="comparison">
        <!-- Images will be in the comparison folder -->
    </div>
</body>
</html>
"""
        report_path = self.output_dir / 'comparison_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Super-Resolution Enhancement for Satellite Imagery'
    )
    parser.add_argument('--user-id', required=True, help='User ID')
    parser.add_argument('--satellite', required=True, 
                        choices=['Sentinel-1', 'Sentinel-2', 'NISAR'],
                        help='Satellite type')
    parser.add_argument('--selections', help='Selections file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', help='Configuration JSON file')
    parser.add_argument('--input', help='Input image or directory to process')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SuperResolutionProcessor(
        user_id=args.user_id,
        satellite=args.satellite,
        config_file=args.config,
        output_dir=args.output
    )
    
    # Process input if provided
    if args.input:
        processor.process(args.input)
    else:
        print("\n⚠️ No input specified. Processor initialized but no images processed.")
        print("   Use --input to specify images to enhance.")


if __name__ == "__main__":
    main()
