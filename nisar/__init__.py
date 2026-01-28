"""
NISAR Processing Package
========================

Comprehensive polarimetric SAR analysis toolkit for NISAR L+S band data.

Modules:
    - polarimetric: H-α-A, Freeman-Durden, Yamaguchi decompositions
    - detection: CFAR target detection, classification
    - terrain: Land cover classification, trafficability
    - insar: InSAR processing, coherence, deformation
    - reports: PDF/HTML/Markdown report generation
    - export: GeoJSON, KML, GeoTIFF export
    - downloader: ASF DAAC data download

Example usage:
    from nisar import PolarimetricProcessor, TargetDetector
    
    processor = PolarimetricProcessor()
    result = processor.process_slc_quad_pol(hh, hv, vh, vv)
    
    detector = TargetDetector()
    targets = detector.detect_targets(result.hh, result.vv, ...)
"""

__version__ = '1.0.0'
__author__ = 'Hellas SAR Platform'

# Import main classes for convenience
try:
    from .polarimetric import PolarimetricProcessor, PolarimetricResult
except ImportError:
    PolarimetricProcessor = None
    PolarimetricResult = None

try:
    from .detection import TargetDetector, DetectedTarget, CFARDetector
except ImportError:
    TargetDetector = None
    DetectedTarget = None
    CFARDetector = None

try:
    from .terrain import TerrainClassifier, TerrainClass
except ImportError:
    TerrainClassifier = None
    TerrainClass = None

try:
    from .insar import InSARProcessor, InSARResult
except ImportError:
    InSARProcessor = None
    InSARResult = None

try:
    from .reports import ReportGenerator, ReportSection
except ImportError:
    ReportGenerator = None
    ReportSection = None

try:
    from .export import ExportManager
except ImportError:
    ExportManager = None

try:
    from .downloader import NISARDownloader, NISARProduct
except ImportError:
    NISARDownloader = None
    NISARProduct = None

__all__ = [
    # Polarimetric
    'PolarimetricProcessor',
    'PolarimetricResult',
    
    # Detection
    'TargetDetector',
    'DetectedTarget',
    'CFARDetector',
    
    # Terrain
    'TerrainClassifier',
    'TerrainClass',
    
    # InSAR
    'InSARProcessor',
    'InSARResult',
    
    # Reports
    'ReportGenerator',
    'ReportSection',
    
    # Export
    'ExportManager',
    
    # Downloader
    'NISARDownloader',
    'NISARProduct',
]
