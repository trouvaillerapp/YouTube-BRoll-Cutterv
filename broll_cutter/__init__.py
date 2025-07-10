"""
YouTube B-Roll Cutter - AI-powered video clip extraction tool
"""

from .core import YouTubeBRollCutter
from .downloader import VideoDownloader
from .scene_detector import SceneDetector
from .watermark_remover import WatermarkRemover
from .video_processor import VideoProcessor

__version__ = "1.0.0"
__author__ = "Roy Antony"
__description__ = "AI-powered YouTube B-roll extraction tool"

__all__ = [
    "YouTubeBRollCutter",
    "VideoDownloader", 
    "SceneDetector",
    "WatermarkRemover",
    "VideoProcessor"
]