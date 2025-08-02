"""
SentinelGem Source Package
Offline Multimodal Cybersecurity Assistant

Author: Muzan Sano
"""

from .inference import GemmaInference, ThreatAnalysis, get_inference_engine
from .audio_pipeline import get_audio_pipeline
from .ocr_pipeline import get_ocr_pipeline
from .utils import Timer, validate_input_file

__version__ = "1.0.0"
__author__ = "Muzan Sano"

__all__ = [
    "GemmaInference",
    "ThreatAnalysis", 
    "get_inference_engine",
    "get_audio_pipeline",
    "get_ocr_pipeline",
    "Timer",
    "validate_input_file"
]
