"""
SentinelGem OCR Pipeline
Author: Muzan Sano

Screenshot analysis and phishing detection using OCR + Gemma 3n
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from typing import Dict, List, Optional, Tuple, Any
import logging
import re
from pathlib import Path

from rich.console import Console
from rich.progress import track

from .utils import Timer, validate_input_file, extract_urls, clean_text, logger
from .inference import get_inference_engine, ThreatAnalysis

console = Console()

class OCRPipeline:
    """
    OCR pipeline for screenshot analysis and phishing detection
    """
    
    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        preprocessing: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.preprocessing = preprocessing
        
        # Configure Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize inference engine (optional for testing)
        try:
            self.inference = get_inference_engine()
        except Exception as e:
            logger.warning(f"Could not initialize inference engine: {e}")
            self.inference = None
        
        # Phishing keywords and patterns
        self.phishing_indicators = self._load_phishing_patterns()
        
        logger.info("OCR Pipeline initialized")
    
    def _load_phishing_patterns(self) -> Dict[str, List[str]]:
        """Load phishing detection patterns"""
        return {
            "urgency_words": [
                "urgent", "immediate", "expire", "suspended", "verify now",
                "act now", "limited time", "expires today", "final notice"
            ],
            "credential_requests": [
                "login", "password", "username", "credit card", "ssn",
                "social security", "bank account", "pin", "verification code"
            ],
            "suspicious_domains": [
                "bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly",
                "short.link", "click.here", "secure-bank", "paypal-secure"
            ],
            "action_words": [
                "click here", "download now", "update payment", "confirm account",
                "verify identity", "update security", "claim reward"
            ],
            "brand_impersonation": [
                "paypal", "amazon", "microsoft", "google", "apple", "netflix",
                "facebook", "instagram", "twitter", "linkedin", "dropbox"
            ]
        }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to PIL Image for enhancement
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Convert back to numpy array
            enhanced = np.array(pil_image)
            
            # Convert to grayscale if color
            if len(enhanced.shape) == 3:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
            else:
                gray = enhanced
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply adaptive thresholding
            threshold = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Validate input file
            validation = validate_input_file(image_path, max_size_mb=25)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "text": "",
                    "confidence": 0.0
                }
            
            with Timer(f"OCR extraction from {Path(image_path).name}"):
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    # Try with PIL for different formats
                    pil_image = Image.open(image_path)
                    image = np.array(pil_image)
                
                # Preprocess if enabled
                if self.preprocessing:
                    processed_image = self.preprocess_image(image)
                else:
                    processed_image = image
                
                # Extract text with confidence scores
                ocr_data = pytesseract.image_to_data(
                    processed_image, 
                    output_type=pytesseract.Output.DICT
                )
                
                # Filter text by confidence
                texts = []
                confidences = []
                
                for i in range(len(ocr_data['text'])):
                    confidence = int(ocr_data['conf'][i])
                    text = ocr_data['text'][i].strip()
                    
                    if confidence > 30 and len(text) > 1:  # Filter low confidence
                        texts.append(text)
                        confidences.append(confidence)
                
                # Combine text
                full_text = ' '.join(texts)
                full_text = clean_text(full_text)
                
                # Calculate average confidence
                avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
                
                return {
                    "success": True,
                    "text": full_text,
                    "confidence": avg_confidence,
                    "word_count": len(texts),
                    "avg_word_confidence": avg_confidence,
                    "extracted_words": len([t for t in texts if len(t) > 2]),
                    "image_info": validation["info"]
                }
                
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0
            }
    
    def detect_phishing_patterns(self, text: str) -> Dict[str, Any]:
        """
        Detect phishing patterns in extracted text
        
        Args:
            text: Extracted text from image
            
        Returns:
            Pattern detection results
        """
        text_lower = text.lower()
        detected_patterns = {}
        total_score = 0.0
        
        # Check each pattern category
        for category, patterns in self.phishing_indicators.items():
            matches = []
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    matches.append(pattern)
            
            if matches:
                detected_patterns[category] = matches
                # Weight different categories
                if category == "credential_requests":
                    total_score += len(matches) * 0.3
                elif category == "urgency_words":
                    total_score += len(matches) * 0.2
                elif category == "suspicious_domains":
                    total_score += len(matches) * 0.4
                elif category == "action_words":
                    total_score += len(matches) * 0.15
                elif category == "brand_impersonation":
                    total_score += len(matches) * 0.25
        
        # Extract and analyze URLs
        urls = extract_urls(text)
        suspicious_urls = []
        
        for url in urls:
            url_lower = url.lower()
            # Check for suspicious URL patterns
            if any(domain in url_lower for domain in self.phishing_indicators["suspicious_domains"]):
                suspicious_urls.append(url)
                total_score += 0.3
            
            # Check for URL shorteners
            if any(short in url_lower for short in ["bit.ly", "tinyurl", "t.co"]):
                suspicious_urls.append(url)
                total_score += 0.2
        
        # Normalize score
        phishing_score = min(total_score, 1.0)
        
        return {
            "phishing_score": phishing_score,
            "detected_patterns": detected_patterns,
            "suspicious_urls": suspicious_urls,
            "all_urls": urls,
            "pattern_count": sum(len(patterns) for patterns in detected_patterns.values())
        }
    
    def analyze_screenshot(self, image_path: str) -> ThreatAnalysis:
        """
        Complete screenshot analysis pipeline
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            ThreatAnalysis with complete results
        """
        try:
            console.print(f"[blue]Analyzing screenshot: {Path(image_path).name}[/blue]")
            
            # Extract text using OCR
            ocr_result = self.extract_text(image_path)
            
            if not ocr_result["success"]:
                return ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="ocr_error",
                    description=f"OCR failed: {ocr_result.get('error', 'Unknown error')}",
                    recommendations=["Check image quality and format"],
                    raw_analysis="",
                    metadata={"ocr_error": ocr_result.get("error")}
                )
            
            extracted_text = ocr_result["text"]
            
            if not extracted_text or len(extracted_text) < 10:
                return ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="insufficient_text",
                    description="Insufficient text extracted from image",
                    recommendations=["Ensure image contains readable text"],
                    raw_analysis="",
                    metadata={"ocr_result": ocr_result}
                )
            
            # Pattern-based phishing detection
            pattern_analysis = self.detect_phishing_patterns(extracted_text)
            
            # AI-powered analysis using Gemma 3n
            ai_analysis = self.inference.analyze_threat(
                extracted_text, 
                analysis_type="phishing_analysis",
                input_type="screenshot"
            )
            
            # Combine pattern and AI analysis
            combined_confidence = (pattern_analysis["phishing_score"] + ai_analysis.confidence_score) / 2
            
            # Determine final threat status
            threat_detected = (
                combined_confidence > self.confidence_threshold or
                pattern_analysis["phishing_score"] > 0.6 or
                ai_analysis.threat_detected
            )
            
            # Enhanced description
            description_parts = [ai_analysis.description]
            
            if pattern_analysis["detected_patterns"]:
                pattern_desc = "Detected patterns: " + ", ".join(
                    f"{cat}({len(patterns)})" 
                    for cat, patterns in pattern_analysis["detected_patterns"].items()
                )
                description_parts.append(pattern_desc)
            
            if pattern_analysis["suspicious_urls"]:
                url_desc = f"Suspicious URLs found: {len(pattern_analysis['suspicious_urls'])}"
                description_parts.append(url_desc)
            
            # Enhanced recommendations
            recommendations = list(ai_analysis.recommendations)
            
            if threat_detected:
                recommendations.extend([
                    "Do not click any links in this content",
                    "Verify sender through alternative communication",
                    "Report as potential phishing attempt"
                ])
            
            return ThreatAnalysis(
                threat_detected=threat_detected,
                confidence_score=combined_confidence,
                threat_type="phishing" if threat_detected else "safe",
                description=" | ".join(description_parts),
                recommendations=recommendations,
                raw_analysis=ai_analysis.raw_analysis,
                metadata={
                    "ocr_result": ocr_result,
                    "pattern_analysis": pattern_analysis,
                    "ai_analysis": {
                        "confidence": ai_analysis.confidence_score,
                        "threat_type": ai_analysis.threat_type
                    },
                    "extracted_text_length": len(extracted_text),
                    "combined_confidence": combined_confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.0,
                threat_type="analysis_error",
                description=f"Analysis failed: {str(e)}",
                recommendations=["Check image file and try again"],
                raw_analysis="",
                metadata={"error": str(e)}
            )
    
    def batch_analyze_screenshots(self, image_paths: List[str]) -> List[ThreatAnalysis]:
        """Analyze multiple screenshots in batch"""
        results = []
        
        for image_path in track(image_paths, description="Analyzing screenshots..."):
            result = self.analyze_screenshot(image_path)
            results.append(result)
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and configuration"""
        return {
            "confidence_threshold": self.confidence_threshold,
            "preprocessing_enabled": self.preprocessing,
            "phishing_patterns": {
                cat: len(patterns) for cat, patterns in self.phishing_indicators.items()
            },
            "tesseract_version": pytesseract.get_tesseract_version(),
            "opencv_version": cv2.__version__
        }

# Global OCR pipeline instance
_ocr_pipeline = None

def get_ocr_pipeline(**kwargs) -> OCRPipeline:
    """Get or create global OCR pipeline instance"""
    global _ocr_pipeline
    
    if _ocr_pipeline is None:
        _ocr_pipeline = OCRPipeline(**kwargs)
    
    return _ocr_pipeline

if __name__ == "__main__":
    # Test OCR pipeline
    console.print("[bold cyan]Testing SentinelGem OCR Pipeline[/bold cyan]")
    
    # Initialize pipeline
    pipeline = OCRPipeline()
    
    # Display pipeline stats
    stats = pipeline.get_pipeline_stats()
    console.print(f"[green]Pipeline initialized with {sum(stats['phishing_patterns'].values())} phishing patterns[/green]")
    
    console.print("OCR Pipeline ready for screenshot analysis!")
