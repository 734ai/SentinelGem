#!/usr/bin/env python3
"""
SentinelGem OCR Pipeline Tests
Author: Muzan Sano

Comprehensive test suite for OCR pipeline including:
- Image preprocessing accuracy
- Text extraction quality
- Phishing pattern detection
- Performance benchmarks
- Edge case handling
"""

import unittest
import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ocr_pipeline import OCRPipeline
from src.inference import ThreatAnalysis


class TestOCRPipeline(unittest.TestCase):
    """Test cases for OCR Pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data_dir = project_root / "tests" / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create test images
        cls.create_test_images()
        
    @classmethod
    def create_test_images(cls):
        """Create test images for OCR testing"""
        # Create a simple text image
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "URGENT: Verify Account", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Click: paypal-secure.bit.ly", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cls.phishing_image_path = cls.test_data_dir / "phishing_test.png"
        cv2.imwrite(str(cls.phishing_image_path), img)
        
        # Create legitimate image
        img2 = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img2, "Order Confirmation", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img2, "Thank you for shopping", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cls.legitimate_image_path = cls.test_data_dir / "legitimate_test.png"
        cv2.imwrite(str(cls.legitimate_image_path), img2)
        
        # Create low quality image
        img3 = np.ones((100, 200, 3), dtype=np.uint8) * 200
        noise = np.random.randint(0, 100, img3.shape, dtype=np.uint8)
        img3 = cv2.add(img3, noise)
        cv2.putText(img3, "Blurry text", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cls.low_quality_image_path = cls.test_data_dir / "low_quality_test.png"
        cv2.imwrite(str(cls.low_quality_image_path), img3)
        
    def setUp(self):
        """Set up test fixtures"""
        # Use mock inference to avoid loading actual models
        with patch('src.ocr_pipeline.get_inference_engine'):
            self.ocr_pipeline = OCRPipeline(confidence_threshold=0.7)
            
        # Mock the inference engine
        self.mock_inference = Mock()
        self.ocr_pipeline.inference = self.mock_inference
        
    def test_initialization(self):
        """Test OCRPipeline initialization"""
        self.assertEqual(self.ocr_pipeline.confidence_threshold, 0.7)
        self.assertTrue(self.ocr_pipeline.preprocessing)
        self.assertIsNotNone(self.ocr_pipeline.phishing_indicators)
        
    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        # Load test image
        image = cv2.imread(str(self.phishing_image_path))
        
        # Test preprocessing
        processed = self.ocr_pipeline.preprocess_image(image)
        
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        
    @patch('src.ocr_pipeline.pytesseract.image_to_data')
    @patch('src.ocr_pipeline.pytesseract.image_to_string')
    def test_text_extraction(self, mock_image_to_string, mock_image_to_data):
        """Test OCR text extraction"""
        # Mock Tesseract responses
        mock_image_to_data.return_value = {
            'text': ['URGENT:', 'Verify', 'Account', 'Click:', 'paypal-secure.bit.ly'],
            'conf': [95, 90, 92, 88, 85]
        }
        mock_image_to_string.return_value = "URGENT: Verify Account Click: paypal-secure.bit.ly"
        
        result = self.ocr_pipeline.extract_text(str(self.phishing_image_path))
        
        self.assertTrue(result["success"])
        self.assertIn("urgent", result["text"].lower())
        self.assertIn("verify", result["text"].lower())
        self.assertGreater(result["confidence"], 0.5)
        
    def test_phishing_pattern_detection(self):
        """Test phishing pattern detection"""
        test_text = "URGENT: Your PayPal account has been suspended. Click here to verify: http://paypal-secure.bit.ly"
        
        result = self.ocr_pipeline.detect_phishing_patterns(test_text)
        
        self.assertGreater(result["phishing_score"], 0.5)
        self.assertIn("urgency_words", result["detected_patterns"])
        self.assertIn("credential_requests", result["detected_patterns"])
        self.assertIn("suspicious_domains", result["detected_patterns"])
        self.assertGreater(len(result["suspicious_urls"]), 0)
        
    def test_legitimate_content_detection(self):
        """Test that legitimate content is not flagged"""
        legitimate_text = "Your order #12345 has been confirmed. Expected delivery: 2-3 business days. Thank you for shopping with us!"
        
        result = self.ocr_pipeline.detect_phishing_patterns(legitimate_text)
        
        self.assertLess(result["phishing_score"], 0.3)
        self.assertEqual(len(result["suspicious_urls"]), 0)
        
    @patch('src.ocr_pipeline.pytesseract.image_to_data')
    @patch('src.ocr_pipeline.pytesseract.image_to_string')
    def test_full_screenshot_analysis(self, mock_image_to_string, mock_image_to_data):
        """Test complete screenshot analysis pipeline"""
        # Mock OCR output
        mock_image_to_data.return_value = {
            'text': ['URGENT:', 'Account', 'Suspended', 'Verify', 'Now'],
            'conf': [90, 85, 88, 92, 87]
        }
        mock_image_to_string.return_value = "URGENT: Account Suspended Verify Now"
        
        # Mock AI analysis
        mock_ai_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.85,
            threat_type="phishing",
            description="Phishing attempt detected",
            recommendations=["Do not click", "Report"],
            raw_analysis="AI detected phishing patterns",
            metadata={}
        )
        self.mock_inference.analyze_threat.return_value = mock_ai_result
        
        result = self.ocr_pipeline.analyze_screenshot(str(self.phishing_image_path))
        
        self.assertIsInstance(result, ThreatAnalysis)
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "phishing")
        self.assertGreater(result.confidence_score, 0.7)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test non-existent file
        result = self.ocr_pipeline.extract_text("nonexistent.jpg")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Test invalid image format
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"This is not an image")
            temp_file.flush()
            
            result = self.ocr_pipeline.extract_text(temp_file.name)
            self.assertFalse(result["success"])
            
    def test_batch_analysis(self):
        """Test batch screenshot analysis"""
        image_paths = [
            str(self.phishing_image_path),
            str(self.legitimate_image_path)
        ]
        
        # Mock AI responses
        mock_results = [
            ThreatAnalysis(True, 0.9, "phishing", "Phishing", [], "", {}),
            ThreatAnalysis(False, 0.2, "safe", "Safe", [], "", {})
        ]
        self.mock_inference.analyze_threat.side_effect = mock_results
        
        with patch('src.ocr_pipeline.pytesseract.image_to_data'), \
             patch('src.ocr_pipeline.pytesseract.image_to_string'):
            results = self.ocr_pipeline.batch_analyze_screenshots(image_paths)
            
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ThreatAnalysis)
        self.assertIsInstance(results[1], ThreatAnalysis)
        
    def test_confidence_thresholding(self):
        """Test confidence threshold enforcement"""
        # Test with different confidence thresholds
        low_confidence_pipeline = OCRPipeline(confidence_threshold=0.3)
        high_confidence_pipeline = OCRPipeline(confidence_threshold=0.9)
        
        # Mock low confidence OCR result
        with patch('src.ocr_pipeline.pytesseract.image_to_data') as mock_data:
            mock_data.return_value = {
                'text': ['unclear', 'text'],
                'conf': [25, 30]  # Low confidence
            }
            
            low_result = low_confidence_pipeline.extract_text(str(self.phishing_image_path))
            high_result = high_confidence_pipeline.extract_text(str(self.phishing_image_path))
            
            # Low threshold should accept low confidence text
            # High threshold should reject it
            self.assertGreater(len(low_result.get("text", "")), 0)
            
    def test_url_extraction(self):
        """Test URL extraction from OCR text"""
        text_with_urls = """
        Visit our secure site at https://legitimate-bank.com
        Or click here: http://suspicious-site.bit.ly/verify
        Contact us at support@company.com
        """
        
        result = self.ocr_pipeline.detect_phishing_patterns(text_with_urls)
        
        self.assertGreater(len(result["all_urls"]), 0)
        self.assertIn("bit.ly", str(result["suspicious_urls"]))
        
    def test_performance_metrics(self):
        """Test OCR performance metrics"""
        # Test processing time
        import time
        
        start_time = time.time()
        with patch('src.ocr_pipeline.pytesseract.image_to_data'), \
             patch('src.ocr_pipeline.pytesseract.image_to_string'):
            result = self.ocr_pipeline.extract_text(str(self.phishing_image_path))
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 5.0, "OCR should complete within 5 seconds")
        
    def test_pipeline_statistics(self):
        """Test pipeline statistics and configuration"""
        stats = self.ocr_pipeline.get_pipeline_stats()
        
        self.assertIn("confidence_threshold", stats)
        self.assertIn("preprocessing_enabled", stats)
        self.assertIn("phishing_patterns", stats)
        self.assertEqual(stats["confidence_threshold"], 0.7)


class TestOCRIntegration(unittest.TestCase):
    """Integration tests for OCR pipeline"""
    
    def test_agent_integration(self):
        """Test integration with agent system"""
        # Mock agent calling OCR pipeline
        with patch('src.ocr_pipeline.get_inference_engine'):
            ocr_pipeline = OCRPipeline()
            
        # Mock successful analysis
        mock_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.8,
            threat_type="phishing",
            description="Screenshot phishing detected",
            recommendations=["Close browser", "Report"],
            raw_analysis="OCR integration test",
            metadata={"source": "ocr_pipeline"}
        )
        
        ocr_pipeline.inference = Mock()
        ocr_pipeline.inference.analyze_threat.return_value = mock_result
        
        with patch('src.ocr_pipeline.pytesseract.image_to_data'), \
             patch('src.ocr_pipeline.pytesseract.image_to_string'):
            result = ocr_pipeline.analyze_screenshot("test_image.png")
            
        self.assertEqual(result.metadata["source"], "ocr_pipeline")
        
    def test_real_world_scenarios(self):
        """Test real-world phishing scenarios"""
        # Common phishing scenarios
        scenarios = [
            {
                "text": "Your Amazon account has been locked. Verify here: amazon-security.tk",
                "expected_threat": True,
                "threat_type": "phishing"
            },
            {
                "text": "Microsoft Security Alert: Your PC is infected. Call +1-800-FAKE",
                "expected_threat": True,
                "threat_type": "tech_support_scam"
            },
            {
                "text": "Welcome to our newsletter! Unsubscribe anytime.",
                "expected_threat": False,
                "threat_type": "safe"
            }
        ]
        
        with patch('src.ocr_pipeline.get_inference_engine'):
            ocr_pipeline = OCRPipeline()
            
        for scenario in scenarios:
            result = ocr_pipeline.detect_phishing_patterns(scenario["text"])
            
            if scenario["expected_threat"]:
                self.assertGreater(result["phishing_score"], 0.5, 
                                 f"Failed to detect threat in: {scenario['text']}")
            else:
                self.assertLess(result["phishing_score"], 0.3, 
                               f"False positive on: {scenario['text']}")


if __name__ == '__main__':
    # Ensure test data directory exists
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, skipping pytest tests")
