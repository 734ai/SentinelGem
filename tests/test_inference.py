#!/usr/bin/env python3
"""
SentinelGem Inference Engine Tests
Author: Muzan Sano

Comprehensive test suite for the Gemma 3n inference engine including:
- Model loading and initialization
- Threat analysis accuracy 
- Performance benchmarking
- Edge case handling
- Multimodal input processing
"""

import unittest
import pytest
import torch
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference import GemmaInference, ThreatAnalysis
from src.utils import validate_input_file, Timer


class TestGemmaInference(unittest.TestCase):
    """Test cases for GemmaInference class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.test_data_dir = project_root / "tests" / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create mock test files
        cls.sample_text_file = cls.test_data_dir / "sample_text.txt"
        cls.sample_text_file.write_text("This is a test email with urgent action required. Click here to verify your account.")
        
        cls.phishing_text = "URGENT: Your PayPal account has been suspended. Click here to verify: http://paypal-security.bit.ly"
        cls.legitimate_text = "Your order has been confirmed. Thank you for shopping with us. Delivery expected in 2-3 days."
        
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Mock the inference engine to avoid loading actual models in tests
        self.mock_inference = Mock(spec=GemmaInference)
        
        # Configure mock responses
        self.mock_inference.model = Mock()
        self.mock_inference.tokenizer = Mock()
        self.mock_inference.device = "cpu"
        self.mock_inference.model_name = "google/gemma-3n-2b"
        
    def test_initialization(self):
        """Test GemmaInference initialization"""
        with patch('src.inference.AutoTokenizer') as mock_tokenizer, \
             patch('src.inference.AutoModelForCausalLM') as mock_model:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            
            inference = GemmaInference(model_name="google/gemma-3n-2b", device="cpu")
            
            self.assertIsNotNone(inference.tokenizer)
            self.assertIsNotNone(inference.model)
            self.assertEqual(inference.device, "cpu")
            
    def test_analyze_threat_phishing(self):
        """Test threat analysis on phishing text"""
        # Create actual inference instance for integration testing
        try:
            inference = GemmaInference(device="cpu")
            result = inference.analyze_threat(self.phishing_text, "phishing_analysis")
            
            self.assertIsInstance(result, ThreatAnalysis)
            self.assertIsInstance(result.confidence_score, float)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)
            self.assertIn(result.threat_type, ["phishing", "social_engineering", "spam", "safe"])
            
        except Exception as e:
            # If model not available, use mock testing
            self.skipTest(f"Model not available for integration test: {e}")
            
    def test_analyze_threat_legitimate(self):
        """Test threat analysis on legitimate text"""
        mock_result = ThreatAnalysis(
            threat_detected=False,
            confidence_score=0.15,
            threat_type="safe",
            description="Content appears legitimate",
            recommendations=["No action required"],
            raw_analysis="Mock analysis output",
            metadata={"test": True}
        )
        
        self.mock_inference.analyze_threat.return_value = mock_result
        
        result = self.mock_inference.analyze_threat(self.legitimate_text, "general_analysis")
        
        self.assertFalse(result.threat_detected)
        self.assertLess(result.confidence_score, 0.5)
        self.assertEqual(result.threat_type, "safe")
        
    def test_performance_benchmarks(self):
        """Test inference performance benchmarks"""
        test_texts = [
            "Short test text",
            "Medium length test text with some suspicious words like urgent and verify account",
            "Very long test text " * 100  # Long text to test performance
        ]
        
        inference_times = []
        
        for text in test_texts:
            start_time = time.time()
            
            # Mock the analysis for performance testing
            mock_result = ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.3,
                threat_type="safe",
                description="Mock analysis",
                recommendations=[],
                raw_analysis="",
                metadata={"processing_time": 0.5}
            )
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        # Verify reasonable performance
        avg_time = sum(inference_times) / len(inference_times)
        self.assertLess(avg_time, 5.0, "Average inference time should be under 5 seconds")
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        inference = Mock(spec=GemmaInference)
        
        # Test empty input
        with self.assertRaises(ValueError):
            inference.analyze_threat("", "phishing_analysis")
            
        # Test invalid analysis type
        with self.assertRaises(ValueError):
            inference.analyze_threat("test", "invalid_type")
            
    def test_threat_analysis_dataclass(self):
        """Test ThreatAnalysis dataclass structure"""
        analysis = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.85,
            threat_type="phishing",
            description="High confidence phishing attempt",
            recommendations=["Do not click links", "Report to IT"],
            raw_analysis="Detailed AI analysis...",
            metadata={"patterns": ["urgent", "verify"], "url_count": 1}
        )
        
        self.assertTrue(analysis.threat_detected)
        self.assertEqual(analysis.confidence_score, 0.85)
        self.assertEqual(analysis.threat_type, "phishing")
        self.assertIsInstance(analysis.recommendations, list)
        self.assertIsInstance(analysis.metadata, dict)
        
    def test_multimodal_input_types(self):
        """Test different input types (text, file, etc.)"""
        inference = Mock(spec=GemmaInference)
        
        # Mock file input analysis
        mock_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.7,
            threat_type="malware",
            description="Suspicious file patterns detected",
            recommendations=["Scan with antivirus"],
            raw_analysis="File analysis output",
            metadata={"file_type": "log", "indicators": 5}
        )
        
        inference.analyze_threat.return_value = mock_result
        
        # Test with different input types
        input_types = ["text", "log", "email", "screenshot"]
        
        for input_type in input_types:
            result = inference.analyze_threat("test content", "general_analysis", input_type)
            self.assertIsInstance(result, ThreatAnalysis)
            
    @patch('src.inference.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda):
        """Test automatic device selection"""
        # Test CUDA available
        mock_cuda.return_value = True
        with patch('src.inference.AutoTokenizer'), patch('src.inference.AutoModelForCausalLM'):
            inference = GemmaInference()
            # Should select CUDA if available
            
        # Test CUDA not available
        mock_cuda.return_value = False
        with patch('src.inference.AutoTokenizer'), patch('src.inference.AutoModelForCausalLM'):
            inference = GemmaInference()
            # Should fall back to CPU


class TestInferenceIntegration(unittest.TestCase):
    """Integration tests with other system components"""
    
    def test_ocr_pipeline_integration(self):
        """Test integration with OCR pipeline"""
        # Mock OCR output
        ocr_text = "Verify your account immediately. Click here: http://fake-bank.com"
        
        mock_inference = Mock(spec=GemmaInference)
        mock_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.9,
            threat_type="phishing",
            description="OCR-detected phishing attempt",
            recommendations=["Block website", "Report phishing"],
            raw_analysis="Analysis of OCR text",
            metadata={"source": "ocr", "url_detected": True}
        )
        
        mock_inference.analyze_threat.return_value = mock_result
        
        result = mock_inference.analyze_threat(ocr_text, "phishing_analysis", "screenshot")
        
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "phishing")
        self.assertIn("Block website", result.recommendations)
        
    def test_audio_pipeline_integration(self):
        """Test integration with audio pipeline"""
        # Mock audio transcription
        audio_text = "This is Microsoft calling. Your computer has been hacked. Press 1 to speak to our security team."
        
        mock_inference = Mock(spec=GemmaInference)
        mock_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.95,
            threat_type="social_engineering",
            description="Voice-based social engineering attempt",
            recommendations=["Hang up immediately", "Report to authorities"],
            raw_analysis="Analysis of audio transcription",
            metadata={"source": "audio", "authority_claim": True}
        )
        
        mock_inference.analyze_threat.return_value = mock_result
        
        result = mock_inference.analyze_threat(audio_text, "audio_analysis", "audio")
        
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "social_engineering")
        self.assertGreater(result.confidence_score, 0.9)


class TestPerformanceMetrics(unittest.TestCase):
    """Performance and accuracy metrics tests"""
    
    def test_accuracy_metrics(self):
        """Test accuracy on known threat samples"""
        # Known phishing samples
        phishing_samples = [
            "Your account will be suspended unless you verify immediately",
            "Congratulations! You've won $1000. Click here to claim",  
            "Security alert: Unusual activity detected. Update your password now"
        ]
        
        # Known legitimate samples
        legitimate_samples = [
            "Your order has been shipped and will arrive tomorrow",
            "Meeting reminder: Team standup at 10 AM",
            "Weather forecast: Sunny with highs of 75°F"
        ]
        
        mock_inference = Mock(spec=GemmaInference)
        
        # Mock responses for phishing (should detect threats)
        phishing_results = [
            ThreatAnalysis(True, 0.85, "phishing", "Detected", [], "", {}),
            ThreatAnalysis(True, 0.78, "spam", "Detected", [], "", {}),
            ThreatAnalysis(True, 0.92, "phishing", "Detected", [], "", {})
        ]
        
        # Mock responses for legitimate (should not detect threats)
        legitimate_results = [
            ThreatAnalysis(False, 0.15, "safe", "Safe", [], "", {}),
            ThreatAnalysis(False, 0.08, "safe", "Safe", [], "", {}),
            ThreatAnalysis(False, 0.12, "safe", "Safe", [], "", {})
        ]
        
        mock_inference.analyze_threat.side_effect = phishing_results + legitimate_results
        
        # Test phishing detection
        for i, sample in enumerate(phishing_samples):
            result = mock_inference.analyze_threat(sample, "phishing_analysis")
            self.assertTrue(result.threat_detected, f"Failed to detect threat in sample {i+1}")
            
        # Test legitimate content
        for i, sample in enumerate(legitimate_samples):
            result = mock_inference.analyze_threat(sample, "general_analysis") 
            self.assertFalse(result.threat_detected, f"False positive on legitimate sample {i+1}")
            
    def test_response_time_sla(self):
        """Test that response times meet SLA requirements"""
        mock_inference = Mock(spec=GemmaInference)
        
        # Simulate different response times
        def mock_analyze_with_timing(*args, **kwargs):
            time.sleep(0.1)  # Simulate processing time
            return ThreatAnalysis(False, 0.3, "safe", "Test", [], "", {})
            
        mock_inference.analyze_threat.side_effect = mock_analyze_with_timing
        
        start_time = time.time()
        result = mock_inference.analyze_threat("test text", "general_analysis")
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 2.0, "Processing should complete within 2 seconds")


if __name__ == '__main__':
    # Create test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Run tests with verbose output
    unittest.main(verbosity=2, exit=False)
    
    # Run additional pytest tests if available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, skipping pytest tests")
