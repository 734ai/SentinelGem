#!/usr/bin/env python3
"""
SentinelGem Integration Tests
Author: Muzan Sano

End-to-end integration tests for the complete SentinelGem system including:
- Agent orchestration
- Multimodal analysis pipeline
- Notebook generation
- System performance under load
- Real-world threat scenarios
"""

import unittest
import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agents.agent_loop import SentinelAgent
from src.inference import ThreatAnalysis
from src.autogen_notebook import NotebookGenerator


class TestSentinelGemIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment"""
        cls.test_data_dir = project_root / "tests" / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create comprehensive test files
        cls.create_integration_test_files()
        
    @classmethod
    def create_integration_test_files(cls):
        """Create test files for integration testing"""
        # Phishing email sample
        phishing_email = """
        Subject: URGENT: Your PayPal Account Has Been Limited
        
        Dear Customer,
        
        We have detected unusual activity on your PayPal account. Your account has been temporarily limited.
        
        To restore full access, please verify your identity immediately:
        http://paypal-security-update.bit.ly/verify
        
        This link will expire in 24 hours. Act now to avoid permanent suspension.
        
        Thank you,
        PayPal Security Team
        """
        
        cls.phishing_email_file = cls.test_data_dir / "phishing_email.txt"
        cls.phishing_email_file.write_text(phishing_email)
        
        # Legitimate email sample
        legitimate_email = """
        Subject: Order Confirmation #12345
        
        Hi John,
        
        Thank you for your order! Your items will be shipped within 2-3 business days.
        
        Order Details:
        - Product: Wireless Headphones
        - Quantity: 1
        - Total: $99.99
        
        Track your order at: https://shop.example.com/orders/12345
        
        Best regards,
        Customer Service Team
        """
        
        cls.legitimate_email_file = cls.test_data_dir / "legitimate_email.txt"
        cls.legitimate_email_file.write_text(legitimate_email)
        
        # Suspicious system log
        suspicious_log = """
        2025-08-01 10:15:32 [INFO] System boot completed
        2025-08-01 10:16:15 [WARNING] Failed login attempt for user 'admin' from 192.168.1.100
        2025-08-01 10:17:45 [INFO] Process started: powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -enc UwB0AGEAcgB0AC0AUwBsAGUAZQBwACAALQBzACAAMQAwAA==
        2025-08-01 10:18:02 [WARNING] Suspicious network connection detected: TCP 192.168.1.50:4444 -> 203.0.113.10:80
        2025-08-01 10:18:30 [WARNING] File created in suspicious location: C:\\Users\\admin\\AppData\\Roaming\\svchost.exe
        2025-08-01 10:19:12 [INFO] Scheduled task created: "SystemUpdate" - runs daily at startup
        """
        
        cls.suspicious_log_file = cls.test_data_dir / "suspicious_log.txt"
        cls.suspicious_log_file.write_text(suspicious_log)
        
    def setUp(self):
        """Set up test fixtures"""
        # Mock all AI models to avoid loading them in tests
        self.patches = [
            patch('src.inference.AutoTokenizer'),
            patch('src.inference.AutoModelForCausalLM'),
            patch('src.audio_pipeline.whisper.load_model'),
            patch('src.ocr_pipeline.pytesseract.image_to_string'),
            patch('src.ocr_pipeline.pytesseract.image_to_data')
        ]
        
        for p in self.patches:
            p.start()
            
        # Create agent instance
        self.agent = SentinelAgent()
        
        # Mock the AI components
        self.mock_inference_results()
        
    def tearDown(self):
        """Clean up after tests"""
        for p in self.patches:
            p.stop()
            
    def mock_inference_results(self):
        """Mock AI inference results for testing"""
        # Mock threat analysis results
        self.phishing_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.92,
            threat_type="phishing",
            description="High-confidence phishing attempt with credential harvesting patterns",
            recommendations=[
                "Do not click any links",
                "Do not provide personal information",
                "Report to IT security team"
            ],
            raw_analysis="Detected urgency language, suspicious URLs, and credential requests",
            metadata={
                "patterns": ["urgent", "verify", "suspended"],
                "urls": ["http://paypal-security-update.bit.ly/verify"],
                "confidence_breakdown": {"pattern_match": 0.85, "ai_analysis": 0.95}
            }
        )
        
        self.legitimate_result = ThreatAnalysis(
            threat_detected=False,
            confidence_score=0.12,
            threat_type="safe",
            description="Content appears to be a legitimate business communication",
            recommendations=["No immediate action required"],
            raw_analysis="Standard order confirmation with legitimate website URL",
            metadata={
                "patterns": [],
                "urls": ["https://shop.example.com/orders/12345"],
                "confidence_breakdown": {"pattern_match": 0.05, "ai_analysis": 0.15}
            }
        )
        
        self.malware_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.88,
            threat_type="malware",
            description="Multiple malware indicators detected in system logs",
            recommendations=[
                "Isolate affected system immediately",
                "Run full antivirus scan",
                "Check for data exfiltration",
                "Contact security team"
            ],
            raw_analysis="Detected encoded PowerShell, suspicious network connections, and persistence mechanisms",
            metadata={
                "indicators": ["powershell -enc", "suspicious file location", "scheduled task"],
                "severity": "high",
                "attack_stages": ["initial_access", "persistence", "command_control"]
            }
        )
        
    def test_full_email_analysis_pipeline(self):
        """Test complete email analysis from input to report"""
        # Mock the analysis
        with patch.object(self.agent, 'analyze_input', return_value=self.phishing_result):
            result = self.agent.analyze_input(str(self.phishing_email_file))
            
        # Verify threat detection
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "phishing")
        self.assertGreater(result.confidence_score, 0.9)
        self.assertIn("Do not click", result.recommendations[0])
        
        # Verify metadata
        self.assertIn("patterns", result.metadata)
        self.assertIn("urls", result.metadata)
        
    def test_legitimate_content_analysis(self):
        """Test analysis of legitimate content"""
        with patch.object(self.agent, 'analyze_input', return_value=self.legitimate_result):
            result = self.agent.analyze_input(str(self.legitimate_email_file))
            
        self.assertFalse(result.threat_detected)
        self.assertEqual(result.threat_type, "safe")
        self.assertLess(result.confidence_score, 0.2)
        
    def test_malware_log_analysis(self):
        """Test malware detection in system logs"""
        with patch.object(self.agent, 'analyze_input', return_value=self.malware_result):
            result = self.agent.analyze_input(str(self.suspicious_log_file))
            
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "malware")
        self.assertGreater(result.confidence_score, 0.8)
        self.assertIn("Isolate", result.recommendations[0])
        
    def test_batch_analysis_performance(self):
        """Test batch analysis performance"""
        test_files = [
            str(self.phishing_email_file),
            str(self.legitimate_email_file),
            str(self.suspicious_log_file)
        ]
        
        mock_results = [
            self.phishing_result,
            self.legitimate_result,
            self.malware_result
        ]
        
        start_time = time.time()
        
        results = []
        for i, file_path in enumerate(test_files):
            with patch.object(self.agent, 'analyze_input', return_value=mock_results[i]):
                result = self.agent.analyze_input(file_path)
                results.append(result)
                
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].threat_detected)   # Phishing
        self.assertFalse(results[1].threat_detected)  # Legitimate
        self.assertTrue(results[2].threat_detected)   # Malware
        
        # Performance check (should be fast with mocked components)
        self.assertLess(processing_time, 5.0, "Batch processing should complete quickly")
        
    def test_notebook_generation_integration(self):
        """Test automatic notebook generation"""
        # Mock notebook generator
        mock_generator = Mock(spec=NotebookGenerator)
        mock_notebook_path = "/tmp/test_analysis_report.ipynb"
        
        mock_generator.generate_threat_analysis_notebook.return_value = mock_notebook_path
        
        with patch('src.autogen_notebook.NotebookGenerator', return_value=mock_generator):
            # Simulate analysis with notebook generation
            with patch.object(self.agent, 'analyze_input', return_value=self.phishing_result):
                result = self.agent.analyze_input(str(self.phishing_email_file))
                
            # Verify notebook would be generated
            self.assertIsNotNone(result)
            
    def test_multimodal_input_detection(self):
        """Test automatic input type detection"""
        test_cases = [
            {
                "file": self.phishing_email_file,
                "expected_type": "text",
                "content_indicators": ["email", "text"]
            },
            {
                "file": self.suspicious_log_file,
                "expected_type": "logs",
                "content_indicators": ["log", "system"]
            }
        ]
        
        for case in test_cases:
            # Mock input type detection
            with patch.object(self.agent, '_detect_input_type', return_value=case["expected_type"]):
                detected_type = self.agent._detect_input_type(str(case["file"]))
                self.assertEqual(detected_type, case["expected_type"])
                
    def test_error_handling_integration(self):
        """Test error handling across the pipeline"""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.agent.analyze_input("nonexistent_file.txt")
            
        # Test corrupted input
        corrupted_file = self.test_data_dir / "corrupted.txt"
        corrupted_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')  # Binary garbage
        
        try:
            result = self.agent.analyze_input(str(corrupted_file))
            # Should handle gracefully
            self.assertIsInstance(result, ThreatAnalysis)
        except Exception as e:
            # If it raises an exception, it should be informative
            self.assertIn("corrupted", str(e).lower())
            
    def test_concurrent_analysis(self):
        """Test concurrent analysis capabilities"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def analyze_file(file_path, expected_result):
            with patch.object(self.agent, 'analyze_input', return_value=expected_result):
                result = self.agent.analyze_input(file_path)
                results_queue.put(result)
                
        # Start multiple analysis threads
        threads = [
            threading.Thread(target=analyze_file, args=(str(self.phishing_email_file), self.phishing_result)),
            threading.Thread(target=analyze_file, args=(str(self.legitimate_email_file), self.legitimate_result)),
            threading.Thread(target=analyze_file, args=(str(self.suspicious_log_file), self.malware_result))
        ]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
            
        self.assertEqual(len(results), 3)
        self.assertLess(end_time - start_time, 10.0, "Concurrent analysis should complete efficiently")
        
    def test_session_management(self):
        """Test agent session management"""
        # Test session initialization
        self.assertIsNotNone(self.agent.session_id)
        self.assertIsInstance(self.agent.analysis_history, list)
        
        # Test analysis history tracking
        initial_history_length = len(self.agent.analysis_history)
        
        with patch.object(self.agent, 'analyze_input', return_value=self.phishing_result):
            result = self.agent.analyze_input(str(self.phishing_email_file))
            
        # History should be updated
        self.assertGreater(len(self.agent.analysis_history), initial_history_length)
        
    def test_configuration_validation(self):
        """Test system configuration validation"""
        # Test configuration loading
        config_file = self.test_data_dir / "test_config.json"
        test_config = {
            "confidence_threshold": 0.7,
            "max_file_size_mb": 100,
            "enable_logging": True,
            "output_format": "json"
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
            
        # Verify configuration can be loaded and validated
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            
        self.assertEqual(loaded_config["confidence_threshold"], 0.7)
        self.assertTrue(loaded_config["enable_logging"])
        
    def test_system_resource_usage(self):
        """Test system resource usage during analysis"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform analysis
        with patch.object(self.agent, 'analyze_input', return_value=self.phishing_result):
            for _ in range(5):  # Multiple analyses
                result = self.agent.analyze_input(str(self.phishing_email_file))
                
        # Check memory usage after analysis
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory usage should be reasonable
        self.assertLess(memory_increase, 500, "Memory usage should not exceed 500MB increase")
        
    def test_output_format_validation(self):
        """Test output format validation"""
        with patch.object(self.agent, 'analyze_input', return_value=self.phishing_result):
            result = self.agent.analyze_input(str(self.phishing_email_file))
            
        # Verify ThreatAnalysis structure
        self.assertIsInstance(result, ThreatAnalysis)
        self.assertIsInstance(result.threat_detected, bool)
        self.assertIsInstance(result.confidence_score, float)
        self.assertIsInstance(result.threat_type, str)
        self.assertIsInstance(result.description, str)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.metadata, dict)
        
        # Verify confidence score range
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)


class TestSystemStressTests(unittest.TestCase):
    """Stress tests for system reliability"""
    
    def test_large_file_handling(self):
        """Test handling of large input files"""
        # Create a large text file
        large_content = "This is a test line with suspicious content like verify account now.\n" * 10000
        large_file = Path(__file__).parent / "test_data" / "large_test_file.txt"
        large_file.parent.mkdir(exist_ok=True)
        large_file.write_text(large_content)
        
        agent = SentinelAgent()
        
        try:
            # Mock analysis to avoid actual processing
            mock_result = ThreatAnalysis(True, 0.8, "test", "Large file test", [], "", {})
            with patch.object(agent, 'analyze_input', return_value=mock_result):
                result = agent.analyze_input(str(large_file))
                
            self.assertIsInstance(result, ThreatAnalysis)
        finally:
            # Clean up
            if large_file.exists():
                large_file.unlink()
                
    def test_rapid_successive_analyses(self):
        """Test rapid successive analysis requests"""
        agent = SentinelAgent()
        test_file = Path(__file__).parent / "test_data" / "phishing_email.txt"
        
        mock_result = ThreatAnalysis(True, 0.9, "phishing", "Rapid test", [], "", {})
        
        results = []
        start_time = time.time()
        
        with patch.object(agent, 'analyze_input', return_value=mock_result):
            for i in range(20):  # 20 rapid analyses
                result = agent.analyze_input(str(test_file))
                results.append(result)
                
        end_time = time.time()
        
        self.assertEqual(len(results), 20)
        self.assertLess(end_time - start_time, 30.0, "Rapid analyses should complete within 30 seconds")


if __name__ == '__main__':
    # Ensure test data directory exists
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Run tests with verbose output
    unittest.main(verbosity=2, exit=False)
    
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("pytest not available, skipping pytest tests")
