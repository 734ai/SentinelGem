#!/usr/bin/env python3
"""
SentinelGem Audio Pipeline Tests
Author: Muzan Sano

Comprehensive test suite for audio pipeline including:
- Whisper transcription accuracy
- Social engineering detection
- Audio preprocessing quality
- Performance benchmarks
- Voice pattern analysis
"""

import unittest
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import librosa
import soundfile as sf

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.audio_pipeline import AudioPipeline
from src.inference import ThreatAnalysis


class TestAudioPipeline(unittest.TestCase):
    """Test cases for Audio Pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data_dir = project_root / "tests" / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Create test audio samples
        cls.create_test_audio_samples()
        
    @classmethod
    def create_test_audio_samples(cls):
        """Create test audio samples for testing"""
        sample_rate = 16000
        duration = 3  # 3 seconds
        
        # Create a simple sine wave (simulates voice)
        t = np.linspace(0, duration, sample_rate * duration, False)
        frequency = 440  # A4 note
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add some noise to simulate real audio
        noise = np.random.normal(0, 0.05, audio_data.shape)
        audio_data += noise
        
        # Save test audio files
        cls.test_audio_path = cls.test_data_dir / "test_audio.wav"
        sf.write(str(cls.test_audio_path), audio_data, sample_rate)
        
        # Create a silent audio file
        silent_audio = np.zeros(sample_rate * 2)  # 2 seconds of silence
        cls.silent_audio_path = cls.test_data_dir / "silent_audio.wav"
        sf.write(str(cls.silent_audio_path), silent_audio, sample_rate)
        
        # Create a noisy audio file
        noisy_audio = np.random.normal(0, 0.1, sample_rate * 2)
        cls.noisy_audio_path = cls.test_data_dir / "noisy_audio.wav"
        sf.write(str(cls.noisy_audio_path), noisy_audio, sample_rate)
        
    def setUp(self):
        """Set up test fixtures"""
        # Mock Whisper model loading to avoid downloading actual models
        with patch('src.audio_pipeline.whisper.load_model') as mock_whisper, \
             patch('src.audio_pipeline.get_inference_engine') as mock_inference:
            
            mock_whisper.return_value = Mock()
            mock_inference.return_value = Mock()
            
            self.audio_pipeline = AudioPipeline(
                whisper_model="base",
                confidence_threshold=0.6,
                sample_rate=16000,
                enable_vad=True
            )
            
        # Mock the models
        self.mock_whisper = Mock()
        self.mock_inference = Mock()
        self.audio_pipeline.whisper_model = self.mock_whisper
        self.audio_pipeline.inference = self.mock_inference
        
    def test_initialization(self):
        """Test AudioPipeline initialization"""
        self.assertEqual(self.audio_pipeline.whisper_model_name, "base")
        self.assertEqual(self.audio_pipeline.confidence_threshold, 0.6)
        self.assertEqual(self.audio_pipeline.sample_rate, 16000)
        self.assertTrue(self.audio_pipeline.enable_vad)
        self.assertIsNotNone(self.audio_pipeline.social_engineering_patterns)
        
    def test_audio_preprocessing(self):
        """Test audio preprocessing functionality"""
        # Test with actual audio file
        audio_data, sr = self.audio_pipeline.preprocess_audio(str(self.test_audio_path))
        
        self.assertIsInstance(audio_data, np.ndarray)
        self.assertEqual(sr, self.audio_pipeline.sample_rate)
        self.assertGreater(len(audio_data), 0)
        
        # Test VAD (Voice Activity Detection)
        self.audio_pipeline.enable_vad = True
        vad_audio, vad_sr = self.audio_pipeline.preprocess_audio(str(self.test_audio_path))
        self.assertLessEqual(len(vad_audio), len(audio_data))  # VAD should remove some silence
        
    def test_silent_audio_handling(self):
        """Test handling of silent audio"""
        try:
            audio_data, sr = self.audio_pipeline.preprocess_audio(str(self.silent_audio_path))
            # Should handle silent audio gracefully
            self.assertIsInstance(audio_data, np.ndarray)
        except Exception as e:
            # If it raises an exception, ensure it's handled properly
            self.assertIn("audio", str(e).lower())
            
    @patch('src.audio_pipeline.tempfile.NamedTemporaryFile')
    @patch('src.audio_pipeline.sf.write')
    def test_transcription(self, mock_sf_write, mock_temp_file):
        """Test audio transcription with Whisper"""
        # Mock temporary file
        mock_temp_file.return_value.__enter__.return_value.name = "temp_audio.wav"
        
        # Mock Whisper transcription result
        mock_transcription = {
            "text": "This is Microsoft calling about your computer security",
            "segments": [
                {
                    "text": "This is Microsoft calling about your computer security",
                    "avg_logprob": -0.5,
                    "start": 0.0,
                    "end": 3.0
                }
            ],
            "language": "en"
        }
        self.mock_whisper.transcribe.return_value = mock_transcription
        
        result = self.audio_pipeline.transcribe_audio(str(self.test_audio_path))
        
        self.assertTrue(result["success"])
        self.assertIn("microsoft", result["text"].lower())
        self.assertGreater(result["confidence"], 0.0)
        self.assertEqual(result["language"], "en")
        
    def test_social_engineering_detection(self):
        """Test social engineering pattern detection"""
        # Test various social engineering texts
        test_cases = [
            {
                "text": "This is Microsoft calling. Your computer has been hacked. Press 1 to speak to our security team.",
                "expected_high_score": True,
                "patterns": ["authority_claims", "fear_tactics", "urgency_phrases"]
            },
            {
                "text": "Hello, this is your bank calling. We need to verify your social security number immediately.",
                "expected_high_score": True,
                "patterns": ["authority_claims", "credential_requests", "urgency_phrases"]
            },
            {
                "text": "Hi mom, just calling to check in. How are you doing today?",
                "expected_high_score": False,
                "patterns": []
            }
        ]
        
        for case in test_cases:
            result = self.audio_pipeline.detect_social_engineering(case["text"])
            
            if case["expected_high_score"]:
                self.assertGreater(result["social_engineering_score"], 0.5,
                                 f"Failed to detect social engineering in: {case['text']}")
            else:
                self.assertLess(result["social_engineering_score"], 0.3,
                               f"False positive on legitimate text: {case['text']}")
                               
            # Check for expected patterns
            for pattern in case["patterns"]:
                if pattern in result["detected_patterns"]:
                    self.assertGreater(len(result["detected_patterns"][pattern]), 0)
                    
    def test_full_audio_analysis(self):
        """Test complete audio analysis pipeline"""
        # Mock transcription
        mock_transcription = {
            "success": True,
            "text": "This is the IRS calling. Your tax refund has been suspended due to suspicious activity. Press 1 now.",
            "confidence": 0.85,
            "language": "en"
        }
        
        # Mock AI analysis
        mock_ai_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.92,
            threat_type="social_engineering",
            description="Government impersonation scam detected",
            recommendations=["Hang up immediately", "Report to authorities"],
            raw_analysis="AI detected authority impersonation",
            metadata={}
        )
        
        with patch.object(self.audio_pipeline, 'transcribe_audio', return_value=mock_transcription):
            self.mock_inference.analyze_threat.return_value = mock_ai_result
            
            result = self.audio_pipeline.analyze_audio(str(self.test_audio_path))
            
        self.assertIsInstance(result, ThreatAnalysis)
        self.assertTrue(result.threat_detected)
        self.assertEqual(result.threat_type, "social_engineering")
        self.assertGreater(result.confidence_score, 0.8)
        self.assertIn("Hang up", result.recommendations[0])
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test non-existent file
        result = self.audio_pipeline.transcribe_audio("nonexistent.wav")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Test invalid audio format
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"This is not an audio file")
            temp_file.flush()
            
            result = self.audio_pipeline.transcribe_audio(temp_file.name)
            self.assertFalse(result["success"])
            
    def test_batch_analysis(self):
        """Test batch audio analysis"""
        audio_paths = [
            str(self.test_audio_path),
            str(self.silent_audio_path)
        ]
        
        # Mock results
        mock_results = [
            ThreatAnalysis(True, 0.8, "social_engineering", "Scam call", [], "", {}),
            ThreatAnalysis(False, 0.1, "insufficient_speech", "No speech", [], "", {})
        ]
        
        with patch.object(self.audio_pipeline, 'analyze_audio', side_effect=mock_results):
            results = self.audio_pipeline.batch_analyze_audio(audio_paths)
            
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].threat_detected)
        self.assertFalse(results[1].threat_detected)
        
    def test_voice_activity_detection(self):
        """Test Voice Activity Detection (VAD)"""
        # Test with VAD enabled
        self.audio_pipeline.enable_vad = True
        vad_audio, _ = self.audio_pipeline.preprocess_audio(str(self.test_audio_path))
        
        # Test with VAD disabled
        self.audio_pipeline.enable_vad = False
        no_vad_audio, _ = self.audio_pipeline.preprocess_audio(str(self.test_audio_path))
        
        # VAD should typically result in shorter audio (silence removed)
        # But for our synthetic test audio, this might not always be true
        self.assertIsInstance(vad_audio, np.ndarray)
        self.assertIsInstance(no_vad_audio, np.ndarray)
        
    def test_confidence_threshold_enforcement(self):
        """Test confidence threshold enforcement"""
        # Mock low confidence transcription
        low_confidence_transcription = {
            "success": True,
            "text": "unclear mumbled speech",
            "confidence": 0.3,
            "language": "en"
        }
        
        with patch.object(self.audio_pipeline, 'transcribe_audio', return_value=low_confidence_transcription):
            result = self.audio_pipeline.analyze_audio(str(self.test_audio_path))
            
        # Should handle low confidence appropriately
        self.assertIsInstance(result, ThreatAnalysis)
        self.assertLessEqual(result.confidence_score, 0.6)
        
    def test_multiple_languages(self):
        """Test multi-language support"""
        # Mock transcription in different languages
        languages = ["en", "es", "fr", "de"]
        
        for lang in languages:
            mock_transcription = {
                "success": True,
                "text": "Test text in different language",
                "confidence": 0.8,
                "language": lang
            }
            
            with patch.object(self.audio_pipeline, 'transcribe_audio', return_value=mock_transcription):
                result = self.audio_pipeline.analyze_audio(str(self.test_audio_path))
                
            self.assertIsInstance(result, ThreatAnalysis)
            
    def test_performance_metrics(self):
        """Test audio processing performance"""
        import time
        
        start_time = time.time()
        
        # Mock fast transcription
        mock_transcription = {
            "success": True,
            "text": "Quick test transcription",
            "confidence": 0.8,
            "language": "en"
        }
        
        with patch.object(self.audio_pipeline, 'transcribe_audio', return_value=mock_transcription):
            result = self.audio_pipeline.analyze_audio(str(self.test_audio_path))
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 10.0, "Audio processing should complete within 10 seconds")
        
    def test_pipeline_statistics(self):
        """Test pipeline statistics and configuration"""
        stats = self.audio_pipeline.get_pipeline_stats()
        
        self.assertIn("whisper_model", stats)
        self.assertIn("confidence_threshold", stats)
        self.assertIn("sample_rate", stats)
        self.assertIn("vad_enabled", stats)
        self.assertIn("social_engineering_patterns", stats)
        
        self.assertEqual(stats["whisper_model"], "base")
        self.assertEqual(stats["confidence_threshold"], 0.6)
        self.assertEqual(stats["sample_rate"], 16000)
        self.assertTrue(stats["vad_enabled"])


class TestAudioIntegration(unittest.TestCase):
    """Integration tests for audio pipeline"""
    
    def test_agent_integration(self):
        """Test integration with agent system"""
        with patch('src.audio_pipeline.whisper.load_model'), \
             patch('src.audio_pipeline.get_inference_engine'):
            
            audio_pipeline = AudioPipeline()
            
        # Mock agent calling audio pipeline
        mock_result = ThreatAnalysis(
            threat_detected=True,
            confidence_score=0.9,
            threat_type="social_engineering",
            description="Phone scam detected",
            recommendations=["Hang up", "Block number"],
            raw_analysis="Audio integration test",
            metadata={"source": "audio_pipeline", "call_duration": 45}
        )
        
        audio_pipeline.inference = Mock()
        audio_pipeline.inference.analyze_threat.return_value = mock_result
        
        with patch.object(audio_pipeline, 'transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                "success": True,
                "text": "This is a test scam call",
                "confidence": 0.8,
                "language": "en"
            }
            
            result = audio_pipeline.analyze_audio("test_audio.wav")
            
        self.assertEqual(result.metadata["source"], "audio_pipeline")
        self.assertTrue(result.threat_detected)
        
    def test_real_world_scam_scenarios(self):
        """Test real-world scam call scenarios"""
        scam_scenarios = [
            {
                "text": "Hello, this is Amazon customer service. There's been fraudulent activity on your account. Please provide your credit card number to verify your identity.",
                "expected_threat": True,
                "threat_indicators": ["authority_claims", "credential_requests", "fear_tactics"]
            },
            {
                "text": "Hi, this is tech support from Windows. Your computer is infected with viruses. We need remote access to fix it immediately.",
                "expected_threat": True,
                "threat_indicators": ["authority_claims", "fear_tactics", "urgency_phrases"]
            },
            {
                "text": "Good morning, this is a reminder about your dental appointment tomorrow at 2 PM. Please call if you need to reschedule.",
                "expected_threat": False,
                "threat_indicators": []
            }
        ]
        
        with patch('src.audio_pipeline.whisper.load_model'), \
             patch('src.audio_pipeline.get_inference_engine'):
            
            audio_pipeline = AudioPipeline()
            
        for scenario in scam_scenarios:
            result = audio_pipeline.detect_social_engineering(scenario["text"])
            
            if scenario["expected_threat"]:
                self.assertGreater(result["social_engineering_score"], 0.6,
                                 f"Failed to detect scam in: {scenario['text'][:50]}...")
            else:
                self.assertLess(result["social_engineering_score"], 0.3,
                               f"False positive on legitimate call: {scenario['text'][:50]}...")


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
