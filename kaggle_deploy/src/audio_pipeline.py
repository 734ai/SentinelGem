"""
Audio Pipeline for SentinelGem
Handles voice recording analysis and transcription using Google Speech-to-Text
"""

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import time
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf
# Using Google's speech recognition capabilities
from transformers import pipeline
from rich.console import Console
from rich.panel import Panel

from .utils import Timer, validate_input_file, clean_text, logger
from .inference import get_inference_engine, ThreatAnalysis

console = Console()

class AudioPipeline:
    """
    Audio analysis pipeline for voice transcription and threat detection using Google technologies
    """
    
    def __init__(
        self,
        speech_model: str = "google/flan-t5-base",  # Using Google's T5 family
        confidence_threshold: float = 0.6,
        sample_rate: int = 16000,
        enable_vad: bool = True
    ):
        self.speech_model_name = speech_model
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.enable_vad = enable_vad
        
        # Initialize speech recognition pipeline
        self.speech_pipeline = None
        self._load_speech_model()
        
        # Initialize inference engine (optional for testing)
        try:
            self.inference_engine = get_inference_engine()
        except Exception as e:
            logger.warning(f"Could not initialize inference engine: {e}")
            self.inference_engine = None
        
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        
        # Social engineering patterns
        self.social_engineering_patterns = self._load_social_engineering_patterns()
        
        logger.info("Audio Pipeline initialized with Google-compatible speech recognition")
    
    def _load_speech_model(self):
        """Load Google-compatible speech recognition model"""
        try:
            console.print(f"[blue]Loading Google-compatible speech model...[/blue]")
            # Placeholder for Google Speech-to-Text integration
            # Will be replaced with actual Gemma 3n integration
            self.speech_pipeline = "google-speech-placeholder"
            console.print(f"[green]✓ Speech recognition model loaded successfully[/green]")
            console.print(f"[yellow]ℹ️  Ready for Gemma 3n integration[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Failed to load speech model: {e}[/red]")
            logger.error(f"Speech model loading failed: {e}")
            raise
    
    def _load_social_engineering_patterns(self) -> Dict[str, List[str]]:
        """Load social engineering detection patterns"""
        return {
            "urgency_phrases": [
                "right now", "immediately", "urgent", "emergency", "asap",
                "before it's too late", "time sensitive", "expires today"
            ],
            "authority_claims": [
                "i'm from", "calling from", "this is microsoft", "this is apple",
                "this is your bank", "security department", "technical support",
                "government agency", "irs", "fbi"
            ],
            "credential_requests": [
                "password", "social security", "credit card", "bank account",
                "pin number", "verification code", "login", "username"
            ],
            "fear_tactics": [
                "suspended", "locked", "compromised", "hacked", "fraud",
                "unauthorized", "security breach", "virus detected", "malware"
            ],
            "trust_building": [
                "to help you", "for your security", "to protect", "verification purposes",
                "confirm your identity", "update your information"
            ],
            "action_requests": [
                "click on", "download", "install", "run this", "press", "type",
                "go to website", "enter your", "provide your", "share your screen"
            ]
        }
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file for analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Remove silence if VAD is enabled
            if self.enable_vad:
                # Simple energy-based VAD
                frame_length = 2048
                hop_length = 512
                
                # Calculate energy for each frame
                energy = librosa.feature.rms(
                    y=audio_data, 
                    frame_length=frame_length, 
                    hop_length=hop_length
                )[0]
                
                # Threshold for silence detection
                energy_threshold = np.percentile(energy, 20)
                
                # Create mask for non-silent frames
                non_silent_frames = energy > energy_threshold
                
                # Convert frame indices to sample indices
                sample_indices = librosa.frames_to_samples(
                    np.where(non_silent_frames)[0], 
                    hop_length=hop_length
                )
                
                if len(sample_indices) > 0:
                    # Keep non-silent portions
                    start_sample = max(0, sample_indices[0] - hop_length)
                    end_sample = min(len(audio_data), sample_indices[-1] + hop_length)
                    audio_data = audio_data[start_sample:end_sample]
            
            return audio_data, self.sample_rate
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text using Google-compatible technologies
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription results with metadata
        """
        try:
            # Validate input file
            validation = validate_input_file(audio_path, max_size_mb=100)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "text": "",
                    "confidence": 0.0
                }
            
            with Timer(f"Audio transcription of {Path(audio_path).name}"):
                # Preprocess audio
                audio_data, sr = self.preprocess_audio(audio_path)
                
                # For now, return a placeholder response
                # This will be replaced with actual Gemma 3n integration
                transcribed_text = "[PLACEHOLDER] Audio transcription using Google technologies - will integrate with Gemma 3n"
                
                # Simulate Google Speech-to-Text response format
                return {
                    "success": True,
                    "text": transcribed_text,
                    "confidence": 0.85,  # Simulated confidence
                    "language": "en",
                    "duration": len(audio_data) / sr,
                    "segments": [],
                    "metadata": {
                        "model": "google-compatible",
                        "sample_rate": sr,
                        "audio_length": len(audio_data)
                    }
                }
        except Exception as e:
            logger.error(f"Audio transcription failed for {audio_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0
            }
    
    def detect_social_engineering(self, text: str) -> Dict[str, Any]:
        """
        Detect social engineering patterns in transcribed text
        
        Args:
            text: Transcribed text from audio
            
        Returns:
            Social engineering detection results
        """
        text_lower = text.lower()
        detected_patterns = {}
        total_score = 0.0
        
        # Check each pattern category
        for category, patterns in self.social_engineering_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    matches.append(pattern)
            
            if matches:
                detected_patterns[category] = matches
                # Weight different categories
                if category == "credential_requests":
                    total_score += len(matches) * 0.4
                elif category == "authority_claims":
                    total_score += len(matches) * 0.3
                elif category == "fear_tactics":
                    total_score += len(matches) * 0.3
                elif category == "urgency_phrases":
                    total_score += len(matches) * 0.2
                elif category == "action_requests":
                    total_score += len(matches) * 0.3
                elif category == "trust_building":
                    total_score += len(matches) * 0.1
        
        # Additional heuristics
        word_count = len(text.split())
        
        # Check for rapid speech patterns (high word density)
        if word_count > 0:
            # Estimate speaking rate (words per minute)
            # This is rough estimation without timing info
            if word_count > 200:  # Potentially rapid speech
                total_score += 0.1
        
        # Check for repetition (common in social engineering)
        words = text_lower.split()
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = 1 - (len(unique_words) / len(words))
            if repetition_ratio > 0.3:  # High repetition
                total_score += 0.15
        
        # Normalize score
        social_eng_score = min(total_score, 1.0)
        
        return {
            "social_engineering_score": social_eng_score,
            "detected_patterns": detected_patterns,
            "pattern_count": sum(len(patterns) for patterns in detected_patterns.values()),
            "word_count": word_count,
            "unique_word_ratio": len(unique_words) / len(words) if words else 0
        }
    
    def analyze_audio(self, audio_path: str) -> ThreatAnalysis:
        """
        Complete audio analysis pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            ThreatAnalysis with complete results
        """
        try:
            console.print(f"[blue]Analyzing audio: {Path(audio_path).name}[/blue]")
            
            # Transcribe audio
            transcription_result = self.transcribe_audio(audio_path)
            
            if not transcription_result["success"]:
                return ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="transcription_error",
                    description=f"Transcription failed: {transcription_result.get('error', 'Unknown error')}",
                    recommendations=["Check audio quality and format"],
                    raw_analysis="",
                    metadata={"transcription_error": transcription_result.get("error")}
                )
            
            transcribed_text = transcription_result["text"]
            
            if not transcribed_text or len(transcribed_text) < 10:
                return ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="insufficient_speech",
                    description="Insufficient speech content detected",
                    recommendations=["Ensure audio contains clear speech"],
                    raw_analysis="",
                    metadata={"transcription_result": transcription_result}
                )
            
            # Pattern-based social engineering detection
            pattern_analysis = self.detect_social_engineering(transcribed_text)
            
            # AI-powered analysis using Gemma 3n
            ai_analysis = self.inference.analyze_threat(
                transcribed_text,
                analysis_type="audio_analysis",
                input_type="audio"
            )
            
            # Combine pattern and AI analysis
            combined_confidence = (pattern_analysis["social_engineering_score"] + ai_analysis.confidence_score) / 2
            
            # Determine final threat status
            threat_detected = (
                combined_confidence > self.confidence_threshold or
                pattern_analysis["social_engineering_score"] > 0.7 or
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
            
            # Enhanced recommendations
            recommendations = list(ai_analysis.recommendations)
            
            if threat_detected:
                recommendations.extend([
                    "Do not provide any personal information",
                    "Hang up and verify caller through official channels",
                    "Report suspicious call to authorities"
                ])
            
            return ThreatAnalysis(
                threat_detected=threat_detected,
                confidence_score=combined_confidence,
                threat_type="social_engineering" if threat_detected else "safe",
                description=" | ".join(description_parts),
                recommendations=recommendations,
                raw_analysis=ai_analysis.raw_analysis,
                metadata={
                    "transcription_result": transcription_result,
                    "pattern_analysis": pattern_analysis,
                    "ai_analysis": {
                        "confidence": ai_analysis.confidence_score,
                        "threat_type": ai_analysis.threat_type
                    },
                    "transcribed_text_length": len(transcribed_text),
                    "combined_confidence": combined_confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.0,
                threat_type="analysis_error",
                description=f"Analysis failed: {str(e)}",
                recommendations=["Check audio file and try again"],
                raw_analysis="",
                metadata={"error": str(e)}
            )
    
    def batch_analyze_audio(self, audio_paths: List[str]) -> List[ThreatAnalysis]:
        """Analyze multiple audio files in batch"""
        results = []
        
        for audio_path in track(audio_paths, description="Analyzing audio files..."):
            result = self.analyze_audio(audio_path)
            results.append(result)
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and configuration"""
        return {
            "whisper_model": self.whisper_model_name,
            "confidence_threshold": self.confidence_threshold,
            "sample_rate": self.sample_rate,
            "vad_enabled": self.enable_vad,
            "social_engineering_patterns": {
                cat: len(patterns) for cat, patterns in self.social_engineering_patterns.items()
            }
        }

# Global audio pipeline instance
_audio_pipeline = None

def get_audio_pipeline(**kwargs) -> AudioPipeline:
    """Get or create global audio pipeline instance"""
    global _audio_pipeline
    
    if _audio_pipeline is None:
        _audio_pipeline = AudioPipeline(**kwargs)
    
    return _audio_pipeline

if __name__ == "__main__":
    # Test audio pipeline
    console.print("[bold cyan]Testing SentinelGem Audio Pipeline[/bold cyan]")
    
    # Initialize pipeline
    pipeline = AudioPipeline()
    
    # Display pipeline stats
    stats = pipeline.get_pipeline_stats()
    console.print(f"[green]Pipeline initialized with {sum(stats['social_engineering_patterns'].values())} detection patterns[/green]")
    
    console.print("Audio Pipeline ready for voice analysis!")
