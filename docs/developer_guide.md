# SentinelGem Developer Guide
# Author: Muzan Sano
# Version: 1.0

## Welcome to SentinelGem Development

This comprehensive developer guide provides everything you need to understand, contribute to, and extend the SentinelGem threat analysis platform. Whether you're fixing bugs, adding features, or integrating SentinelGem into your own systems, this guide will help you navigate the codebase effectively.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [AI Model Integration](#ai-model-integration)
5. [Testing Framework](#testing-framework)
6. [API Development](#api-development)
7. [Contributing Guidelines](#contributing-guidelines)
8. [Advanced Topics](#advanced-topics)

---

## Development Environment Setup

### Prerequisites

- **Python 3.11+**: Modern Python with latest features
- **Git**: Version control and collaboration
- **Docker**: Containerization and deployment
- **CUDA Toolkit**: GPU acceleration (optional)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/734ai/SentinelGem.git
cd SentinelGem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/ -v
```

### Development Dependencies

```txt
# requirements-dev.txt
# Core AI/ML
torch>=2.0.0
transformers>=4.35.0
whisper-openai>=20231117
pytesseract>=0.3.10

# Development Tools
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.1
black>=23.7.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.5.0

# Documentation
sphinx>=7.1.0
myst-parser>=2.0.0
sphinx-rtd-theme>=1.3.0

# Testing & Mocking
responses>=0.23.0
httpx>=0.24.0
factory-boy>=3.3.0

# Performance & Monitoring
memory-profiler>=0.61.0
line-profiler>=4.0.0
py-spy>=0.3.14
```

### IDE Configuration

#### VS Code Setup
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true
    }
}
```

#### PyCharm Configuration
- **Interpreter**: Point to `venv/bin/python`
- **Code Style**: Import Black and isort configurations
- **Test Runner**: Configure pytest as default
- **Type Checking**: Enable mypy integration

---

## Architecture Overview

### System Design Principles

SentinelGem follows modern software architecture principles:

```python
# Core architectural patterns used:

# 1. Dependency Injection
class ThreatAnalyzer:
    def __init__(self, model_hub: ModelHub, cache: Cache):
        self.model_hub = model_hub
        self.cache = cache

# 2. Factory Pattern
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict):
        if model_type == "gemma":
            return GemmaModel(config)
        elif model_type == "whisper":
            return WhisperModel(config)

# 3. Observer Pattern
class ThreatDetector:
    def __init__(self):
        self.observers = []
    
    def notify_observers(self, threat):
        for observer in self.observers:
            observer.on_threat_detected(threat)

# 4. Strategy Pattern
class AnalysisStrategy:
    def analyze(self, input_data): pass

class TextAnalysisStrategy(AnalysisStrategy):
    def analyze(self, text): pass

class AudioAnalysisStrategy(AnalysisStrategy):
    def analyze(self, audio): pass
```

### Directory Structure

```
SentinelGem/
├── src/                        # Core application code
│   ├── __init__.py
│   ├── inference.py           # AI model inference
│   ├── ocr_pipeline.py        # OCR processing
│   ├── audio_pipeline.py      # Audio processing
│   ├── threat_rules.py        # Rule-based detection
│   ├── autogen_notebook.py    # Report generation
│   └── utils/                 # Utility functions
├── agents/                     # Agent orchestration
│   ├── agent_loop.py          # Main agent logic
│   ├── orchestrator.py        # Task coordination
│   ├── prompts/               # LLM prompts
│   └── reward_model.py        # Output evaluation
├── api/                       # REST API
│   ├── main.py               # FastAPI application
│   ├── routes/               # API endpoints
│   ├── middleware/           # Request processing
│   └── schemas/              # Pydantic models
├── ui/                        # User interface
│   ├── streamlit_app.py      # Web interface
│   ├── components/           # UI components
│   └── static/               # Static assets
├── tests/                     # Test suite
│   ├── test_inference.py     # Unit tests
│   ├── integration_test.py   # Integration tests
│   ├── conftest.py          # Test configuration
│   └── fixtures/            # Test data
├── docs/                     # Documentation
├── config/                   # Configuration files
├── scripts/                  # Utility scripts
└── notebooks/               # Jupyter notebooks
```

---

## Core Components

### 1. Inference Engine (`src/inference.py`)

The inference engine is the heart of SentinelGem's AI capabilities:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class ThreatAnalysis:
    """Standardized threat analysis result"""
    threat_detected: bool
    confidence_score: float
    threat_type: str
    description: str
    recommendations: List[str]
    raw_analysis: str
    metadata: Dict[str, Any]

class GemmaInference:
    """Gemma 3n model inference wrapper"""
    
    def __init__(self, model_name: str = "google/gemma-3n-2b", device: str = "auto"):
        self.device = self._setup_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=self.device
        )
        
    def analyze_text(self, text: str, context: str = "") -> ThreatAnalysis:
        """Analyze text for threats using Gemma model"""
        prompt = self._build_analysis_prompt(text, context)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse structured response
        return self._parse_analysis_response(response, text)
    
    def _build_analysis_prompt(self, text: str, context: str) -> str:
        """Build structured prompt for threat analysis"""
        return f"""
        Analyze the following content for cybersecurity threats:
        
        Context: {context}
        Content: {text}
        
        Provide analysis in this format:
        - Threat Detected: Yes/No
        - Confidence: 0.0-1.0
        - Threat Type: phishing/malware/social_engineering/safe
        - Description: Brief explanation
        - Recommendations: Bullet-pointed actions
        """
    
    def _parse_analysis_response(self, response: str, original_text: str) -> ThreatAnalysis:
        """Parse AI response into structured format"""
        # Implementation for parsing AI response
        # This would include regex/NLP parsing of the structured output
        pass
```

### 2. OCR Pipeline (`src/ocr_pipeline.py`)

Visual content analysis for screenshot-based threats:

```python
import cv2
import pytesseract
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

class OCRPipeline:
    """OCR-based visual threat detection"""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Phishing indicators to look for
        self.phishing_patterns = [
            r'verify.{0,20}account',
            r'click.{0,20}here.{0,20}immediately',
            r'suspended.{0,20}account',
            r'confirm.{0,20}identity',
            r'update.{0,20}payment'
        ]
    
    def process_image(self, image_path: str) -> ThreatAnalysis:
        """Process image and detect visual threats"""
        # Load and preprocess image
        image = self._load_and_preprocess(image_path)
        
        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(image)
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Analyze for threats
        threat_analysis = self._analyze_visual_content(extracted_text, ocr_data, image)
        
        return threat_analysis
    
    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for better OCR"""
        # Load image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        # 1. Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. Contrast enhancement
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
        
        # 3. Thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _analyze_visual_content(self, text: str, ocr_data: dict, image: np.ndarray) -> ThreatAnalysis:
        """Analyze extracted content for threats"""
        threats_found = []
        confidence_scores = []
        
        # Check for phishing patterns
        for pattern in self.phishing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                threats_found.extend(matches)
                confidence_scores.append(0.8)
        
        # Visual analysis (colors, layout, etc.)
        visual_threats = self._analyze_visual_elements(image, ocr_data)
        threats_found.extend(visual_threats)
        
        # Calculate overall confidence
        if threats_found:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            return ThreatAnalysis(
                threat_detected=True,
                confidence_score=avg_confidence,
                threat_type="phishing",
                description=f"Visual phishing indicators detected: {', '.join(threats_found)}",
                recommendations=[
                    "Do not interact with this content",
                    "Report to security team",
                    "Verify legitimacy through official channels"
                ],
                raw_analysis=text,
                metadata={
                    "patterns_found": threats_found,
                    "ocr_confidence": self._calculate_ocr_confidence(ocr_data)
                }
            )
        else:
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.1,
                threat_type="safe",
                description="No visual threats detected",
                recommendations=["Content appears safe"],
                raw_analysis=text,
                metadata={"ocr_confidence": self._calculate_ocr_confidence(ocr_data)}
            )
```

### 3. Audio Pipeline (`src/audio_pipeline.py`)

Audio analysis for social engineering detection:

```python
import whisper
import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple
import speech_recognition as sr

class AudioPipeline:
    """Audio analysis for social engineering detection"""
    
    def __init__(self, whisper_model: str = "base"):
        self.whisper_model = whisper.load_model(whisper_model)
        self.recognizer = sr.Recognizer()
        
        # Social engineering indicators
        self.social_engineering_patterns = [
            r'verify.{0,20}social.{0,20}security',
            r'IRS.{0,20}owe.{0,20}money',
            r'urgent.{0,20}action.{0,20}required',
            r'computer.{0,20}virus.{0,20}detected',
            r'bank.{0,20}account.{0,20}frozen',
            r'winning.{0,20}lottery.{0,20}prize'
        ]
    
    def process_audio(self, audio_path: str) -> ThreatAnalysis:
        """Process audio file for social engineering threats"""
        # Load audio
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Voice Activity Detection
        speech_segments = self._detect_speech_segments(audio_data, sample_rate)
        
        # Transcribe using Whisper
        transcript = self._transcribe_audio(audio_path)
        
        # Audio feature analysis
        audio_features = self._extract_audio_features(audio_data, sample_rate)
        
        # Analyze transcript for threats
        threat_analysis = self._analyze_transcript_threats(transcript, audio_features)
        
        return threat_analysis
    
    def _transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper"""
        result = self.whisper_model.transcribe(audio_path)
        return result
    
    def _detect_speech_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Detect speech segments in audio"""
        # Simple VAD using energy-based detection
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.01 * sample_rate)     # 10ms hop
        
        # Calculate frame energy
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # Threshold-based VAD
        threshold = np.mean(energy) * 0.1
        speech_frames = energy > threshold
        
        # Convert frame indices to time segments
        segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            time = i * hop_length / sample_rate
            
            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((start_time, time))
                in_speech = False
        
        return segments
    
    def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Extract audio features for analysis"""
        features = {}
        
        # Basic features
        features['duration'] = len(audio_data) / sample_rate
        features['rms_energy'] = np.sqrt(np.mean(audio_data ** 2))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        
        # Zero crossing rate (indicator of speech vs. noise)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        
        return features
    
    def _analyze_transcript_threats(self, transcript: Dict, audio_features: Dict) -> ThreatAnalysis:
        """Analyze transcript for social engineering threats"""
        text = transcript.get('text', '')
        
        threats_found = []
        confidence_scores = []
        
        # Pattern matching
        for pattern in self.social_engineering_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                threats_found.extend(matches)
                confidence_scores.append(0.85)
        
        # Audio-based indicators
        if audio_features.get('rms_energy', 0) > 0.05:  # High energy could indicate urgency
            confidence_scores.append(0.3)
        
        if threats_found:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            return ThreatAnalysis(
                threat_detected=True,
                confidence_score=avg_confidence,
                threat_type="social_engineering",
                description=f"Social engineering indicators detected in audio: {', '.join(threats_found)}",
                recommendations=[
                    "Do not provide any personal information",
                    "Hang up and verify caller through official channels",
                    "Report to authorities if financial threat"
                ],
                raw_analysis=text,
                metadata={
                    "transcript": transcript,
                    "audio_features": audio_features,
                    "patterns_found": threats_found
                }
            )
        else:
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.15,
                threat_type="safe",
                description="No social engineering indicators detected",
                recommendations=["Audio content appears safe"],
                raw_analysis=text,
                metadata={
                    "transcript": transcript,
                    "audio_features": audio_features
                }
            )
```

---

## AI Model Integration

### Model Hub Architecture

SentinelGem uses a centralized model hub for efficient model management:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import whisper

class BaseModel(ABC):
    """Base class for all AI models"""
    
    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def unload(self) -> None:
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
    
    @abstractmethod
    def analyze(self, input_data: Any) -> Dict[str, Any]:
        pass

class ModelHub:
    """Central hub for model management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.model_cache_size = config.get('model_cache_size', 2)
        self.loaded_models: List[str] = []
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get model instance, loading if necessary"""
        if model_name not in self.models:
            self.models[model_name] = self._create_model(model_name)
        
        model = self.models[model_name]
        
        if not model.is_loaded():
            self._load_model_with_cache_management(model_name)
        
        return model
    
    def _create_model(self, model_name: str) -> BaseModel:
        """Factory method for creating models"""
        if model_name.startswith('gemma'):
            return GemmaModel(model_name, self.config.get('gemma', {}))
        elif model_name.startswith('whisper'):
            return WhisperModel(model_name, self.config.get('whisper', {}))
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _load_model_with_cache_management(self, model_name: str):
        """Load model with intelligent cache management"""
        # If cache is full, unload least recently used model
        if len(self.loaded_models) >= self.model_cache_size:
            lru_model = self.loaded_models.pop(0)
            self.models[lru_model].unload()
        
        # Load requested model
        self.models[model_name].load()
        self.loaded_models.append(model_name)

class GemmaModel(BaseModel):
    """Gemma model implementation"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = self._setup_device()
    
    def load(self) -> None:
        """Load Gemma model"""
        print(f"Loading Gemma model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        # Apply quantization if configured
        if self.config.get('quantization', False):
            self._apply_quantization()
    
    def unload(self) -> None:
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
    
    def analyze(self, input_data: str, context: str = "") -> Dict[str, Any]:
        """Analyze input using Gemma model"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        prompt = self._build_prompt(input_data, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_tokens', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'response': response,
            'prompt': prompt,
            'model_name': self.model_name
        }
```

### Custom Fine-tuning Support

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import json

class ModelFineTuner:
    """Fine-tuning support for custom threat detection"""
    
    def __init__(self, base_model: str, output_dir: str):
        self.base_model = base_model
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
    
    def prepare_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for fine-tuning"""
        def tokenize_function(examples):
            inputs = [f"Analyze: {ex['input']}" for ex in examples['data']]
            targets = [ex['expected_output'] for ex in examples['data']]
            
            model_inputs = self.tokenizer(inputs, truncation=True, padding=True)
            labels = self.tokenizer(targets, truncation=True, padding=True)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = Dataset.from_dict({"data": training_data})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def fine_tune(self, training_data: List[Dict[str, str]], 
                  validation_data: List[Dict[str, str]] = None,
                  epochs: int = 3,
                  learning_rate: float = 5e-5):
        """Fine-tune model on custom data"""
        
        train_dataset = self.prepare_dataset(training_data)
        eval_dataset = self.prepare_dataset(validation_data) if validation_data else None
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            learning_rate=learning_rate
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        trainer.save_model()
        
        return f"{self.output_dir}/final_model"
```

---

## Testing Framework

### Test Architecture

SentinelGem uses a comprehensive testing framework with multiple levels:

```python
# conftest.py - Shared test configuration
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

@pytest.fixture
def mock_gemma_model():
    """Mock Gemma model for testing"""
    with patch('src.inference.AutoModelForCausalLM') as mock_model:
        mock_instance = Mock()
        mock_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing"""
    with patch('whisper.load_model') as mock_whisper:
        mock_instance = Mock()
        mock_instance.transcribe.return_value = {
            'text': 'This is a test transcription',
            'segments': []
        }
        mock_whisper.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def test_data_dir():
    """Temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test files
        (test_dir / "test_email.txt").write_text("Test phishing email content")
        (test_dir / "test_log.txt").write_text("System log with suspicious activity")
        
        yield test_dir

@pytest.fixture
def sample_threat_analysis():
    """Sample threat analysis for testing"""
    return ThreatAnalysis(
        threat_detected=True,
        confidence_score=0.85,
        threat_type="phishing",
        description="Test threat detected",
        recommendations=["Test recommendation"],
        raw_analysis="Test analysis",
        metadata={"test": "data"}
    )
```

### Performance Testing

```python
import time
import psutil
import pytest
from memory_profiler import profile

class TestPerformance:
    """Performance testing suite"""
    
    def test_analysis_speed_requirements(self):
        """Test that analysis meets speed requirements"""
        analyzer = ThreatAnalyzer()
        test_input = "Suspicious email content for testing"
        
        start_time = time.time()
        result = analyzer.analyze_text(test_input)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete within 2 seconds
        assert analysis_time < 2.0, f"Analysis took {analysis_time:.2f}s, exceeds 2s limit"
        assert result is not None
    
    @profile
    def test_memory_usage(self):
        """Test memory usage during analysis"""
        analyzer = ThreatAnalyzer()
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple analyses
        for i in range(10):
            result = analyzer.analyze_text(f"Test input {i}")
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Should not increase by more than 500MB
        assert memory_increase < 500, f"Memory increased by {memory_increase:.2f}MB"
    
    def test_concurrent_analysis(self):
        """Test concurrent analysis performance"""
        import threading
        import queue
        
        analyzer = ThreatAnalyzer()
        results_queue = queue.Queue()
        
        def analyze_worker(input_text):
            result = analyzer.analyze_text(input_text)
            results_queue.put(result)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=analyze_worker, args=(f"Test input {i}",))
            threads.append(thread)
            thread.start()
        
        start_time = time.time()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify results
        assert len(results) == 5
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
```

### Integration Testing

```python
class TestIntegration:
    """Integration testing suite"""
    
    def test_full_email_analysis_workflow(self):
        """Test complete email analysis workflow"""
        # Create test email
        email_content = """
        Subject: Urgent: Verify Your Account
        
        Dear Customer,
        Your account has been suspended. Click here to verify:
        http://suspicious-site.com/verify
        """
        
        # Initialize components
        agent = SentinelAgent()
        
        # Process email
        with patch.object(agent, 'analyze_input') as mock_analyze:
            mock_analyze.return_value = ThreatAnalysis(
                threat_detected=True,
                confidence_score=0.92,
                threat_type="phishing",
                description="Phishing email detected",
                recommendations=["Do not click links"],
                raw_analysis=email_content,
                metadata={"suspicious_urls": ["http://suspicious-site.com/verify"]}
            )
            
            result = agent.analyze_input(email_content)
        
        # Verify result
        assert result.threat_detected
        assert result.confidence_score > 0.9
        assert "phishing" in result.threat_type
        assert len(result.recommendations) > 0
    
    def test_multimodal_analysis_pipeline(self):
        """Test analysis of multiple input types"""
        agent = SentinelAgent()
        
        test_cases = [
            ("text", "Suspicious email content"),
            ("image", "path/to/screenshot.jpg"),
            ("audio", "path/to/voice_call.wav")
        ]
        
        for input_type, content in test_cases:
            with patch.object(agent, f'_analyze_{input_type}') as mock_analyzer:
                mock_analyzer.return_value = ThreatAnalysis(
                    threat_detected=True,
                    confidence_score=0.8,
                    threat_type=f"{input_type}_threat",
                    description=f"{input_type} threat detected",
                    recommendations=[f"Action for {input_type}"],
                    raw_analysis=content,
                    metadata={"input_type": input_type}
                )
                
                result = agent.analyze_input(content, input_type=input_type)
                
                assert result.threat_detected
                assert input_type in result.metadata["input_type"]
```

---

## API Development

### FastAPI Application Structure

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(
    title="SentinelGem API",
    description="AI-powered threat analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class AnalysisRequest(BaseModel):
    content: str
    context: Optional[str] = ""
    analysis_type: str = "auto"

class AnalysisResponse(BaseModel):
    threat_detected: bool
    confidence_score: float
    threat_type: str
    description: str
    recommendations: List[str]
    analysis_id: str
    timestamp: str

class BatchAnalysisRequest(BaseModel):
    items: List[AnalysisRequest]
    parallel: bool = True

# Dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate authentication token"""
    # Implement token validation logic
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return get_user_from_token(credentials.credentials)

# Routes
@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalysisRequest,
    user = Depends(get_current_user)
):
    """Analyze text content for threats"""
    try:
        analyzer = get_analyzer_instance()
        result = await asyncio.to_thread(
            analyzer.analyze_text, 
            request.content, 
            request.context
        )
        
        return AnalysisResponse(
            threat_detected=result.threat_detected,
            confidence_score=result.confidence_score,
            threat_type=result.threat_type,
            description=result.description,
            recommendations=result.recommendations,
            analysis_id=generate_analysis_id(),
            timestamp=get_current_timestamp()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """Analyze uploaded file for threats"""
    try:
        # Validate file type and size
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Process file based on type
        file_content = await file.read()
        
        analyzer = get_analyzer_instance()
        result = await asyncio.to_thread(
            analyzer.analyze_file,
            file_content,
            file.filename,
            file.content_type
        )
        
        return AnalysisResponse(**result.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analyze(
    request: BatchAnalysisRequest,
    user = Depends(get_current_user)
):
    """Batch analysis of multiple items"""
    try:
        analyzer = get_analyzer_instance()
        
        if request.parallel:
            # Parallel processing
            tasks = [
                asyncio.to_thread(analyzer.analyze_text, item.content, item.context)
                for item in request.items
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Sequential processing
            results = []
            for item in request.items:
                result = await asyncio.to_thread(
                    analyzer.analyze_text, 
                    item.content, 
                    item.context
                )
                results.append(result)
        
        return {
            "results": [AnalysisResponse(**result.dict()) for result in results],
            "total_processed": len(results),
            "processing_mode": "parallel" if request.parallel else "sequential"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": get_current_timestamp(),
        "version": "1.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics(user = Depends(get_current_user)):
    """Get system metrics"""
    return {
        "active_analyses": get_active_analysis_count(),
        "total_analyses": get_total_analysis_count(),
        "average_response_time": get_average_response_time(),
        "system_health": get_system_health_status()
    }
```

### WebSocket Support for Real-time Analysis

```python
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """Real-time analysis via WebSocket"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Perform analysis
            analyzer = get_analyzer_instance()
            result = await asyncio.to_thread(
                analyzer.analyze_text,
                request_data.get("content", ""),
                request_data.get("context", "")
            )
            
            # Send result back to client
            response = {
                "type": "analysis_result",
                "data": {
                    "threat_detected": result.threat_detected,
                    "confidence_score": result.confidence_score,
                    "threat_type": result.threat_type,
                    "description": result.description,
                    "recommendations": result.recommendations
                }
            }
            
            await manager.send_personal_message(json.dumps(response), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

---

## Contributing Guidelines

### Code Style and Standards

SentinelGem follows strict coding standards:

```python
# Example of proper code style

from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Use descriptive names
class ThreatAnalysisEngine:
    """
    Comprehensive threat analysis engine.
    
    This class provides the main interface for analyzing various types of
    content for cybersecurity threats.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the threat analysis engine.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_models()
    
    def analyze_content(
        self, 
        content: str, 
        content_type: str = "text",
        context: Optional[str] = None
    ) -> ThreatAnalysis:
        """
        Analyze content for threats.
        
        Args:
            content: Content to analyze
            content_type: Type of content (text, image, audio)
            context: Additional context for analysis
            
        Returns:
            ThreatAnalysis object with results
            
        Raises:
            ValueError: If content_type is not supported
            AnalysisError: If analysis fails
        """
        if not content:
            raise ValueError("Content cannot be empty")
        
        if content_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        try:
            self.logger.info(f"Starting analysis of {content_type} content")
            
            # Route to appropriate analyzer
            analyzer = self._get_analyzer(content_type)
            result = analyzer.analyze(content, context or "")
            
            self.logger.info(
                f"Analysis complete. Threat detected: {result.threat_detected}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"Failed to analyze content: {str(e)}") from e
```

### Testing Requirements

All contributions must include comprehensive tests:

```python
# Example test structure
class TestThreatAnalysisEngine:
    """Test suite for ThreatAnalysisEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        config = {
            "model_type": "gemma-3n-2b",
            "confidence_threshold": 0.7
        }
        return ThreatAnalysisEngine(config)
    
    def test_analyze_text_phishing_detection(self, engine):
        """Test phishing detection in text content"""
        # Arrange
        phishing_content = "Urgent: Your account will be suspended. Click here: http://fake-bank.com"
        
        # Act
        result = engine.analyze_content(phishing_content, "text")
        
        # Assert
        assert result.threat_detected is True
        assert result.threat_type == "phishing"
        assert result.confidence_score > 0.8
        assert len(result.recommendations) > 0
    
    def test_analyze_text_legitimate_content(self, engine):
        """Test legitimate content detection"""
        # Arrange
        legitimate_content = "Thank you for your recent purchase. Your order will arrive in 2-3 days."
        
        # Act
        result = engine.analyze_content(legitimate_content, "text")
        
        # Assert
        assert result.threat_detected is False
        assert result.threat_type == "safe"
        assert result.confidence_score < 0.3
    
    def test_analyze_empty_content_raises_error(self, engine):
        """Test that empty content raises ValueError"""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            engine.analyze_content("", "text")
    
    def test_analyze_unsupported_type_raises_error(self, engine):
        """Test that unsupported content type raises ValueError"""
        with pytest.raises(ValueError, match="Unsupported content type"):
            engine.analyze_content("test content", "unsupported_type")
    
    @pytest.mark.parametrize("content_type,content", [
        ("text", "Sample text content"),
        ("image", "path/to/image.jpg"),
        ("audio", "path/to/audio.wav")
    ])
    def test_analyze_supported_content_types(self, engine, content_type, content):
        """Test analysis of all supported content types"""
        result = engine.analyze_content(content, content_type)
        assert isinstance(result, ThreatAnalysis)
        assert hasattr(result, 'threat_detected')
        assert hasattr(result, 'confidence_score')
```

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Write tests** for all new functionality
3. **Ensure all tests pass** and coverage remains high
4. **Follow code style** guidelines and run linting
5. **Update documentation** as needed
6. **Submit pull request** with detailed description

### Development Workflow

```bash
# Set up development environment
git clone https://github.com/734ai/SentinelGem.git
cd SentinelGem
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Create feature branch
git checkout -b feature/new-threat-detection

# Make changes and run tests
python -m pytest tests/ -v --cov=src

# Check code style
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# Commit and push
git add .
git commit -m "Add new threat detection capability"
git push origin feature/new-threat-detection

# Create pull request on GitHub
```

---

## Advanced Topics

### Custom Model Integration

```python
class CustomThreatModel(BaseModel):
    """Custom threat detection model"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
    
    def load(self) -> None:
        """Load custom model"""
        # Implementation depends on model type
        # Could be TensorFlow, PyTorch, ONNX, etc.
        pass
    
    def analyze(self, input_data: Any) -> Dict[str, Any]:
        """Analyze using custom model"""
        # Implement custom analysis logic
        pass

# Register custom model
model_hub.register_model_type("custom_threat", CustomThreatModel)
```

### Plugin Architecture

```python
from abc import ABC, abstractmethod

class ThreatAnalysisPlugin(ABC):
    """Base class for threat analysis plugins"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get supported content types"""
        pass
    
    @abstractmethod
    def analyze(self, content: Any, context: str = "") -> ThreatAnalysis:
        """Perform analysis"""
        pass

class PluginManager:
    """Plugin management system"""
    
    def __init__(self):
        self.plugins: Dict[str, ThreatAnalysisPlugin] = {}
    
    def register_plugin(self, plugin: ThreatAnalysisPlugin):
        """Register a new plugin"""
        self.plugins[plugin.get_name()] = plugin
    
    def get_plugin(self, name: str) -> Optional[ThreatAnalysisPlugin]:
        """Get plugin by name"""
        return self.plugins.get(name)
    
    def get_plugins_for_type(self, content_type: str) -> List[ThreatAnalysisPlugin]:
        """Get all plugins that support a content type"""
        return [
            plugin for plugin in self.plugins.values()
            if content_type in plugin.get_supported_types()
        ]

# Example custom plugin
class EmailHeaderAnalysisPlugin(ThreatAnalysisPlugin):
    """Plugin for analyzing email headers"""
    
    def get_name(self) -> str:
        return "email_header_analyzer"
    
    def get_supported_types(self) -> List[str]:
        return ["email", "text"]
    
    def analyze(self, content: str, context: str = "") -> ThreatAnalysis:
        """Analyze email headers for threats"""
        # Implementation for email header analysis
        pass
```

### Distributed Processing

```python
from celery import Celery
import redis

# Celery configuration
celery_app = Celery(
    'sentinelgem',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def analyze_content_async(content: str, content_type: str, context: str = ""):
    """Asynchronous content analysis task"""
    analyzer = ThreatAnalyzer()
    result = analyzer.analyze_content(content, content_type, context)
    return result.dict()

@celery_app.task
def batch_analyze_async(content_list: List[Dict[str, str]]):
    """Asynchronous batch analysis task"""
    results = []
    analyzer = ThreatAnalyzer()
    
    for item in content_list:
        result = analyzer.analyze_content(
            item['content'], 
            item.get('type', 'text'), 
            item.get('context', '')
        )
        results.append(result.dict())
    
    return results

# Usage
def submit_analysis_job(content: str, content_type: str):
    """Submit analysis job to queue"""
    task = analyze_content_async.delay(content, content_type)
    return task.id

def get_analysis_result(task_id: str):
    """Get analysis result by task ID"""
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return result.get()
    else:
        return {"status": "pending"}
```

---

This developer guide provides comprehensive information for contributing to and extending SentinelGem. For specific questions or advanced use cases, please refer to the API documentation or contact the development team.
