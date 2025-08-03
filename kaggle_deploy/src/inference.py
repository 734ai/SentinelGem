"""
SentinelGem Inference Engine
Author: Muzan Sano

Core Gemma 3n model interface for multimodal threat analysis
"""

import os
import torch
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class ThreatAnalysis:
    """Data structure for threat analysis results"""
    threat_detected: bool
    confidence_score: float
    threat_type: str
    description: str
    recommendations: List[str]
    raw_analysis: str
    metadata: Dict[str, Any]

class GemmaInference:
    """
    Gemma 3n inference engine for cybersecurity threat analysis
    """
    
    def __init__(
        self, 
        model_path: str = "google/gemma-2-2b-it",
        device: str = "auto",
        quantization: bool = True,
        max_length: int = 2048
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.quantization = quantization
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Threat analysis prompts
        self.prompts = self._load_prompts()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemma 3n model with optimal settings"""
        console.print("[blue]Loading Gemma 3n model...[/blue]")
        
        try:
            # Configure quantization for efficient inference
            bnb_config = None
            if self.quantization:
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, disabling quantization")
                    self.quantization = False
                except Exception as e:
                    logger.warning(f"Quantization setup failed: {e}, disabling quantization")
                    self.quantization = False
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            console.print(f"[green]✓ Gemma 3n loaded successfully[/green]")
            logger.info(f"Model loaded: {self.model_path}")
            
        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load cybersecurity analysis prompts"""
        return {
            "phishing_analysis": """
You are a cybersecurity expert analyzing potential phishing attempts. 
Analyze the following content and determine if it's malicious:

CONTENT: {content}

Provide analysis in this format:
THREAT_DETECTED: [YES/NO]
CONFIDENCE: [0.0-1.0]
THREAT_TYPE: [phishing/spear-phishing/credential-harvesting/none]
DESCRIPTION: [Brief explanation]
RECOMMENDATIONS: [Actionable advice]
""",
            
            "log_analysis": """
You are a SOC analyst reviewing system logs for suspicious activity.
Analyze these logs for potential threats using MITRE ATT&CK framework:

LOGS: {content}

Provide analysis:
THREAT_DETECTED: [YES/NO]
CONFIDENCE: [0.0-1.0]
THREAT_TYPE: [malware/persistence/lateral-movement/exfiltration/none]
MITRE_TACTICS: [List relevant tactics]
DESCRIPTION: [Technical analysis]
RECOMMENDATIONS: [Response actions]
""",
            
            "audio_analysis": """
You are analyzing audio transcription for security anomalies.
Look for social engineering, unauthorized access attempts, or suspicious conversations:

TRANSCRIPT: {content}

Analysis format:
THREAT_DETECTED: [YES/NO]
CONFIDENCE: [0.0-1.0]
THREAT_TYPE: [social-engineering/eavesdropping/unauthorized-access/none]
DESCRIPTION: [What was detected]
RECOMMENDATIONS: [Security measures]
""",
            
            "general_analysis": """
As a cybersecurity expert, analyze this content for any security threats:

CONTENT: {content}
INPUT_TYPE: {input_type}

Provide comprehensive security analysis:
THREAT_DETECTED: [YES/NO]
CONFIDENCE: [0.0-1.0]
THREAT_TYPE: [specific threat category]
DESCRIPTION: [Detailed analysis]
RECOMMENDATIONS: [Security recommendations]
"""
        }
    
    def analyze_threat(
        self, 
        content: str, 
        analysis_type: str = "general",
        input_type: str = "text"
    ) -> ThreatAnalysis:
        """
        Perform threat analysis using Gemma 3n
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis (phishing, log, audio, general)
            input_type: Type of input (text, image, audio, log)
        
        Returns:
            ThreatAnalysis object with results
        """
        try:
            # Select appropriate prompt
            prompt_template = self.prompts.get(f"{analysis_type}_analysis", self.prompts["general_analysis"])
            
            # Format prompt with content
            if "{input_type}" in prompt_template:
                prompt = prompt_template.format(content=content, input_type=input_type)
            else:
                prompt = prompt_template.format(content=content)
            
            console.print(f"[yellow]Analyzing {input_type} content...[/yellow]")
            
            # Generate analysis
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0]["generated_text"]
                # Remove the original prompt from response
                analysis_text = generated_text[len(prompt):].strip()
            else:
                analysis_text = str(response)
            
            # Parse analysis results
            result = self._parse_analysis(analysis_text)
            
            logger.info(f"Threat analysis completed: {result.threat_type}")
            return result
            
        except Exception as e:
            console.print(f"[red]Analysis error: {e}[/red]")
            logger.error(f"Threat analysis failed: {e}")
            
            # Return safe fallback
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.0,
                threat_type="error",
                description=f"Analysis failed: {str(e)}",
                recommendations=["Review input and try again"],
                raw_analysis="",
                metadata={"error": str(e)}
            )
    
    def _parse_analysis(self, analysis_text: str) -> ThreatAnalysis:
        """Parse Gemma 3n analysis output into structured format"""
        try:
            lines = analysis_text.split('\n')
            
            # Default values
            threat_detected = False
            confidence_score = 0.0
            threat_type = "unknown"
            description = ""
            recommendations = []
            
            # Parse structured output
            for line in lines:
                line = line.strip()
                if line.startswith("THREAT_DETECTED:"):
                    threat_detected = "YES" in line.upper()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence_score = float(line.split(":")[1].strip())
                    except:
                        confidence_score = 0.5
                elif line.startswith("THREAT_TYPE:"):
                    threat_type = line.split(":", 1)[1].strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()
                elif line.startswith("RECOMMENDATIONS:"):
                    rec_text = line.split(":", 1)[1].strip()
                    recommendations = [rec_text] if rec_text else []
            
            # If no structured output found, use entire text as description
            if not description and analysis_text:
                description = analysis_text[:500]  # Truncate if too long
            
            return ThreatAnalysis(
                threat_detected=threat_detected,
                confidence_score=confidence_score,
                threat_type=threat_type,
                description=description,
                recommendations=recommendations,
                raw_analysis=analysis_text,
                metadata={
                    "model": self.model_path,
                    "analysis_length": len(analysis_text)
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis parsing failed: {e}")
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.0,
                threat_type="parse_error",
                description=f"Failed to parse analysis: {str(e)}",
                recommendations=["Check analysis format"],
                raw_analysis=analysis_text,
                metadata={"parse_error": str(e)}
            )
    
    def batch_analyze(self, contents: List[str], analysis_type: str = "general") -> List[ThreatAnalysis]:
        """Analyze multiple inputs in batch"""
        results = []
        
        for content in track(contents, description="Analyzing batch..."):
            result = self.analyze_threat(content, analysis_type)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_path": self.model_path,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "quantization": self.quantization,
            "max_length": self.max_length,
            "tokenizer_vocab_size": len(self.tokenizer.vocab) if self.tokenizer else 0,
            "status": "loaded"
        }

# Global inference instance (singleton pattern)
_inference_instance = None

def get_inference_engine(**kwargs) -> GemmaInference:
    """Get or create global inference engine instance"""
    global _inference_instance
    
    if _inference_instance is None:
        _inference_instance = GemmaInference(**kwargs)
    
    return _inference_instance

if __name__ == "__main__":
    # Test the inference engine
    console.print("[bold cyan]Testing SentinelGem Inference Engine[/bold cyan]")
    
    # Initialize engine
    engine = GemmaInference()
    
    # Test analysis
    test_content = "Click here to verify your account: http://suspicious-bank-site.com/login"
    result = engine.analyze_threat(test_content, "phishing_analysis")
    
    console.print(f"[green]Test Result:[/green]")
    console.print(f"Threat Detected: {result.threat_detected}")
    console.print(f"Confidence: {result.confidence_score}")
    console.print(f"Type: {result.threat_type}")
    console.print(f"Description: {result.description}")
