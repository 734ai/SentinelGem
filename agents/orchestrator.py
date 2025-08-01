#!/usr/bin/env python3
"""
SentinelGem Agent Orchestrator
Author: Muzan Sano

Advanced orchestration system for coordinating multimodal AI analysis,
planning input delegation, and managing agent workflows.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import SentinelGem components
from .inference import ThreatAnalysis, get_inference_engine
from ..agents.agent_loop import SentinelAgent
from .ocr_pipeline import OCRPipeline
from .audio_pipeline import AudioPipeline
from .log_parser import LogParser
from .autogen_notebook import NotebookGenerator
from .utils import Timer

console = Console()
logger = logging.getLogger(__name__)

class InputType(Enum):
    """Supported input types for analysis"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    LOGS = "logs"
    URL = "url"
    EMAIL = "email"
    UNKNOWN = "unknown"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AnalysisTask:
    """Represents an analysis task"""
    task_id: str
    input_type: InputType
    input_data: str
    priority: Priority
    created_at: datetime
    metadata: Dict[str, Any]
    status: str = "pending"
    result: Optional[ThreatAnalysis] = None
    processing_time: Optional[float] = None

@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    max_concurrent_tasks: int = 3
    enable_auto_notebook: bool = True
    batch_processing: bool = False
    threat_threshold: float = 0.7
    priority_boost_threshold: float = 0.9

class SentinelOrchestrator:
    """
    Advanced orchestrator for coordinating multimodal threat analysis
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # Initialize components
        self.inference_engine = get_inference_engine()
        self.ocr_pipeline = OCRPipeline()
        self.audio_pipeline = AudioPipeline()
        self.log_parser = LogParser()
        self.notebook_generator = NotebookGenerator()
        
        # Task management
        self.pending_tasks: List[AnalysisTask] = []
        self.active_tasks: Dict[str, AnalysisTask] = {}
        self.completed_tasks: List[AnalysisTask] = []
        
        # Session management
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_stats = {
            'tasks_processed': 0,
            'threats_detected': 0,
            'total_processing_time': 0.0,
            'start_time': datetime.now()
        }
        
        logger.info(f"SentinelOrchestrator initialized - Session: {self.session_id}")
    
    def detect_input_type(self, input_data: str, metadata: Dict[str, Any] = None) -> InputType:
        """
        Automatically detect the type of input data
        
        Args:
            input_data: Path to file or direct text content
            metadata: Additional metadata about the input
            
        Returns:
            Detected input type
        """
        if not input_data:
            return InputType.UNKNOWN
        
        # Check if it's a file path
        if isinstance(input_data, str) and len(input_data) < 500:
            input_path = Path(input_data)
            
            # File-based detection
            if input_path.exists():
                suffix = input_path.suffix.lower()
                
                # Image files
                if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                    return InputType.IMAGE
                
                # Audio files
                elif suffix in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
                    return InputType.AUDIO
                
                # Log files
                elif suffix in ['.log', '.txt'] or 'log' in input_path.name.lower():
                    # Check content to confirm it's a log file
                    try:
                        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                            sample = f.read(1000)
                            if self._looks_like_log_content(sample):
                                return InputType.LOGS
                    except:
                        pass
                
                # Default to text for other files
                return InputType.TEXT
        
        # Content-based detection for non-file inputs
        if isinstance(input_data, str):
            # URL detection
            if input_data.startswith(('http://', 'https://', 'ftp://')):
                return InputType.URL
            
            # Email detection (basic patterns)
            if any(keyword in input_data.lower() for keyword in ['subject:', 'from:', 'to:', '@']):
                if len(input_data.split('\\n')) > 3:  # Multi-line email-like
                    return InputType.EMAIL
            
            # Log-like content
            if self._looks_like_log_content(input_data):
                return InputType.LOGS
            
            # Default to text
            return InputType.TEXT
        
        return InputType.UNKNOWN
    
    def _looks_like_log_content(self, content: str) -> bool:
        """Check if content looks like log data"""
        lines = content.split('\\n')[:10]  # Check first 10 lines
        
        log_indicators = 0
        for line in lines:
            if not line.strip():
                continue
                
            # Check for timestamp patterns
            if any(pattern in line for pattern in [':', '-', '/', '[', ']']):
                log_indicators += 1
            
            # Check for log levels
            if any(level in line.upper() for level in ['INFO', 'ERROR', 'WARN', 'DEBUG', 'FATAL']):
                log_indicators += 1
        
        return log_indicators >= 3
    
    def calculate_priority(self, input_type: InputType, metadata: Dict[str, Any] = None) -> Priority:
        """
        Calculate task priority based on input type and metadata
        
        Args:
            input_type: Type of input
            metadata: Additional context
            
        Returns:
            Calculated priority level
        """
        # Base priority by input type
        base_priorities = {
            InputType.LOGS: Priority.HIGH,      # System logs are high priority
            InputType.EMAIL: Priority.HIGH,     # Email threats are common
            InputType.IMAGE: Priority.MEDIUM,   # Screenshots need analysis
            InputType.URL: Priority.MEDIUM,     # URLs can be malicious
            InputType.AUDIO: Priority.MEDIUM,   # Social engineering
            InputType.TEXT: Priority.LOW,       # General text analysis
            InputType.UNKNOWN: Priority.LOW
        }
        
        priority = base_priorities.get(input_type, Priority.LOW)
        
        # Boost priority based on metadata
        if metadata:
            # High-risk keywords
            high_risk_keywords = [
                'urgent', 'suspicious', 'malware', 'phishing', 'attack',
                'breach', 'compromised', 'threat', 'alert', 'critical'
            ]
            
            content_str = str(metadata).lower()
            if any(keyword in content_str for keyword in high_risk_keywords):
                priority = Priority.CRITICAL
            
            # User-specified priority
            if 'priority' in metadata:
                try:
                    user_priority = Priority(metadata['priority'])
                    priority = max(priority, user_priority)
                except:
                    pass
        
        return priority
    
    def create_task(self, 
                   input_data: str, 
                   input_type: Optional[InputType] = None,
                   priority: Optional[Priority] = None,
                   metadata: Dict[str, Any] = None) -> AnalysisTask:
        """
        Create a new analysis task
        
        Args:
            input_data: Input data or file path
            input_type: Type of input (auto-detected if None)
            priority: Task priority (auto-calculated if None)
            metadata: Additional task metadata
            
        Returns:
            Created analysis task
        """
        # Auto-detect input type if not provided
        if input_type is None:
            input_type = self.detect_input_type(input_data, metadata)
        
        # Auto-calculate priority if not provided
        if priority is None:
            priority = self.calculate_priority(input_type, metadata)
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        task = AnalysisTask(
            task_id=task_id,
            input_type=input_type,
            input_data=input_data,
            priority=priority,
            created_at=datetime.now(),
            metadata=metadata or {},
            status="pending"
        )
        
        self.pending_tasks.append(task)
        
        # Sort by priority (critical first)
        self.pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        console.print(f"[blue]📋 Created task {task_id} ({input_type.value}, {priority.name})[/blue]")
        logger.info(f"Task created: {task_id} - {input_type.value} - {priority.name}")
        
        return task
    
    async def process_task(self, task: AnalysisTask) -> ThreatAnalysis:
        """
        Process a single analysis task
        
        Args:
            task: Task to process
            
        Returns:
            Analysis results
        """
        console.print(f"[yellow]🔄 Processing task {task.task_id} ({task.input_type.value})[/yellow]")
        
        with Timer(f"Task {task.task_id}") as timer:
            try:
                task.status = "processing"
                
                # Route to appropriate pipeline
                if task.input_type == InputType.IMAGE:
                    result = self.ocr_pipeline.analyze_screenshot(task.input_data)
                
                elif task.input_type == InputType.AUDIO:
                    result = self.audio_pipeline.analyze_audio(task.input_data)
                
                elif task.input_type == InputType.LOGS:
                    log_result = self.log_parser.analyze_log_file(task.input_data)
                    result = log_result.ai_analysis
                    
                    # Enhance with log-specific metadata
                    result.metadata.update({
                        'total_entries': log_result.total_entries,
                        'suspicious_entries': len(log_result.suspicious_entries),
                        'threat_indicators': log_result.threat_indicators
                    })
                
                elif task.input_type in [InputType.TEXT, InputType.EMAIL, InputType.URL]:
                    # Read content if it's a file
                    content = task.input_data
                    if Path(task.input_data).exists():
                        with open(task.input_data, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    
                    result = self.inference_engine.analyze_threat(
                        content,
                        analysis_type=f"{task.input_type.value}_analysis",
                        input_type=task.input_type.value
                    )
                
                else:
                    # Fallback to basic text analysis
                    result = self.inference_engine.analyze_threat(
                        str(task.input_data),
                        analysis_type="general_analysis",
                        input_type="unknown"
                    )
                
                task.status = "completed"
                task.result = result
                task.processing_time = timer.elapsed
                
                # Update session stats
                self.session_stats['tasks_processed'] += 1
                self.session_stats['total_processing_time'] += timer.elapsed
                if result.threat_detected:
                    self.session_stats['threats_detected'] += 1
                
                console.print(f"[green]✅ Task {task.task_id} completed - Threat: {result.threat_detected}[/green]")
                
                return result
                
            except Exception as e:
                task.status = "failed"
                error_result = ThreatAnalysis(
                    threat_detected=False,
                    confidence_score=0.0,
                    threat_type="processing_error",
                    description=f"Task processing failed: {str(e)}",
                    recommendations=["Review input data and try again"],
                    raw_analysis="",
                    metadata={"error": str(e), "task_id": task.task_id}
                )
                task.result = error_result
                task.processing_time = timer.elapsed
                
                console.print(f"[red]❌ Task {task.task_id} failed: {str(e)}[/red]")
                logger.error(f"Task processing failed: {task.task_id} - {str(e)}")
                
                return error_result
    
    async def process_all_tasks(self) -> List[ThreatAnalysis]:
        """
        Process all pending tasks with concurrency control
        
        Returns:
            List of analysis results
        """
        if not self.pending_tasks:
            console.print("[yellow]No pending tasks to process[/yellow]")
            return []
        
        console.print(f"[blue]🚀 Processing {len(self.pending_tasks)} tasks (max concurrent: {self.config.max_concurrent_tasks})[/blue]")
        
        results = []
        
        # Process tasks in batches with concurrency limit
        while self.pending_tasks:
            # Take up to max_concurrent_tasks
            current_batch = []
            for _ in range(min(self.config.max_concurrent_tasks, len(self.pending_tasks))):
                if self.pending_tasks:
                    task = self.pending_tasks.pop(0)
                    self.active_tasks[task.task_id] = task
                    current_batch.append(task)
            
            if not current_batch:
                break
            
            # Process batch concurrently
            batch_tasks = [self.process_task(task) for task in current_batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for i, result in enumerate(batch_results):
                task = current_batch[i]
                
                if isinstance(result, Exception):
                    console.print(f"[red]❌ Task {task.task_id} exception: {result}[/red]")
                    result = ThreatAnalysis(
                        threat_detected=False,
                        confidence_score=0.0,
                        threat_type="exception",
                        description=f"Task exception: {str(result)}",
                        recommendations=["Check logs and retry"],
                        raw_analysis="",
                        metadata={"exception": str(result)}
                    )
                
                results.append(result)
                
                # Move task to completed
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)
        
        console.print(f"[green]✅ All tasks completed: {len(results)} results[/green]")
        
        # Generate notebook if enabled
        if self.config.enable_auto_notebook and results:
            await self.generate_session_notebook(results)
        
        return results
    
    async def generate_session_notebook(self, results: List[ThreatAnalysis]):
        """Generate analysis notebook for the session"""
        try:
            console.print("[blue]📓 Generating session analysis notebook...[/blue]")
            
            # Prepare analysis records
            analysis_records = []
            for i, (task, result) in enumerate(zip(self.completed_tasks, results)):
                analysis_records.append({
                    'timestamp': task.created_at,
                    'input_file': task.input_data,
                    'input_type': task.input_type.value,
                    'session_id': self.session_id,
                    'result': result,
                    'processing_time': task.processing_time,
                    'priority': task.priority.name
                })
            
            # Generate notebook
            notebook_path = await asyncio.to_thread(
                self.notebook_generator.generate_analysis_notebook,
                analysis_records,
                f"orchestrator_session_{self.session_id}"
            )
            
            console.print(f"[green]✅ Session notebook generated: {notebook_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Notebook generation failed: {e}[/red]")
            logger.error(f"Notebook generation failed: {e}")
    
    def add_batch_inputs(self, input_list: List[Dict[str, Any]]) -> List[AnalysisTask]:
        """
        Add multiple inputs for batch processing
        
        Args:
            input_list: List of input specifications
            
        Returns:
            List of created tasks
        """
        tasks = []
        
        for input_spec in input_list:
            task = self.create_task(
                input_data=input_spec.get('input_data'),
                input_type=InputType(input_spec.get('input_type')) if input_spec.get('input_type') else None,
                priority=Priority(input_spec.get('priority')) if input_spec.get('priority') else None,
                metadata=input_spec.get('metadata', {})
            )
            tasks.append(task)
        
        console.print(f"[blue]📦 Added {len(tasks)} tasks for batch processing[/blue]")
        return tasks
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        current_time = datetime.now()
        session_duration = (current_time - self.session_stats['start_time']).total_seconds()
        
        return {
            'session_id': self.session_id,
            'session_duration_seconds': session_duration,
            'tasks_processed': self.session_stats['tasks_processed'],
            'threats_detected': self.session_stats['threats_detected'],
            'total_processing_time': self.session_stats['total_processing_time'],
            'pending_tasks': len(self.pending_tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'threat_detection_rate': (
                self.session_stats['threats_detected'] / max(1, self.session_stats['tasks_processed'])
            ),
            'avg_processing_time': (
                self.session_stats['total_processing_time'] / max(1, self.session_stats['tasks_processed'])
            )
        }
    
    def export_session_report(self, output_file: str):
        """Export comprehensive session report"""
        try:
            report_data = {
                'session_info': {
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'config': asdict(self.config)
                },
                'statistics': self.get_session_stats(),
                'completed_tasks': []
            }
            
            # Add completed task details
            for task in self.completed_tasks:
                task_data = {
                    'task_id': task.task_id,
                    'input_type': task.input_type.value,
                    'priority': task.priority.name,
                    'created_at': task.created_at.isoformat(),
                    'status': task.status,
                    'processing_time': task.processing_time,
                    'metadata': task.metadata
                }
                
                if task.result:
                    task_data['result'] = {
                        'threat_detected': task.result.threat_detected,
                        'confidence_score': task.result.confidence_score,
                        'threat_type': task.result.threat_type,
                        'description': task.result.description,
                        'recommendations': task.result.recommendations
                    }
                
                report_data['completed_tasks'].append(task_data)
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            console.print(f"[green]✅ Session report exported: {output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Report export failed: {e}[/red]")
            logger.error(f"Report export failed: {e}")

# Convenience functions
async def orchestrate_analysis(inputs: List[str], **kwargs) -> List[ThreatAnalysis]:
    """Convenience function for orchestrated analysis"""
    orchestrator = SentinelOrchestrator()
    
    # Create tasks
    for input_data in inputs:
        orchestrator.create_task(input_data, **kwargs)
    
    # Process all tasks
    return await orchestrator.process_all_tasks()

async def analyze_directory(directory_path: str, **kwargs) -> List[ThreatAnalysis]:
    """Analyze all files in a directory"""
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Find all analyzable files
    file_patterns = ['*.txt', '*.log', '*.jpg', '*.png', '*.wav', '*.mp3']
    files = []
    for pattern in file_patterns:
        files.extend(directory.glob(pattern))
    
    file_paths = [str(f) for f in files]
    return await orchestrate_analysis(file_paths, **kwargs)

if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            # Process command line arguments
            inputs = sys.argv[1:]
            results = await orchestrate_analysis(inputs)
            
            print(f"\n=== ORCHESTRATOR RESULTS ===")
            for i, result in enumerate(results):
                print(f"\nInput {i+1}: {result.threat_type}")
                print(f"Threat detected: {result.threat_detected}")
                print(f"Confidence: {result.confidence_score:.2f}")
                print(f"Description: {result.description}")
        else:
            print("Usage: python orchestrator.py <input1> [input2] ...")
    
    asyncio.run(main())
