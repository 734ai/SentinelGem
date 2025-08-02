"""
SentinelGem Agent Loop
Author: Muzan Sano

Main orchestration agent for multimodal threat analysis
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference import get_inference_engine, ThreatAnalysis
from src.ocr_pipeline import get_ocr_pipeline
from src.audio_pipeline import get_audio_pipeline
from src.utils import Timer, validate_input_file, logger
from src.autogen_notebook import NotebookGenerator

console = Console()

class SentinelAgent:
    """
    Main SentinelGem agent for orchestrating multimodal threat analysis
    """
    
    def __init__(
        self,
        verbose: bool = False,
        auto_generate_notebooks: bool = True,
        confidence_threshold: float = 0.7
    ):
        self.verbose = verbose
        self.auto_generate_notebooks = auto_generate_notebooks
        self.confidence_threshold = confidence_threshold
        
        # Analysis session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Initialize components
        self.inference_engine = None
        self.ocr_pipeline = None
        self.audio_pipeline = None
        self.notebook_generator = None
        
        self._initialize_components()
        
        logger.info(f"SentinelGem Agent initialized - Session: {self.session_id}")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        try:
            console.print("[blue]Initializing SentinelGem Agent components...[/blue]")
            
            # Initialize inference engine
            self.inference_engine = get_inference_engine()
            
            # Initialize OCR pipeline
            self.ocr_pipeline = get_ocr_pipeline()
            
            # Initialize audio pipeline
            self.audio_pipeline = get_audio_pipeline()
            
            # Initialize notebook generator
            if self.auto_generate_notebooks:
                self.notebook_generator = NotebookGenerator()
            
            console.print("[green]✓ All components initialized successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to initialize components: {e}[/red]")
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def detect_input_type(self, file_path: str) -> str:
        """
        Auto-detect input type based on file extension
        
        Args:
            file_path: Path to input file
            
        Returns:
            Detected input type
        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
        # Audio extensions
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        # Text/log extensions
        text_extensions = {'.txt', '.log', '.csv', '.json', '.xml'}
        
        if extension in image_extensions:
            return "screenshot"
        elif extension in audio_extensions:
            return "audio"
        elif extension in text_extensions:
            return "logs"
        else:
            return "text"  # Default fallback
    
    def analyze_input(
        self, 
        input_file: str, 
        input_type: Optional[str] = None
    ) -> ThreatAnalysis:
        """
        Analyze input using appropriate pipeline
        
        Args:
            input_file: Path to input file
            input_type: Type of input (auto-detected if None)
            
        Returns:
            ThreatAnalysis results
        """
        try:
            # Auto-detect input type if not provided
            if input_type is None:
                input_type = self.detect_input_type(input_file)
            
            console.print(f"[cyan]Analyzing {input_type} input: {Path(input_file).name}[/cyan]")
            
            # Route to appropriate pipeline
            with Timer(f"{input_type.capitalize()} analysis"):
                if input_type == "screenshot":
                    result = self.ocr_pipeline.analyze_screenshot(input_file)
                elif input_type == "audio":
                    result = self.audio_pipeline.analyze_audio(input_file)
                elif input_type in ["logs", "text"]:
                    # Read text content
                    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    result = self.inference_engine.analyze_threat(
                        content, 
                        analysis_type="log_analysis" if input_type == "logs" else "general_analysis",
                        input_type=input_type
                    )
                else:
                    raise ValueError(f"Unsupported input type: {input_type}")
            
            # Record analysis in history
            analysis_record = {
                "timestamp": datetime.now(),
                "input_file": input_file,
                "input_type": input_type,
                "result": result,
                "session_id": self.session_id
            }
            self.analysis_history.append(analysis_record)
            
            # Display results
            self._display_analysis_result(result, input_file, input_type)
            
            # Auto-generate notebook if enabled
            if self.auto_generate_notebooks and self.notebook_generator:
                try:
                    notebook_path = self.notebook_generator.generate_analysis_notebook(
                        analysis_record
                    )
                    console.print(f"[dim]📓 Analysis notebook saved: {notebook_path}[/dim]")
                except Exception as e:
                    logger.warning(f"Notebook generation failed: {e}")
            
            return result
            
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            logger.error(f"Input analysis failed for {input_file}: {e}")
            
            # Return error result
            return ThreatAnalysis(
                threat_detected=False,
                confidence_score=0.0,
                threat_type="analysis_error",
                description=f"Analysis failed: {str(e)}",
                recommendations=["Check input file and try again"],
                raw_analysis="",
                metadata={"error": str(e), "input_file": input_file}
            )
    
    def _display_analysis_result(
        self, 
        result: ThreatAnalysis, 
        input_file: str, 
        input_type: str
    ):
        """Display analysis results in formatted output"""
        
        # Create status panel
        status_color = "red" if result.threat_detected else "green"
        status_text = "🚨 THREAT DETECTED" if result.threat_detected else "✅ NO THREAT"
        
        # Main result panel
        result_text = Text()
        result_text.append(f"{status_text}\n", style=f"bold {status_color}")
        result_text.append(f"File: {Path(input_file).name}\n", style="cyan")
        result_text.append(f"Type: {input_type.capitalize()}\n", style="blue")
        result_text.append(f"Confidence: {result.confidence_score:.2%}\n", style="yellow")
        result_text.append(f"Threat Type: {result.threat_type}\n", style="magenta")
        
        console.print(Panel(result_text, title="[bold]Analysis Result[/bold]", border_style=status_color))
        
        # Description
        if result.description:
            console.print(Panel(result.description, title="[bold]Description[/bold]", border_style="blue"))
        
        # Recommendations
        if result.recommendations:
            rec_text = "\n".join(f"• {rec}" for rec in result.recommendations)
            console.print(Panel(rec_text, title="[bold]Recommendations[/bold]", border_style="yellow"))
        
        # Verbose details
        if self.verbose and result.metadata:
            console.print("\n[dim]--- Verbose Details ---[/dim]")
            for key, value in result.metadata.items():
                if isinstance(value, dict):
                    console.print(f"[dim]{key}:[/dim] {len(value)} items")
                else:
                    console.print(f"[dim]{key}:[/dim] {value}")
    
    def interactive_mode(self):
        """Run agent in interactive mode"""
        console.print(Panel(
            Text("🛡️ SentinelGem Interactive Mode\n", style="bold cyan") +
            Text("Enter file paths to analyze, or 'help' for commands", style="dim"),
            title="[bold]Ready for Analysis[/bold]",
            border_style="cyan"
        ))
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask(
                    "\n[cyan]Enter file path (or command)[/cyan]",
                    default="help"
                ).strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'history':
                    self._show_history()
                elif user_input.lower() == 'stats':
                    self._show_stats()
                elif user_input.lower() == 'clear':
                    console.clear()
                elif user_input.startswith('batch '):
                    # Batch analysis
                    pattern = user_input[6:].strip()
                    self._batch_analyze(pattern)
                else:
                    # Analyze file
                    if os.path.exists(user_input):
                        self.analyze_input(user_input)
                    else:
                        console.print(f"[red]File not found: {user_input}[/red]")
                        
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit SentinelGem?[/yellow]"):
                    break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print("[green]SentinelGem session ended[/green]")
    
    def _show_help(self):
        """Display help information"""
        help_table = Table(title="SentinelGem Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("/path/to/file", "Analyze specified file")
        help_table.add_row("batch pattern", "Analyze files matching pattern")
        help_table.add_row("history", "Show analysis history")
        help_table.add_row("stats", "Show session statistics")
        help_table.add_row("clear", "Clear screen")
        help_table.add_row("help", "Show this help")
        help_table.add_row("exit/quit/q", "Exit SentinelGem")
        
        console.print(help_table)
    
    def _show_history(self):
        """Display analysis history"""
        if not self.analysis_history:
            console.print("[yellow]No analysis history available[/yellow]")
            return
        
        history_table = Table(title=f"Analysis History - Session {self.session_id}")
        history_table.add_column("Time", style="cyan")
        history_table.add_column("File", style="white")
        history_table.add_column("Type", style="blue")
        history_table.add_column("Threat", style="red")
        history_table.add_column("Confidence", style="yellow")
        
        for record in self.analysis_history[-10:]:  # Show last 10
            result = record["result"]
            history_table.add_row(
                record["timestamp"].strftime("%H:%M:%S"),
                Path(record["input_file"]).name,
                record["input_type"],
                "Yes" if result.threat_detected else "No",
                f"{result.confidence_score:.1%}"
            )
        
        console.print(history_table)
    
    def _show_stats(self):
        """Display session statistics"""
        if not self.analysis_history:
            console.print("[yellow]No analysis performed yet[/yellow]")
            return
        
        # Calculate stats
        total_analyses = len(self.analysis_history)
        threats_detected = sum(1 for record in self.analysis_history if record["result"].threat_detected)
        
        # Group by input type
        type_counts = {}
        for record in self.analysis_history:
            input_type = record["input_type"]
            type_counts[input_type] = type_counts.get(input_type, 0) + 1
        
        # Create stats display
        stats_text = Text()
        stats_text.append(f"Session ID: {self.session_id}\n", style="cyan")
        stats_text.append(f"Total Analyses: {total_analyses}\n", style="white")
        stats_text.append(f"Threats Detected: {threats_detected}\n", style="red")
        stats_text.append(f"Threat Rate: {threats_detected/total_analyses:.1%}\n", style="yellow")
        
        if type_counts:
            stats_text.append("\nInput Types:\n", style="blue")
            for input_type, count in type_counts.items():
                stats_text.append(f"  {input_type}: {count}\n", style="dim")
        
        console.print(Panel(stats_text, title="[bold]Session Statistics[/bold]", border_style="green"))
    
    def _batch_analyze(self, pattern: str):
        """Analyze multiple files matching pattern"""
        try:
            from glob import glob
            files = glob(pattern)
            
            if not files:
                console.print(f"[yellow]No files found matching: {pattern}[/yellow]")
                return
            
            console.print(f"[blue]Found {len(files)} files to analyze...[/blue]")
            
            threat_count = 0
            for file_path in files:
                result = self.analyze_input(file_path)
                if result.threat_detected:
                    threat_count += 1
                console.print()  # Add spacing
            
            console.print(f"[green]Batch analysis complete: {threat_count}/{len(files)} threats detected[/green]")
            
        except Exception as e:
            console.print(f"[red]Batch analysis failed: {e}[/red]")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        if not self.analysis_history:
            return {"session_id": self.session_id, "analyses": 0}
        
        total_analyses = len(self.analysis_history)
        threats_detected = sum(1 for record in self.analysis_history if record["result"].threat_detected)
        
        # Group by input type
        type_counts = {}
        threat_types = {}
        
        for record in self.analysis_history:
            input_type = record["input_type"]
            type_counts[input_type] = type_counts.get(input_type, 0) + 1
            
            threat_type = record["result"].threat_type
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_analyses": total_analyses,
            "threats_detected": threats_detected,
            "threat_rate": threats_detected / total_analyses if total_analyses > 0 else 0,
            "input_type_counts": type_counts,
            "threat_type_counts": threat_types,
            "start_time": self.analysis_history[0]["timestamp"] if self.analysis_history else None,
            "end_time": self.analysis_history[-1]["timestamp"] if self.analysis_history else None
        }

if __name__ == "__main__":
    # Test agent
    console.print("[bold cyan]Testing SentinelGem Agent[/bold cyan]")
    
    # Initialize agent
    agent = SentinelAgent(verbose=True)
    
    # Start interactive mode
    agent.interactive_mode()
