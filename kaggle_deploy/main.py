#!/usr/bin/env python3
"""
SentinelGem: Offline Multimodal Cybersecurity Assistant
Author: Muzan Sano
Entry point for the SentinelGem system
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def display_banner():
    """Display SentinelGem startup banner"""
    banner = Text()
    banner.append("🛡️  SentinelGem v1.0\n", style="bold cyan")
    banner.append("Offline Multimodal Cybersecurity Assistant\n", style="cyan")
    banner.append("Powered by Gemma 3n • Author: Muzan Sano\n", style="dim")
    
    console.print(Panel(banner, title="[bold]Initializing...[/bold]", border_style="cyan"))

def main():
    """Main entry point for SentinelGem"""
    parser = argparse.ArgumentParser(
        description="SentinelGem: Offline Multimodal Cybersecurity Assistant"
    )
    parser.add_argument(
        "--mode", 
        choices=["agent", "notebook", "ui", "test"],
        default="agent",
        help="Operating mode (default: agent)"
    )
    parser.add_argument(
        "--input-type",
        choices=["screenshot", "audio", "logs", "text"],
        help="Input type for analysis"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    display_banner()
    
    try:
        if args.mode == "agent":
            from agents.agent_loop import SentinelAgent
            console.print("[green]Starting SentinelGem Agent...[/green]")
            agent = SentinelAgent(verbose=args.verbose)
            
            if args.input_file:
                result = agent.analyze_input(args.input_file, args.input_type)
                console.print(f"[cyan]Analysis Result:[/cyan] {result}")
            else:
                agent.interactive_mode()
                
        elif args.mode == "notebook":
            console.print("[blue]Launching Jupyter environment...[/blue]")
            os.system("jupyter notebook notebooks/00_bootstrap.ipynb")
            
        elif args.mode == "ui":
            console.print("[magenta]Starting Streamlit UI...[/magenta]")
            os.system("streamlit run ui/app.py")
            
        elif args.mode == "test":
            console.print("[yellow]Running test suite...[/yellow]")
            os.system("python -m pytest tests/ -v")
            
    except KeyboardInterrupt:
        console.print("\n[red]SentinelGem interrupted by user[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
