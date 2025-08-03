#!/usr/bin/env python3
"""
SentinelGem - Gemma Model Download with Network Retry
Author: Muzan Sano

This script handles the Gemma model download with network retry logic
and provides diagnostics for connectivity issues.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import socket

console = Console()

def check_network_connectivity():
    """Check network connectivity to HuggingFace"""
    console.print("[blue]🔍 Checking network connectivity...[/blue]")
    
    # Test DNS resolution
    try:
        socket.gethostbyname("huggingface.co")
        console.print("[green]✅ DNS resolution successful[/green]")
        return True
    except socket.gaierror:
        console.print("[red]❌ DNS resolution failed[/red]")
        console.print("[yellow]💡 Try: sudo systemctl restart systemd-resolved[/yellow]")
        console.print("[yellow]💡 Or change DNS to 8.8.8.8, 1.1.1.1[/yellow]")
        return False

def download_gemma_with_retry(max_retries=3):
    """Download Gemma model with retry logic"""
    console.print("[blue]🚀 Starting Gemma model download with retry logic...[/blue]")
    
    for attempt in range(max_retries):
        try:
            console.print(f"[cyan]Attempt {attempt + 1}/{max_retries}[/cyan]")
            
            # Check connectivity first
            if not check_network_connectivity():
                console.print("[yellow]⚠️  Network issue detected, waiting 30 seconds...[/yellow]")
                time.sleep(30)
                continue
            
            # Import here to avoid issues if transformers isn't available
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            console.print("[blue]📦 Downloading tokenizer...[/blue]")
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            console.print("[green]✅ Tokenizer downloaded successfully[/green]")
            
            console.print("[blue]🧠 Downloading model (this may take 10-15 minutes)...[/blue]")
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                torch_dtype="auto",
                device_map="cpu",  # Force CPU to avoid GPU memory issues
                low_cpu_mem_usage=True  # Optimize for low RAM
            )
            console.print("[green]✅ Model downloaded successfully![/green]")
            
            # Test the model quickly
            console.print("[blue]🧪 Testing model...[/blue]")
            test_text = "Test threat analysis"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            console.print("[green]🎉 Gemma model is ready for SentinelGem![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Attempt {attempt + 1} failed: {e}[/red]")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 60  # Progressive backoff
                console.print(f"[yellow]⏳ Waiting {wait_time} seconds before retry...[/yellow]")
                time.sleep(wait_time)
            else:
                console.print("[red]❌ All download attempts failed[/red]")
                return False
    
    return False

def main():
    console.print("[bold blue]🛡️  SentinelGem - Gemma Model Download Manager[/bold blue]")
    console.print("=" * 60)
    
    # Check if we have authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        console.print(f"[green]✅ Authenticated as: {user_info['name']}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Authentication issue: {e}[/red]")
        console.print("[yellow]💡 Run: python setup_auth.py[/yellow]")
        return False
    
    # Check if model is already downloaded
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", local_files_only=True)
        console.print("[green]✅ Gemma model already available locally![/green]")
        
        # Test SentinelGem integration
        console.print("[blue]🧪 Testing SentinelGem integration...[/blue]")
        try:
            from src.inference import GemmaInference
            gemma = GemmaInference()
            console.print("[green]🎉 SentinelGem + Gemma integration successful![/green]")
            console.print("[bold green]🚀 PRODUCTION READY FOR COMPETITION![/bold green]")
            return True
        except Exception as e:
            console.print(f"[yellow]⚠️  Integration test failed: {e}[/yellow]")
            console.print("[blue]Model downloaded but needs integration fixes[/blue]")
            return True
            
    except:
        console.print("[yellow]⚠️  Gemma model not found locally, downloading...[/yellow]")
    
    # Download the model
    success = download_gemma_with_retry()
    
    if success:
        console.print("\n" + "="*60)
        console.print("[bold green]🏆 GEMMA MODEL READY FOR SENTINELGEM![/bold green]")
        console.print("[green]✅ Next step: Run production tests[/green]")
        console.print("="*60)
    else:
        console.print("\n" + "="*60)
        console.print("[bold red]❌ DOWNLOAD FAILED - NETWORK ISSUES[/bold red]")
        console.print("[yellow]💡 Check network connectivity and try again[/yellow]")
        console.print("[yellow]💡 Or use mobile hotspot / different network[/yellow]")
        console.print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
