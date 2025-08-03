#!/usr/bin/env python3
"""
SentinelGem - Hugging Face Authentication Setup
Author: Muzan Sano

Helper script to set up Hugging Face authentication for Gemma access
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, whoami
from rich.console import Console
from rich.prompt import Prompt

console = Console()

def main():
    console.print("[bold blue]🔑 SentinelGem - Hugging Face Authentication Setup[/bold blue]")
    console.print("=" * 60)
    
    # Check current auth status
    try:
        user_info = whoami()
        console.print(f"[green]✅ Already authenticated as: {user_info['name']}[/green]")
        
        # Test Gemma access
        console.print("\n[blue]Testing Gemma model access...[/blue]")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            console.print("[green]✅ Gemma model access confirmed![/green]")
            console.print("[bold green]🚀 Ready to run SentinelGem with Gemma 3n![/bold green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Gemma access denied: {e}[/red]")
            console.print("[yellow]⚠️  You may need to request access at: https://huggingface.co/google/gemma-2-2b-it[/yellow]")
            return False
            
    except Exception:
        console.print("[yellow]⚠️  Not currently authenticated[/yellow]")
    
    console.print("\n[blue]Authentication required for Gemma access[/blue]")
    console.print("Since you mentioned email approval, you need to authenticate with your token.")
    
    # Check for existing token
    hf_home = Path.home() / ".cache" / "huggingface"
    token_file = hf_home / "token"
    
    if token_file.exists():
        console.print(f"[green]Found existing token file: {token_file}[/green]")
        try:
            with open(token_file) as f:
                token = f.read().strip()
            login(token)
            console.print("[green]✅ Successfully authenticated with existing token![/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Existing token failed: {e}[/red]")
    
    # Interactive setup
    console.print("\n[bold]Setup Options:[/bold]")
    console.print("1. 🌐 Open https://huggingface.co/settings/tokens to get your token")
    console.print("2. 📋 Copy your token")
    console.print("3. 🔐 Paste it when prompted below")
    
    token = Prompt.ask("\n[blue]Enter your Hugging Face token[/blue]", password=True)
    
    if not token or len(token) < 20:
        console.print("[red]❌ Invalid token. Tokens are typically 40+ characters.[/red]")
        return False
    
    try:
        login(token)
        console.print("[green]✅ Authentication successful![/green]")
        
        # Save token for future use
        hf_home.mkdir(parents=True, exist_ok=True)
        with open(token_file, 'w') as f:
            f.write(token)
        console.print(f"[green]💾 Token saved to: {token_file}[/green]")
        
        # Test Gemma access
        console.print("\n[blue]Testing Gemma model access...[/blue]")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        console.print("[green]✅ Gemma model access confirmed![/green]")
        console.print("[bold green]🚀 SentinelGem is ready for Gemma 3n integration![/bold green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Authentication failed: {e}[/red]")
        console.print("[yellow]💡 Make sure you have access to the Gemma model and your token has 'Read' permissions[/yellow]")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
