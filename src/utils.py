"""
SentinelGem Utilities
Author: Muzan Sano

Common utilities for file handling, logging, and system operations
"""

import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import hashlib
import mimetypes

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup rich logging configuration"""
    
    # Configure logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("sentinelgem")
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_file.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                console.print(f"[red]Unsupported config format: {config_file.suffix}[/red]")
                return {}
                
    except Exception as e:
        console.print(f"[red]Error loading config {config_path}: {e}[/red]")
        return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to YAML or JSON file"""
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == '.json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                console.print(f"[red]Unsupported config format: {config_file.suffix}[/red]")
                return False
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error saving config {config_path}: {e}[/red]")
        return False

def get_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate file hash for integrity checking"""
    try:
        hash_algo = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()
        
    except Exception as e:
        console.print(f"[red]Error calculating hash for {file_path}: {e}[/red]")
        return ""

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    try:
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return {"exists": False}
        
        stat = file_path_obj.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path_obj))
        
        return {
            "exists": True,
            "name": file_path_obj.name,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "mime_type": mime_type,
            "extension": file_path_obj.suffix,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "hash": get_file_hash(file_path)
        }
        
    except Exception as e:
        console.print(f"[red]Error getting file info for {file_path}: {e}[/red]")
        return {"exists": False, "error": str(e)}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage"""
    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename

def create_timestamp_filename(prefix: str = "", suffix: str = "", extension: str = "") -> str:
    """Create filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = [p for p in [prefix, timestamp, suffix] if p]
    filename = "_".join(parts)
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    return filename + extension

def ensure_directory(directory: str) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        console.print(f"[red]Error creating directory {directory}: {e}[/red]")
        return False

def list_files_by_pattern(directory: str, pattern: str = "*") -> List[str]:
    """List files in directory matching pattern"""
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return []
        
        return [str(f) for f in directory_path.glob(pattern) if f.is_file()]
        
    except Exception as e:
        console.print(f"[red]Error listing files in {directory}: {e}[/red]")
        return []

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def display_file_table(files: List[str], title: str = "Files") -> None:
    """Display files in a rich table"""
    table = Table(title=title)
    table.add_column("File", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Modified", style="blue")
    
    for file_path in files:
        info = get_file_info(file_path)
        if info.get("exists"):
            table.add_row(
                Path(file_path).name,
                format_file_size(info["size"]),
                info.get("mime_type", "unknown"),
                info["modified"].strftime("%Y-%m-%d %H:%M")
            )
    
    console.print(table)

def validate_input_file(file_path: str, max_size_mb: int = 50) -> Dict[str, Any]:
    """Validate input file for processing"""
    info = get_file_info(file_path)
    
    if not info.get("exists"):
        return {
            "valid": False,
            "error": "File does not exist",
            "info": info
        }
    
    if info["size_mb"] > max_size_mb:
        return {
            "valid": False,
            "error": f"File too large: {info['size_mb']}MB > {max_size_mb}MB",
            "info": info
        }
    
    # Check for common malicious file types
    dangerous_extensions = ['.exe', '.scr', '.bat', '.cmd', '.com', '.pif']
    if info.get("extension", "").lower() in dangerous_extensions:
        return {
            "valid": False,
            "error": f"Potentially dangerous file type: {info['extension']}",
            "info": info
        }
    
    return {
        "valid": True,
        "info": info
    }

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text using simple regex"""
    import re
    
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    
    return list(set(urls))  # Remove duplicates

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        console.print(f"[dim]{self.description} completed in {duration:.2f}s[/dim]")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

# Global logger instance
logger = setup_logging()

if __name__ == "__main__":
    # Test utilities
    console.print("[bold cyan]Testing SentinelGem Utilities[/bold cyan]")
    
    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(1)
    
    # Test file operations
    test_files = list_files_by_pattern(".", "*.py")
    console.print(f"Found {len(test_files)} Python files")
    
    if test_files:
        display_file_table(test_files[:5], "Sample Python Files")
