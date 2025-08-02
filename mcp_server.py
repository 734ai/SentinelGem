#!/usr/bin/env python3
"""
SentinelGem MCP Server for Kaggle Integration
Provides tools for automatic project deployment to Kaggle
"""

import json
import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Sequence

# Add the project root to the Python path
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

def check_kaggle_auth() -> str:
    """Check Kaggle authentication"""
    try:
        # Check if kaggle is installed
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            return "❌ Kaggle CLI not installed. Run: pip install kaggle"
        
        # Check authentication
        result = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True)
        if result.returncode == 0:
            return f"✅ Kaggle authentication successful!\n{result.stdout}"
        else:
            return f"❌ Kaggle authentication failed: {result.stderr}"
    
    except Exception as e:
        return f"❌ Error checking Kaggle auth: {str(e)}"

def prepare_kaggle_notebook(notebook_title: str = "SentinelGem: Offline Multimodal Cybersecurity Assistant") -> str:
    """Prepare notebook for Kaggle deployment"""
    try:
        project_root = Path(PROJECT_ROOT)
        kaggle_dir = project_root / "kaggle_deploy"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Create kernel metadata
        metadata = {
            "id": "muzansano/sentinelgem",
            "title": notebook_title,
            "code_file": "main.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "false",
            "enable_gpu": "false",
            "enable_internet": "false",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        metadata_file = kaggle_dir / "kernel-metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Copy essential files
        essential_files = [
            "main.py", "requirements.txt", "README.md",
            "src", "agents", "config", "assets"
        ]
        
        copied_files = []
        for item in essential_files:
            src_path = project_root / item
            if src_path.exists():
                if src_path.is_file():
                    dest_path = kaggle_dir / item
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(item)
                elif src_path.is_dir():
                    dest_path = kaggle_dir / item
                    import shutil
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(src_path, dest_path)
                    copied_files.append(f"{item}/")
        
        return f"✅ Kaggle notebook prepared successfully!\n" \
               f"📁 Deploy directory: {kaggle_dir}\n" \
               f"📄 Files copied: {', '.join(copied_files)}\n" \
               f"📋 Metadata created: kernel-metadata.json"
    
    except Exception as e:
        return f"❌ Error preparing Kaggle notebook: {str(e)}"

def upload_to_kaggle(version_notes: str = "Updated SentinelGem project") -> str:
    """Upload to Kaggle"""
    try:
        project_root = Path(PROJECT_ROOT)
        kaggle_dir = project_root / "kaggle_deploy"
        
        if not kaggle_dir.exists():
            return "❌ Kaggle deploy directory not found. Run prepare_kaggle_notebook first."
        
        # Change to kaggle directory and push
        original_cwd = os.getcwd()
        os.chdir(kaggle_dir)
        
        try:
            result = subprocess.run([
                "kaggle", "kernels", "push", 
                "-p", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return f"✅ Successfully uploaded to Kaggle!\n{result.stdout}"
            else:
                return f"❌ Upload failed: {result.stderr}"
        
        finally:
            os.chdir(original_cwd)
    
    except Exception as e:
        return f"❌ Error uploading to Kaggle: {str(e)}"

def commit_to_github(commit_message: str = "Update SentinelGem project for production") -> str:
    """Commit to GitHub"""
    try:
        project_root = Path(PROJECT_ROOT)
        os.chdir(project_root)
        
        # Add all files (respecting .gitignore)
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        
        # Commit
        result = subprocess.run([
            "git", "commit", "-m", commit_message
        ], capture_output=True, text=True)
        
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            return f"❌ Commit failed: {result.stderr}"
        
        # Push to origin
        push_result = subprocess.run([
            "git", "push", "origin", "main"
        ], capture_output=True, text=True)
        
        if push_result.returncode == 0:
            return f"✅ Successfully committed and pushed to GitHub!\n" \
                   f"Commit: {commit_message}\n" \
                   f"{push_result.stdout}"
        else:
            return f"⚠️ Committed locally but push failed: {push_result.stderr}"
    
    except Exception as e:
        return f"❌ Error committing to GitHub: {str(e)}"

def validate_project() -> str:
    """Validate project structure"""
    project_root = Path(PROJECT_ROOT)
    issues = []
    
    # Check required files
    required_files = {
        "main.py": "Main entry point",
        "requirements.txt": "Python dependencies",
        "README.md": "Project documentation",
        "setup.py": "Package configuration",
        ".gitignore": "Git ignore rules",
        "src/inference.py": "Core inference engine",
        "agents/agent_loop.py": "Agent orchestrator",
    }
    
    for file_path, description in required_files.items():
        full_path = project_root / file_path
        if not full_path.exists():
            issues.append(f"❌ Missing {file_path} ({description})")
    
    # Check directory structure
    required_dirs = ["src", "agents", "notebooks", "assets", "config", "tests"]
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            issues.append(f"❌ Missing directory: {dir_name}/")
    
    if issues:
        return f"❌ Project validation failed:\n" + "\n".join(issues)
    else:
        return "✅ Project validation passed! Ready for deployment."

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SentinelGem Kaggle MCP Server")
    parser.add_argument("command", choices=[
        "check-auth", "prepare", "upload", "commit", "validate", "deploy-all"
    ], help="Command to execute")
    parser.add_argument("--message", "-m", default="Update SentinelGem project", 
                       help="Commit/version message")
    parser.add_argument("--title", "-t", default="SentinelGem: Offline Multimodal Cybersecurity Assistant",
                       help="Notebook title")
    
    args = parser.parse_args()
    
    if args.command == "check-auth":
        print(check_kaggle_auth())
    elif args.command == "prepare":
        print(prepare_kaggle_notebook(args.title))
    elif args.command == "upload":
        print(upload_to_kaggle(args.message))
    elif args.command == "commit":
        print(commit_to_github(args.message))
    elif args.command == "validate":
        print(validate_project())
    elif args.command == "deploy-all":
        print("🚀 Starting full deployment process...\n")
        
        print("1. Validating project...")
        validation = validate_project()
        print(validation)
        if "❌" in validation:
            print("❌ Deployment stopped due to validation errors")
            return
        
        print("\n2. Preparing Kaggle notebook...")
        prepare_result = prepare_kaggle_notebook(args.title)
        print(prepare_result)
        if "❌" in prepare_result:
            print("❌ Deployment stopped due to preparation errors")
            return
        
        print("\n3. Uploading to Kaggle...")
        upload_result = upload_to_kaggle(args.message)
        print(upload_result)
        
        print("\n4. Committing to GitHub...")
        commit_result = commit_to_github(args.message)
        print(commit_result)
        
        print("\n🎉 Deployment process completed!")

if __name__ == "__main__":
    main()
