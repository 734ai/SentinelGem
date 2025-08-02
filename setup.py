#!/usr/bin/env python3
"""
SentinelGem Setup Script
Author: Muzan Sano
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() 
                   if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

setup(
    name="sentinelgem",
    version="1.0.0",
    author="Muzan Sano",
    author_email="sanosensei36@gmail.com",
    description="Offline Multimodal Cybersecurity Assistant powered by Gemma 3n",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/734ai/SentinelGem",
    project_urls={
        "Bug Tracker": "https://github.com/734ai/SentinelGem/issues",
        "Documentation": "https://github.com/734ai/SentinelGem/tree/main/docs",
        "Source Code": "https://github.com/734ai/SentinelGem",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "ui": [
            "streamlit>=1.20.0",
            "plotly>=5.10.0",
            "gradio>=3.20.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sentinelgem=main:main",
            "sentinelgem-agent=agents.agent_loop:main",  
        ],
    },
    include_package_data=True,
    package_data={
        "sentinelgem": [
            "config/*.json",
            "config/*.yaml", 
            "assets/*.txt",
            "assets/*.wav",
            "assets/README.md",
        ],
    },
    keywords=[
        "cybersecurity", 
        "ai", 
        "machine-learning", 
        "threat-detection",
        "multimodal", 
        "gemma", 
        "privacy", 
        "offline",
        "phishing-detection",
        "malware-analysis"
    ],
    zip_safe=False,
)
